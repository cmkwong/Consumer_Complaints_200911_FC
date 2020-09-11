from lib import data, models, criterions, common
import torch
from torch import optim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import re

# get the now time
now = datetime.now()
dt_string = now.strftime("%y%m%d_%H%M%S")

DATA_PATH = "../data/Consumer_Complaints.csv"
MAIN_PATH = "../docs/1"
NET_SAVE_PATH = MAIN_PATH + '/checkpoint'
RUNS_SAVE_PATH = MAIN_PATH + "/runs/" + dt_string
NET_FILE = "checkpoint-10000.data"
LOAD_NET = False
TRAIN_ON_GPU = True
BATCH_SIZE = 512
lr = 0.0001
CHECKPOINT_STEP = 50000
PRINT_EVERY = 5000
SCALAR_VISUALIZE_EVERY = 500
EMBEDDING_VISUALIZE_EVERY = 50000
MOVING_AVERAGE_STEP = 1000


# read file
col_names, raw_data = data.read_csv(path=DATA_PATH)

# define batch generator
domain_col = raw_data[7]
codomain_col = raw_data[3]
batch_generator = data.Batch_Generator()
generator_prepare = batch_generator.prepare_generator(domain_col=domain_col, codomain_col=codomain_col)

# define model
skip_gram_model = models.SkipGram(len(generator_prepare.domain_int2vocab), len(generator_prepare.codomain_int2vocab),
                                  embedding_size=3, train_on_gpu=TRAIN_ON_GPU)
if LOAD_NET:
    print("Loading net params...")
    with open(os.path.join(NET_SAVE_PATH, NET_FILE), "rb") as f:
        checkpoint = torch.load(f)
    skip_gram_model.load_state_dict(checkpoint['state_dict'])
    step_idx = int(re.match('checkpoint-([\d]*)', NET_FILE).group(1))
    print("Successful.")
else:
    step_idx = 0

# optimizer
optimizer = optim.Adam(skip_gram_model.parameters(), lr=lr)

# define criterion
criterion = criterions.NegativeSamplingLoss()

writer = SummaryWriter(log_dir=RUNS_SAVE_PATH, comment="USDJPY_2020")

# training
sg_train_set = None
epochs = 0
accumulate_loss = 0
average_loss = 0
while True:
    if epochs % 5 == 0:
        sg_train_set = batch_generator.create_sg_batches(40, generator_prepare.category, generator_prepare.noise_dist)

    for x, y, neg_y in batch_generator.get_sg_batches(BATCH_SIZE, sg_train_set):

        # input, output, and noise vectors
        input_vectors = skip_gram_model.forward_input(x)
        output_vectors = skip_gram_model.forward_output(y)
        noise_vectors = skip_gram_model.forward_noise(neg_y)

        # loss
        loss = criterion(input_vectors, output_vectors, noise_vectors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # moving average loss
        accumulate_loss += loss.item()
        if step_idx % MOVING_AVERAGE_STEP == 0:
            average_loss = accumulate_loss / MOVING_AVERAGE_STEP
            accumulate_loss = 0

        # loss stats
        if step_idx % PRINT_EVERY == 0:
            print("Epoch: {} - step: {}".format(epochs, step_idx))
            print("Average Loss: ", average_loss)  # avg batch loss at this point in training
            valid_examples, valid_similarities = common.cosine_similarity(skip_gram_model.in_embed, train_on_gpu=TRAIN_ON_GPU)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [generator_prepare.domain_int2vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(generator_prepare.domain_int2vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")

        if step_idx % CHECKPOINT_STEP == 0:
            checkpoint = {
                "state_dict": skip_gram_model.state_dict()
            }
            with open(os.path.join(NET_SAVE_PATH, "checkpoint-%d.data" % step_idx), "wb") as f:
                torch.save(checkpoint, f)
        
        if step_idx % SCALAR_VISUALIZE_EVERY == 0:
            # scalar plot
            loss_value = loss.item()
            writer.add_scalar("loss", loss_value, step_idx)

        if step_idx % EMBEDDING_VISUALIZE_EVERY == 0:
            # embedding plot
            writer.add_embedding(mat=skip_gram_model.in_embed.weight,
                                 metadata=list(generator_prepare.domain_vocab2int.keys()), global_step=step_idx)
        step_idx += 1
    epochs += 1