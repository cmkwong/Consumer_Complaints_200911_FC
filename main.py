from lib import data, models, trainers, criterions
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# get the now time
now = datetime.now()
dt_string = now.strftime("%y%m%d_%H%M%S")

DATA_PATH = "../data/Consumer_Complaints.csv"
MAIN_PATH = "../docs/1"
NET_SAVE_PATH = MAIN_PATH + '/checkpoint'
RUNS_SAVE_PATH = MAIN_PATH + "/runs/" + dt_string
NET_FILE = "checkpoint-5000000.data"
BATCH_SIZE = 64
lr = 0.00001
CHECKPOINT_STEP = 500000
PRINT_EVERY = 5000

# read file
col_names, raw_data = data.read_csv(path=DATA_PATH)

# define batch generator
domain_col = raw_data[7]
codomain_col = raw_data[3]
batch_generator = data.Batch_Generator()
generator_prepare = batch_generator.prepare_generator(domain_col=domain_col, codomain_col=codomain_col)

# define model
skip_gram_model = models.SkipGram(len(generator_prepare.domain_int2vocab), len(generator_prepare.codomain_int2vocab), embedding_size=3)

# optimizer
optimizer = optim.Adam(skip_gram_model.parameters(), lr=lr)

# define criterion
criterion = criterions.NegativeSamplingLoss()

# define trainer
skip_gram_trainer = trainers.SkipGram_Trainer(batch_generator, generator_prepare, skip_gram_model, optimizer, criterion, checkpoint_path=NET_SAVE_PATH)

# training
skip_gram_trainer.train(epochs=2, sampling_sizes=[20,5], batch_size=64, print_every=5000, checkpoint_step=50000, device="cuda")
