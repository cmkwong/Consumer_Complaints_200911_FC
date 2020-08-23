import torch
from lib import common
import os

class SkipGram_Trainer:
    def __init__(self, batch_generator, generator_prepare, model, optimizer, criterion, checkpoint_path):
        self.batch_generator = batch_generator
        self.generator_prepare = generator_prepare
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_path = checkpoint_path
        self.steps = 0

    def train(self, epochs, sampling_sizes, batch_size=64, print_every=5000, checkpoint_step=50000, device="cuda"):

        self.model = self.model.to(device)
        for e in range(epochs):
            for domain, codomain, neg_codomain in self.batch_generator.get_batches(sampling_sizes, batch_size, self.generator_prepare.category, self.generator_prepare.noise_dist):
                domain, codomain, neg_codomain = torch.tensor(domain, dtype=torch.long).to(device), \
                                                 torch.tensor(codomain, dtype=torch.long).to(device),\
                                                 torch.tensor(neg_codomain, dtype=torch.long).to(device)


                # input, output, and noise vectors
                input_vectors = self.model.forward_input(domain)
                output_vectors = self.model.forward_output(codomain)
                noise_vectors = self.model.forward_noise(neg_codomain)

                # loss
                loss = self.criterion(input_vectors, output_vectors, noise_vectors)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss stats
                if self.steps % print_every == 0:
                    print("Epoch: {}/{} - step: {}".format(e + 1, epochs, self.steps))
                    print("Loss: ", loss.item())  # avg batch loss at this point in training
                    valid_examples, valid_similarities = common.cosine_similarity(self.model.in_embed, device=device)
                    _, closest_idxs = valid_similarities.topk(6)

                    valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
                    for ii, valid_idx in enumerate(valid_examples):
                        closest_words = [self.generator_prepare.domain_int2vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                        print(self.generator_prepare.domain_int2vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
                    print("...\n")

                if self.steps % checkpoint_step == 0:
                    checkpoint = {
                        "state_dict": self.model.state_dict()
                    }
                    with open(os.path.join(self.checkpoint_path, "checkpoint-%d.data" % self.steps), "wb") as f:
                        torch.save(checkpoint, f)

                self.steps += 1