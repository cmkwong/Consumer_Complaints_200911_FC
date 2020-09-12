import torch
from torch import nn

class FC_Embed(nn.Module):
    def __init__(self, n_domain, n_codomain, embedding_size, train_on_gpu):
        super(FC_Embed, self).__init__()

        if train_on_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.embed = nn.Embedding(n_domain, embedding_size).to(self.device)

        # build the fc network
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, n_codomain),
            nn.LogSoftmax(dim=1)
        ).to(self.device)

        # init the weights
        self.embed.weight.data.uniform_(-1,1)

    def forward(self, domains):
        """
        :param domains: [0,1,4,6,2,4,5,6, ...]
        :return:    [0.10,0.15,...,0.12] sum=1 * batch
        """
        domains = torch.tensor(domains, dtype=torch.long).to(self.device)
        embed_vectors = self.embed(domains)
        output = self.fc(embed_vectors)
        return output

