import torch
from torch import nn

class SkipGram(nn.Module):
    def __init__(self, n_domain, n_codomain, embedding_size):
        super(SkipGram, self).__init__()

        self.n_domain = n_domain
        self.n_codomain = n_codomain

        self.in_embed = nn.Embedding(n_domain, embedding_size)
        self.out_embed = nn.Embedding(n_codomain, embedding_size)

        self.in_embed.weight.data.uniform_(-1,1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, domains):
        """
        :param domains: [0,1,4,6,2,4,5,6, ...]
        :return:    torch.array(vector_1, vector_2, vector_3, ...)
        """
        input_vectors = self.in_embed(domains)
        return input_vectors

    def forward_output(self, codomains):
        """
        :param codomains: [0,2,3,1,5,6,3, ...]
        :return: torch.array(vector_1, vector_2, vector_3, ...)
        """
        output_vectors = self.out_embed(codomains)
        return output_vectors

    def forward_noise(self, neg_codomains):
        """
        :param neg_codomains: [2,3,7,9,4,6,8, ...]
        :return: torch.array(vector_1, vector_2, vector_3, ...)
        """
        noise_vectors = self.out_embed(neg_codomains)
        return noise_vectors
