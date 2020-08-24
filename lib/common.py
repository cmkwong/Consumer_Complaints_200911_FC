import torch
import numpy as np
import random

def cosine_similarity(embedding, valid_size=16, valid_window=100, train_on_gpu=True):
    if train_on_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    embed_vectors = embedding.weight

    # magnitude of embedding vectors, |b|
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))
    valid_examples = torch.tensor(valid_examples, dtype=torch.long).to(device)

    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes

    return valid_examples, similarities
