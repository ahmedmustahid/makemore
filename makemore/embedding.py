import torch

class Embedding:
    def __init__(self, num_embeddings: int, embedding_size: int):
        self.weight = torch.randn((num_embeddings, embedding_size))


    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]


if __name__=="__main__":
    emb = Embedding(5, 10)

    x = torch.arange(3)
    print(emb(x))