import torch

class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)

        if x.shape[1]==1:
            x = torch.squeeze(x, dim=1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []



if __name__=="__main__":
    e = torch.randn(4, 8, 20)
    f = FlattenConsecutive(2)
    e = f(e)

    print(e.shape)