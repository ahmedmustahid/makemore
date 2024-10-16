
import torch


class BatchNorm1d:
    def __init__(self, dim, eps=1e-05, momentum=0.1, training=True):
        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.training = training

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):

        if x.ndim==2:
            dim = 0
        elif x.ndim==3:
            dim = (0,1)
        else: 
            dim = 0
        if self.training:
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var


        xhat = (x - xmean)/torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar


        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    

if __name__=="__main__":

    x = torch.randn(3)

    print(f"{x=}")

    bn = BatchNorm1d(3)
    print("bn: ", bn(x))

    x = torch.randn(3,2)

    print(f"{x=}")

    bn = BatchNorm1d((3,2))
    print("bn: ", bn(x))

