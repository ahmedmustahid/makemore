
import torch


class BatchNorm1d:
    def __init__(self, num_features, eps=1e-05, momentum=0.1, training=True):
        self.eps = eps
        self.momentum = momentum

        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        self.training = training

        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
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

