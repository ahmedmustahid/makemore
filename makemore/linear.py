import torch

class Linear:
    def __init__(self, in_features, out_features, bias = True):
        self.weight = torch.randn((in_features, out_features))/(in_features)**0.5
        self.bias = torch.zeros(out_features) if bias else None


    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias
        
        return self.out
    
    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])
    

if __name__=="__main__":

    layer = Linear(1,3)
    x = torch.tensor([2.0])
    print(layer(x))

    print(f"parameters {layer.parameters()}")