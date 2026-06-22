import torch

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # Scale by fan_in ** 0.5 (Kaiming/Xavier hybrid approach for linear/tanh)
        self.weight = (torch.randn((fan_in, fan_out)) / fan_in ** 0.5).requires_grad_()
        self.bias = torch.zeros(fan_out, requires_grad=True) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        # Learnable parameters (Gradients explicitly turned ON)
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)
        
        # Fixed Buffers (Gradients turned OFF)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=False) # unbiased=False aligns with PyTorch default
        else:
            xmean = self.running_mean
            xvar = self.running_var

        # Standardize
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        # Update buffers during training using history + current batch
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean.flatten()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar.flatten()
        
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
    

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []