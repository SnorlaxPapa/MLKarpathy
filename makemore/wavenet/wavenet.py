import torch
import torch.nn.functional as F
import random

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5 # note: kaiming init
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

# -----------------------------------------------------------------------------------------------
class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  

  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True) # batch mean
      xvar = x.var(dim, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  

  def parameters(self):
    return [self.gamma, self.beta]

# -----------------------------------------------------------------------------------------------
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  

  def parameters(self):
    return []
  

class Embedding:
  
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, ix):
      self.out = self.weight[ix]
      return self.out
    
    def parameters(self):
      return [self.weight]
    

class Flatten:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
           x = x.squeeze(1)
        self.out = x
        return self.out
  
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def eval(self):
        for layer in self.layers:
            layer.training = False

    def train(self):
        for layer in self.layers:
            layer.training = True

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def backward(loss, parameters):
    for p in parameters:
      p.grad = None
    loss.backward()


def update(parameters, lr):
    for p in parameters:
      p.data += -lr * p.grad


def build_dataset(words, block_size, stoi):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return [X, Y]


def split_loss(split, Xtr, Ytr, Xdev, Ydev, Xte, Yte, model):
   x, y = {
    'train': (Xtr, Ytr),
    'val': (Xdev, Ydev),
    'test': (Xte, Yte),
   }[split]
   logits = model(x)
   loss = F.cross_entropy(logits, y)
   print(split, loss.item())


def sample(model, block_size, itos):
    out = []
    context = [0] * block_size
    while True:
        #forward pass
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        #sample distribution
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:
           break
    print(''.join(itos[i] for i in out))


def main():
    #extract and preprocess dataset
    n_embd = 24
    n_hidden = 128
    block_size = 8
    max_steps = 200000
    words = open('/home/yerongyang/ml/kaparthy/makemore/names.txt', 'r').read().splitlines()
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)
    Xtr, Ytr = build_dataset(words[:n1], block_size, stoi)
    Xdev, Ydev = build_dataset(words[n1: n2], block_size, stoi)
    Xte, Yte = build_dataset(words[n2:], block_size, stoi)

    #init model
    model = Sequential([
      Embedding(vocab_size, n_embd),
      Flatten(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
      Flatten(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
      Flatten(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(), 
      Linear(n_hidden, vocab_size)
    ])

    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    
    parameters = model.parameters()
    for p in parameters:
      p.requires_grad=True

    batch_size = 32
    for epoch in range(max_steps):
        lr = 0.1 if epoch < 15000 else 0.01

        ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        backward(loss, parameters)
        update(parameters, lr)

        if epoch % 10000 == 0:
            print(f'{epoch:7d}/{max_steps:7d}: {loss.item():.4f}')

    
if __name__ == "__main__":
    main()