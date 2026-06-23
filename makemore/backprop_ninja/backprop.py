import torch
import torch.nn.functional as F
import random


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


def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')


def init_params(n_embd, n_hidden, vocab_size, block_size):
    C = torch.randn((vocab_size, n_embd))
    W1 = torch.randn((n_embd * block_size, n_hidden))
    b1 = torch.randn(n_hidden)
    W2 = torch.randn(n_hidden, vocab_size)
    b2 = torch.randn(vocab_size)

    bngain = torch.randn((1, n_hidden))*0.1 + 1.0
    bnbias = torch.randn((1, n_hidden))*0.1

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    for p in parameters:
       p.requires_grad = True

    return parameters
   

def main():
    block_size = 3
    n_embd = 10
    n_hidden = 64

    #extract and preprocess dataset
    words = open('/home/yerongyang/ml/kaparthy/makemore/names.txt', 'r').read().splitlines()
    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(itos)

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    #intiialize tr, val and test sets
    Xtr,  Ytr  = build_dataset(words[:n1], block_size, stoi)     # 80%
    Xdev, Ydev = build_dataset(words[n1:n2], block_size, stoi)   # 10%
    Xte,  Yte  = build_dataset(words[n2:], block_size, stoi)     # 10%

    #initialize params
    parameters = init_params(n_embd, n_hidden, vocab_size, block_size)
    C, W1, b1, W2, b2, bngain, bnbias = parameters

    batch_size = 32
    n = batch_size 
    # construct a minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
    
    #forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer 1
    hprebn = embcat @ W1 + b1 # hidden layer pre-activation
    # BatchNorm layer
    bnmeani = 1/n*hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani #bnmeani is broadcasted, so hprebn needs to be summed when backprop
    bndiff2 = bndiff**2
    bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
    bnvar_inv = (bnvar + 1e-5)**-0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    # Linear layer 2
    logits = h @ W2 + b2 # output layer
    # cross entropy loss (same as F.cross_entropy(logits, Yb))
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes # subtract max for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    #torch back pass
    for p in parameters:
        p.grad = None
        for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
                norm_logits, logit_maxes, logits, h, hpreact, bnraw,
                bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
                embcat, emb]:
            t.retain_grad()
    loss.backward()

    # Exercise 1: backprop through the whole thing manually, 
    # backpropagating through exactly all of the variables 
    # as they are defined in the forward pass above, one by one
    #soft max layer loss
    logprobs_diff = torch.zeros_like(logprobs)
    logprobs_diff[range(batch_size), Yb] = -1.0 / batch_size
    probs_diff = logprobs_diff / probs
    counts_diff = probs_diff * counts_sum_inv
    counts_sum_inv_diff = (probs_diff * counts).sum(1, keepdims=True)
    counts_sum_diff = -counts_sum_inv_diff * counts_sum**-2
    counts_diff += counts_sum_diff * 1.0
    norm_logits_diff = counts_diff * counts
    logits_diff = norm_logits_diff * 1.0
    logit_maxes_diff = (-1 * norm_logits_diff).sum(1, keepdim=True)
    logits_diff += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * logit_maxes_diff
    
    #logits shape is (32, 27), W2 shape is (64, 27) b2 size is (27, ), h is (32, 64), 
    b2_loss = logits_diff.sum(0) * 1.0
    W2_loss = h.T @ logits_diff
    h_loss = logits_diff @ W2.T #(32, 64)
    hpreact_loss = h_loss * (1 - torch.tanh(hpreact)**2) #(32, 64)

    #batch norm
    bngain_loss = (hpreact_loss * bnraw).sum(0) #(64,)
    bnbias_loss = hpreact_loss.sum(0) #(64, )
    bnraw_loss = hpreact_loss * bngain  #(32, 64)
    bndiff_loss = bnraw_loss * bnvar_inv #(32, 64)
    bnvar_inv_loss = (bnraw_loss * bndiff).sum(0, keepdim=True) #(1, 64)
    bnvar_loss = bnvar_inv_loss * -0.5 * (bnvar + 1e-5)**-1.5 #(1, 64)
    bndiff2_loss = bnvar_loss * 1/(n-1) #(1, 64)
    bndiff_loss += bndiff2_loss * 2 * bndiff#(32, 64)
    hprebn_loss = bndiff_loss.clone() #(32, 64)
    bnmeani_loss = (-1.0 * bndiff_loss).sum(0, keepdim=True) #(1, 64)
    hprebn_loss += bnmeani_loss * 1/n #(32, 64)

    #linear layer
    embcat_loss = hprebn_loss @ W1.T
    W1_loss = embcat.T @ hprebn_loss
    emb_loss = embcat_loss.view(emb.shape) #(32, 3, 10)
    b1_loss = hprebn_loss.sum(0)
    C_loss = torch.zeros_like(C) #(27, 10)

    #update C_loss with emb_loss
    for k in range (Xb.shape[0]):
       for j in range(Xb.shape[1]):
          ix = Xb[k, j]
          C_loss[ix] += emb_loss[k, j]
    cmp('logprobs', logprobs_diff, logprobs)
    cmp('probs', probs_diff, probs)
    cmp('counts_sum_inv', counts_sum_inv_diff, counts_sum_inv)
    cmp('counts_sum', counts_sum_diff, counts_sum)
    cmp('counts', counts_diff, counts)
    cmp('norm_logits', norm_logits_diff, norm_logits)
    cmp('logit_maxes', logit_maxes_diff, logit_maxes)
    cmp('logits', logits_diff, logits)
    cmp('h', h_loss, h)
    cmp('W2', W2_loss, W2)
    cmp('b2', b2_loss, b2)
    cmp('hpreact', hpreact_loss, hpreact)
    cmp('bngain', bngain_loss, bngain)
    cmp('bnbias', bnbias_loss, bnbias)
    cmp('bnraw', bnraw_loss, bnraw)
    cmp('bnvar_inv', bnvar_inv_loss, bnvar_inv)
    cmp('bnvar', bnvar_loss, bnvar)
    cmp('bndiff2', bndiff2_loss, bndiff2)
    cmp('bndiff', bndiff_loss, bndiff)
    cmp('bnmeani', bnmeani_loss, bnmeani)
    cmp('hprebn', hprebn_loss, hprebn)
    cmp('embcat', embcat_loss, embcat)
    cmp('W1', W1_loss, W1)
    cmp('b1', b1_loss, b1)
    cmp('emb', emb_loss, emb)
    cmp('C', C_loss, C)

    # Exercise 2: backprop through cross_entropy but all in one go
    # to complete this challenge look at the mathematical expression of the loss,
    # take the derivative, simplify the expression, and just write it out
    probs = torch.softmax(logits, dim = 1)
    logits_diff = probs - F.one_hot(Yb, num_classes=logits.shape[1]) / batch_size

    #exericse 3
    hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias





if __name__ == "__main__":
   main()