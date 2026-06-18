import torch
import torch.nn.functional as F


def create_heat_map(words, s_to_i, i_to_s):
    N = torch.zeros((27, 27), dtype=torch.int32)
    for word in words:
        chs = ["."] + list(word) + ["."]

        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = s_to_i[ch1] #we get the index of the position of the first element, this will correspond to the row in the N heatmap
            ix2 = s_to_i[ch2] #we get the index of the positioin of the second character, this will be the corresponding column in the N heatment
            N[ix1, ix2] += 1 #we update the frequency of this [ch1 ch2] pair in the N heatmap
    return N


def bigram():
    words = open("names.txt", 'r').read().splitlines()
    s_to_i, i_to_s = get_tokenizer(words)
    N = create_heat_map(words, s_to_i, i_to_s)

    ix = 0
    g = torch.Generator().manual_seed(2147483647)
    P = N.float() #P is a 28*28 array with floating point precision fp32
    P /= P.sum(dim=1, keepdim=True) 

    #a psa on how broadcasting works:
    #look from right to left, broadcasting will only work if both dimensions are the same or one of them is a 1. 
    #e.g. 27, 27 and 27, 1 27, 27, 27 and 27, 2 will not work. if a dimension does not exist, a 1 will be added
    #basically what happens is the 1 is "copied" 27 times across that dimension. must be careful on how you want this broadcasting done because if the wrong dimension is copied, could lead to silently wrong ops

    for i in range(50):
        out = []
        while True:
            # p = N[ix].float()
            # p = p/p.sum()

            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(i_to_s[ix])
            if ix == 0: #EOS
                break
        print(''.join(out))


def get_tokenizer(words: list[str]):
    characters = sorted(list(set(''.join(words))))

    s_to_i = {s: i+1 for i, s in enumerate(characters)}
    s_to_i['.'] = 0
    i_to_s = {i: s for s, i in s_to_i.items()}

    return [s_to_i, i_to_s]


def create_dataset(words):
    xs, ys = [], []
    s_to_i, i_to_s = get_tokenizer(words)

    for word in words:
        chs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = s_to_i[ch1] 
            ix2 = s_to_i[ch2] 
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)

    xenc = F.one_hot(xs, num_classes=27).float() #one hot defaults to int64, .float to convert to float32
    return [xenc, ys]


def train_nn(xenc, ys, epochs: int = 10):
    W = torch.randn((27, 27), requires_grad=True) #27 inputs, with 27 outputs (output probabilities) if we initialize a 0 weight, will lead to label smoothing

    for epoch in range(epochs):
        W.grad = None
        print(f"Epoch {epoch}:")
        logits = xenc @ W 
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        
        loss = -probs[torch.arange(xenc.shape[0]), ys].log().mean()
        loss.backward()
        W.data += -30*W.grad

        print(f"Likelihood loss: {loss:4f}")


    


if __name__ == "__main__":
    words = open("names.txt", 'r').read().splitlines()
    xenc, ys = create_dataset(words)
    train_nn(xenc, ys, epochs = 1000)
