import torch
import torch.nn.functional as F

#one hot default is torch.int64

def get_tokenizer(words: list[str]):
    characters = sorted(list(set(''.join(words))))

    s_to_i = {s: i+1 for i, s in enumerate(characters)}
    s_to_i['.'] = 0
    i_to_s = {i: s for s, i in s_to_i.items()}

    return [s_to_i, i_to_s]


def load_dataset(words, block_size, s_to_i, i_to_s):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = s_to_i[ch] #i get the index of the character in the unique character list
            X.append(context) #i append the previous running context to the X dataset
            Y.append(ix) #I append the current element to y. this is the target. based on the past 3 
            context = context[1:] + [ix] #i slice the context and add the current word for the rolling window

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return [X, Y]

 
def create_embedding_space():
    C = torch.randn((27, 2), requires_grad=True) #each unique character will have a two dimensional embedding
    return C


if __name__ == "__main__":
    words = open('names.txt', 'r').read().splitlines()
    s_to_i, i_to_s = get_tokenizer(words)
    X, Y = load_dataset(words, 3, s_to_i, i_to_s)
    #weights initialized are 2d e.g. 6 input channels, 100 output channels
    #the emb dimensions 32 * 3 * 2 do not fit the 6 * 100 weight size
    #to solve this, we cat the 32 * 3 * 2 embeddings - > 32 * 6 

    #torch.cat(emb[:, 0, :], emb[:, 1, :], emb[: 2, :]) # what this does is extract all rows and embeddings for each index, and concatenates them
    #hardcoded. however, we can unbind a dimension (slice along it), then get back a tuple of the sliced tensors
    #torch.cat(torch.unbind(emb, 1), 1)
    # when i unbind it along dim 1, it now becomes 3 tensors of 32 rows with 2 embeddings. now it becomes 2d where the embedding is the 1st dimesnion
    #so cat will connect back the tensors along that dim 32 * 2 -> 32 * 6
    #or we can use view. tensors are stored as 1d by pytorch. .view changes how this tensor is viewed, so you can specify how the tensor is viewed by the internal code
    #emb.view(32, 6)
    W1 = torch.randn((6, 100), requires_grad=True)
    b1 = torch.randn(100, requires_grad=True)
    W2 = torch.randn((100, 27), requires_grad=True)
    b2 = torch.randn(27, requires_grad=True)
    C = create_embedding_space()
    parameters = [C, W1, b1, W2, b2]
    
    for epoch in range(0, 1000):
        ix = torch.randint(0, X.shape[0], (32, ))
        emb = C[X[ix]] #randomly selects 32 elements from range 0 to number of rows in X (i.e. number of datasets), slices X to find these rows then slices C
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1) #-1 means based on 6, calculate what dimension is required to fit the view.
        logits = h @ W2 + b2
        # counts = logits.exp()
        # prob = counts/counts.sum(dim=1, keepdims=True)
        # loss = -prob[torch.arange(32), Y].log().mean() #goes into each tensor row, then Y extracts the gt probability in the column
        loss = F.cross_entropy(logits, Y[ix]) #remember to index Y also. built in library is better to be called because of numerical stability (if logit big will lead to nan in custom operations) and how torch clusters calculatios
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -0.1 * p.grad
        print(f"Epoch {epoch}: loss {loss.item()}")