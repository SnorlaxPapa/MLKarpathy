import torch
import torch.nn.functional as F


#one hot default is torch.int64

def get_tokenizer(words: list[str]):
    characters = sorted(list(set(''.join(words))))

    s_to_i = {s: i+1 for i, s in enumerate(characters)}
    s_to_i['.'] = 0
    i_to_s = {i: s for s, i in s_to_i.items()}

    return [s_to_i, i_to_s]


def load_dataset(words: List[str], block_size: int = 3, s_to_i, i_to_s):
    X, Y = [], []

    for word in words:
        context = [0] * block_dize
        for ch in w + '.':
            ix = stoi[ch] #i get the index of the character in the unique character list
            X.append(context) #i append the previous running context to the X dataset
            Y.append(ix) #I append the current element to y. this is the target. based on the past 3 
            context = context[1:] + [ix] #i slice the context and add the current word for the rolling window

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return [X, Y]

 
def create_embedding_space():
    C = torch.randn((27, 2)) #each unique character will have a two dimensional embedding
    return C


if __name__ == "__main__":
    words = open('names.txt', 'r').read().splitlines
    s_to_i, i_to_s = get_tokenizer(words)
    X, Y = load_dataset(words, block_size=3, s_to_i, i_to_s)

    C = create_embedding_space()
    # C[X] embedding, 
    # each row in X is basically the context (i.e. the index of the 3 characters)
    # X is 2D
    # so it would be something like
    # [[5, 6, 7] [2, 1, 10]], and accessing this would be a lisst of row aacces as X is contained in an entire tensor?
    # so this accesses the corresponding embedding in C  so it would be like X dim * 2 for shape of C
    emb = C[X]


    #weights initialized are 2d e.g. 6 input channels, 100 output channels
    #the emb dimensions 32 * 3 * 2 do not fit the 6 * 100 weight size
    #to solve this, we cat the 32 * 3 * 2 embeddings - > 32 * 6 

    torch.cat(emb[:, 0, :], emb[:, 1, :], emb[: 2, :]) # what this does is extract all rows and embeddings for each index, and concatenates them
    #hardcoded. however, we can unbind a dimension (slice along it), then get back a tuple of the sliced tensors
    torch.cat(torch.unbind(emb, 1), 1)
    # when i unbind it along dim 1, it now becomes 3 tensors of 32 rows with 2 embeddings. now it becomes 2d where the embedding is the 1st dimesnion
    #so cat will connect back the tensors along that dim 32 * 2 -> 32 * 6
    #or we can use view. tensors are stored as 1d by pytorch. .view changes how this tensor is viewed, so you can specify how the tensor is viewed by the internal code
    emb.view(32, 6)
    W1 = torch.randn((6, 100))
    b1 = torch.randn(100)

    h = emb.view(-1, 6) @ W1 + b1 #-1 means based on 6, calculate what dimension is required to fit the view.

