import torch
import torch.nn as nn
from torch.nn import functional as F



BATCH_SIZE = 32
BLOCK_SIZE = 8
device = "cuda" if torch.cuda.is_available() else 'cpu'
lr = 1e-3
epochs = 3000
eval_interval = 5000
n_emb = 32
dropout = 0.2
n_layer = 8

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        #attention
        weight = q @ k.transpose(-2, -1) * self.head_size**-0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        #pass forward info
        v = self.value(x)
        out = weight @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_emb // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x



class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size, n_emb):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head=4) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #extracts last BLOCK_SIZE characters to do inference
        for _ in range(max_new_tokens):
            #passes through current idx
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            #extracts the last character in each input batch
            logits = logits[:, -1, :]
            #finds softmax and samples the output
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            #concatenate output along the first dimension (i.e. the index)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

@torch.no_grad()
def estimate_loss(model, eval_iters, train, val):
    #run test on train and val datasets
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train, val)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split, train, val):
    #create batch based on train/val split
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y


def main():
    #read in and preprocess data
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = { ch:i for i, ch in enumerate(chars)}
    itos = { i:ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    #create train val ds
    data = torch.tensor(encode(text), dtype=torch.long).to(device)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    #run the bigram model
    m = BigramLanguageModel(vocab_size, n_emb)
    m = m.to(device)
    idx = torch.zeros((1, 1), dtype=torch.long)

    #declare a new PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for epoch in range(50000):
        optimizer.zero_grad()
        if epoch % eval_interval == 0:
            losses = estimate_loss(m, 10, train_data, val_data)
            print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train', train_data, val_data)

        logits, loss = m(xb, yb)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"epoch {epoch}: {loss.item()}")


if __name__ == "__main__":
    main()