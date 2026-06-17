import torch


def bigram():
    words = open("names.txt", 'r').read().splitlines()

    stoi, itos = get_tokenizer(words)
    N = torch.zeros((27, 27), dtype=torch.int32)

    for word in words:
        chs = ["."] + list(word) + ["."]

        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1] #we get the index of the position of the first element, this will correspond to the row in the N heatmap
            ix2 = stoi[ch2] #we get the index of the positioin of the second character, this will be the corresponding column in the N heatment
            N[ix1, ix2] += 1 #we update the frequency of this [ch1 ch2] pair in the N heatmap

    ix = 0
    g = torch.Generator().manual_seed(2147483647)

    for i in range(50):
        out = []
        while True:
            p = N[ix].float()
            p = p/p.sum()
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0: #EOS
                break
        print(''.join(out))



def get_tokenizer(words: list[str]):
    characters = sorted(list(set(''.join(words))))

    s_to_i = {s: i+1 for i, s in enumerate(characters)}
    s_to_i['.'] = 0
    i_to_s = {i: s for s, i in s_to_i.items()}

    return [s_to_i, i_to_s]


if __name__ == "__main__":
    bigram()
