import regex
import tiktoken

class BasicTokenizer:
    def __init__(self):
        self.merges = {}
        self.re = regex.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        self.vocab = {idx: bytes([idx]) for idx in range (256)}


    def get_count(self, token_chunks):
        #given a list of split up words, find highest pair count in each word, globally track, then return the sorted count
        count = {}
        for chunk in token_chunks:
            for pair in zip (chunk, chunk[1:]):
                count[pair] = count.get(pair, 0) + 1
        
        return count
    
    
    def replace(self, token_chunks, pair, idx):
        new_ids = []
        for token in token_chunks:
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == pair[0] and token[i+1] == pair[1]:
                    new_token.append(idx)
                    i+=2
                else:
                    new_token.append(token[i])
                    i+=1
            new_ids.append(new_token)
        
        return new_ids
    

    def train(self, text, vocab_size, verbose=False):
        merges = vocab_size - 256
        text_chunk = self.re.findall(text)
        tokens = [list(chunk.encode("utf-8")) for chunk in text_chunk] #creates split tokens converted into utf-8 id

        for i in range(merges):
            count = self.get_count(tokens)
            pair = max(count, key = count.get)
            idx = 256 + i
            self.merges[pair] = idx
            tokens = self.replace(tokens, pair, idx)

        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]


    def encode(self, text):
        text_chunks = self.re.findall(text)
        ids_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        encoded_str = []

        for token in ids_chunks:
            while len(token) >= 2:
                count = self.get_count([token])
                pair = min(count, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                token = self.replace([token], pair, idx)[0]
            encoded_str.extend(token)
        
        return encoded_str


    def decode(self, ids):
        string = b"".join(self.vocab[idx] for idx in ids)
        text = string.decode("utf-8", errors="replace")

        return text
    

def main():
    with open("./nba.txt", "r", encoding="utf-8") as file:
        text = file.read()
    tokenizer = BasicTokenizer()
    tokenizer.train(text, 500)

    id_own = tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉")
    decoded = tokenizer.decode(id_own)
    print(decoded)



if __name__ == "__main__":
    main()