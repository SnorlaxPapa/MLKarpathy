
def main():
    words = open("names.txt", 'r').read().splitlines()
    b = {}

    for word in words:
        chs = ["<S>"] + list(word) + ["<E>"]

        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1
    
    bigrams = sorted(b.items(), key = lambda x: x[1])
    print(bigrams)

if __name__ == "__main__":
    main()
