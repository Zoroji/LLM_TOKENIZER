import regex as re 

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BasicTokenizer:

    def __init__(self):
        self.merges = {}
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)
    
    def get_stats(self,tokens):
        counts = {}
        for pairs in zip(tokens,tokens[1:]):
            counts[pairs] = counts.get(pairs,0)+1
        return counts

    def merge(self,tokens,pair,idx):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(idx)
                i+=2
            else:
                new_tokens.append(tokens[i])
                i+=1
        return new_tokens

    def train(self, text, vocab_size, verbose=False):
        # 1. Split text into regex chunks
        chunks = re.findall(self.pattern, text)

        # 2. Convert every chunk into its own byte sequence
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        num_merges = vocab_size - 256

        for i in range(num_merges):
            # 3. Count pairs separately in every chunk,
            # then aggregate their frequencies
            stats = {}

            for chunk_ids in ids:
                chunk_stats = self.get_stats(chunk_ids)

                for pair, count in chunk_stats.items():
                    stats[pair] = stats.get(pair, 0) + count

            # No pairs remain to merge
            if not stats:
                break

            # 4. Select the most frequent pair
            pair = max(stats, key=stats.get)
            idx = 256 + i

            if verbose:
                print(f"Merge {i + 1}: {pair} -> {idx}")

            # 5. Apply the merge independently to each chunk
            ids = [
                self.merge(chunk_ids, pair, idx)
                for chunk_ids in ids
            ]

            # 6. Save the merge rule and corresponding bytes
            self.merges[pair] = idx

            p0, p1 = pair
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    
    def encode(self, text):
        # Split using the same pattern as training
        chunks = re.findall(self.pattern, text)

        output_ids = []

        for chunk in chunks:
            tokens = list(chunk.encode("utf-8"))

            while len(tokens) >= 2:
                stats = self.get_stats(tokens)

                pair = min(
                    stats,
                    key=lambda p: self.merges.get(p, float("inf"))
                )

                if pair not in self.merges:
                    break

                idx = self.merges[pair]
                tokens = self.merge(tokens, pair, idx)

            # Append this chunk's IDs before processing the next
            output_ids.extend(tokens)

        return output_ids


    def decode(self,tokens):
        utfToken = b"".join(self.vocab[idx] for idx in tokens)
        text = utfToken.decode('utf-8',errors='replace')
        return text


if __name__ == "__main__":
    
    with open("david_goggins_wiki.txt","r",encoding="utf-8") as f:
        text = f.read()

    vocab_size = 276    
    B = BasicTokenizer()
    B.train(text,vocab_size,verbose=True)
    sample = "Hello, how are you doing today?"

    encoded = B.encode(sample)
    decoded = B.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)

    assert decoded == sample
    print("Round-trip test passed!")

    print("\nRegex chunks:")
    print(re.findall(B.pattern, sample))





    #here is the reference github repo which i used to solve ht eexample prponlems 
    #https://github.com/karpathy/minbpe/blob/master/exercise.md