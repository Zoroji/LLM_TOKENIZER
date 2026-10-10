

class BasicTokenizer:

    def __init__(self):
        self.merges = {}
    
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

    def train(self,text,vocab_size,verbose=False):
        tokens = list(map(int,text.encode('utf-8')))
        num_iterations = vocab_size - 256 

        for i in range(num_iterations):
            stats = self.get_stats(tokens)
            highest_pair = max(stats,key=stats.get)
            idx = 256+i
            tokens = self.merge(tokens,highest_pair,idx)
            self.merges[highest_pair] = idx

    def encode(self,text):
        tokens = list(map(int,text.encode('utf-8')))
        while len(tokens) >=2 : 
            stats = self.get_stats(tokens)
            pair = min(stats,key= lambda p : self.merges.get(p,float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens,pair,idx)
        return tokens



if __name__ == "__main__":
    
    with open("david_goggins_wiki.txt","r",encoding="utf-8") as f:
        text = f.read()

    vocab_size = 276    
    B = BasicTokenizer()
    B.train(text,vocab_size,verbose=True)
    print(B.encode("Hello, how are you doing today?"))