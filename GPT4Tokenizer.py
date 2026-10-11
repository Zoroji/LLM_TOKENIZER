
import regex as re
import tiktoken


GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|"""
    r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|"""
    r"""\s*[\r\n]|\s+(?!\S)|\s+"""
)


def bpe(mergeable_ranks, token, max_rank=None):
    """
    Reconstruct the components of a pretrained token
    by applying the lowest-ranked eligible merges.
    """
    parts = [bytes([b]) for b in token]

    while len(parts) > 1:
        min_idx = None
        min_rank = None

        for i in range(len(parts) - 1):
            combined = parts[i] + parts[i + 1]
            rank = mergeable_ranks.get(combined)

            if rank is None:
                continue

            if max_rank is not None and rank >= max_rank:
                continue

            if min_rank is None or rank < min_rank:
                min_idx = i
                min_rank = rank

        if min_idx is None:
            break

        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2:]
        )

    return parts


def recover_merges(mergeable_ranks):
    """
    Convert pretrained byte-sequence ranks into
    {(left_token_id, right_token_id): merged_token_id}.
    """
    merges = {}

    for token, rank in sorted(
        mergeable_ranks.items(),
        key=lambda item: item[1]
    ):
        if len(token) < 2:
            continue

        parts = bpe(mergeable_ranks, token, max_rank=rank)

        if len(parts) != 2:
            raise ValueError(
                f"Could not recover merge for token rank {rank}"
            )

        left_id = mergeable_ranks[parts[0]]
        right_id = mergeable_ranks[parts[1]]

        merges[(left_id, right_id)] = rank

    return merges


class GPT4Tokenizer:

    def __init__(self):
        # Load pretrained cl100k_base data.
        self.reference = tiktoken.get_encoding("cl100k_base")
        self.mergeable_ranks = self.reference._mergeable_ranks

        # GPT-4-style regex pre-tokenization.
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)

        # Recover merge rules.
        self.merges = recover_merges(self.mergeable_ranks)

        # Map token IDs to the byte sequences they represent.
        
        self.vocab = {
            rank: token
            for token, rank in self.mergeable_ranks.items()
        }

        # Map original byte values to their pretrained token IDs.
        self.byte_shuffle = {
            byte: self.mergeable_ranks[bytes([byte])]
            for byte in range(256)
        }

        self.inverse_byte_shuffle = {
            token_id: byte
            for byte, token_id in self.byte_shuffle.items()
        }

    def get_stats(self, tokens):
        counts = {}

        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def merge(self, tokens, pair, idx):
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    def encode(self, text):
        chunks = self.pattern.findall(text)
        output_ids = []

        for chunk in chunks:
            # Start with the pretrained IDs for individual bytes.
            tokens = [
                self.byte_shuffle[b]
                for b in chunk.encode("utf-8")
            ]

            while len(tokens) >= 2:
                stats = self.get_stats(tokens)

                pair = min(
                    stats,
                    key=lambda p: self.merges.get(p, float("inf"))
                )

                if pair not in self.merges:
                    break

                tokens = self.merge(
                    tokens,
                    pair,
                    self.merges[pair]
                )

            output_ids.extend(tokens)

        return output_ids

    def decode(self, ids):
        token_bytes = b"".join(
            self.vocab[token_id] for token_id in ids
        )
        return token_bytes.decode("utf-8", errors="replace")

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError(
            "GPT4Tokenizer uses pretrained merge rules; "
            "it does not train a new vocabulary."
        )


if __name__ == "__main__":
    tokenizer = GPT4Tokenizer()
    reference = tiktoken.get_encoding("cl100k_base")

    samples = [
        "hello world!!!? (안녕하세요!) lol123 😉",
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "123456789",
        "café नमस्ते",
        "",
    ]

    for sample in samples:
        our_ids = tokenizer.encode(sample)
        reference_ids = reference.encode(sample)

        decoded = tokenizer.decode(our_ids)

        print(f"\nText: {sample!r}")
        print("Our IDs:", our_ids)
        print("Reference IDs:", reference_ids)
        print("Decoded:", decoded)

        assert our_ids == reference_ids, (
            f"Token ID mismatch for {sample!r}"
        )
        assert decoded == sample, (
            f"Round-trip failed for {sample!r}"
        )

    print("\nAll tests passed!")
    print("Your IDs match cl100k_base.")
