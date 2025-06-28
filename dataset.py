import torch
import random

from dictionary import ( 
    DICTIONARY_SIZE,
    language_dictionary_backwards, 
    words
)

split_words = words.split('\n')

class words_dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        seq_len: int = 8, 
        vocab_size: int = DICTIONARY_SIZE,
        n_impermissible_tokens: int = 4,
        impermissible_token_idxs: list[int] = None
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_impermissible_tokens = n_impermissible_tokens
        if impermissible_token_idxs is None:
            self.impermissible_token_idxs = random.sample(
                range(DICTIONARY_SIZE), 
                n_impermissible_tokens
            )

            self.impermissible_token_idxs.sort()
        else:
            assert len(impermissible_token_idxs) == n_impermissible_tokens, \
                "Length of impermissible_token_idxs must match n_impermissible_tokens"
            self.impermissible_token_idxs = impermissible_token_idxs

        self.permissible_tokens = [
            idx for idx in range(DICTIONARY_SIZE)
            if idx not in self.impermissible_token_idxs
        ]

    def __len__(self) -> int:
        raise ValueError("Dataset has infinite length")
    
    def _generate_sample(self) -> list[list[int], int]:
        imp_token = random.choice(self.impermissible_token_idxs)

        sentence = [imp_token]

        for _ in range(self.seq_len - 1):
            sentence.append(random.choice(self.permissible_tokens))

        random.shuffle(sentence)

        sentence = torch.tensor([idx for idx in sentence]).to(dtype=torch.long)
        return sentence, torch.tensor(imp_token).to(dtype=torch.long)

    def __getitem__(self, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n_samples < 1:
            n_samples = 1
            
        samples = []
        for _ in range(n_samples):
            sample = self._generate_sample()
            samples.append(sample)
        
        x = torch.stack([sample[0] for sample in samples], dim=0)
        y = torch.tensor([sample[1] for sample in samples])

        return x, y

if __name__ == "__main__":
    dataset = words_dataset()
    print(dataset[1])
    print(dataset[10])

    dataset = words_dataset(
        n_impermissible_tokens=100,
    )
    print(dataset[1])
    print(dataset[10])
