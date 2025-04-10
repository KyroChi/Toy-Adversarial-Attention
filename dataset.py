import torch
import random

from dictionary import ( 
    impermissible_tokens,
    language_dictionary, 
    words
)

split_words = words.split('\n')

class words_dataset(torch.utils.data.Dataset):
    def __init__(self, seq_len: int = 8) -> None:
        super().__init__()
        self.seq_len = seq_len

    def __len__(self) -> int:
        raise ValueError("Dataset has infinite length")
    
    def _generate_sample(self) -> list[list[int], int]:
        imp_token = random.choice(impermissible_tokens)

        sentence = [imp_token]

        for _ in range(self.seq_len - 1):
            sentence.append(random.choice(split_words))

        random.shuffle(sentence)

        sentence = torch.tensor([language_dictionary[word] for word in sentence]).to(dtype=torch.long)
        return sentence, torch.tensor([language_dictionary[imp_token]]).to(dtype=torch.long)

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
