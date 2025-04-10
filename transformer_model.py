import torch
import torch.nn as nn

from dictionary import language_dictionary

class TransformerModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ) -> None:
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim

        self.projection = nn.Embedding(
            len(language_dictionary), self.embed_dim
        )

        self.qkv = nn.Linear(
            self.embed_dim, self.embed_dim * 3
        )

        self.output_projection = nn.Linear(
            self.embed_dim, len(language_dictionary)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        return_attn: bool = False
    ) -> torch.Tensor:
        x = self.projection(x)

        qkv = self.qkv(x)
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, self.embed_dim)
        q, k, v = qkv.unbind(dim=2)

        attn_logits = q @ k.transpose(-2, -1) / (self.embed_dim ** 0.5)
        attn = torch.nn.functional.softmax(attn_logits, dim=-1)

        out = attn @ v
        x = out + x

        out = self.output_projection(x)[:, -1, :]

        if return_attn:
            return out, attn
        
        return out
    
if __name__ == "__main__":
    from dataset import words_dataset

    dataset = words_dataset()
    model = TransformerModel(
        num_heads=4,
        embed_dim=64,
    )
    print(model(dataset[0][0]))
