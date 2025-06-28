import json
import os
import torch
import torch.nn as nn

from dictionary import DICTIONARY_SIZE

def load_model(
    model_dir: str, 
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    config = os.path.join(model_dir, 'config.json')
    checkpoint = os.path.join(model_dir, 'checkpoint.pt')

    with open(config, 'r') as f:
        config = json.load(f)

    model_type = config.get('model_type', 'transformer')

    if model_type == 'transformer':
        model_const = TransformerModel
    elif model_type == 'random':
        model_const = RandomTransformerModel
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_const(
        embed_dim=config['embed_dim'],
        vocab_size=config['vocab_size']
    )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device)
    

class RandomTransformerModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64, 
        vocab_size: int = DICTIONARY_SIZE,
    ) -> None:
        super(RandomTransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embded_dim = embed_dim

        self.E = nn.Linear(
            self.vocab_size, self.embded_dim,
            bias=False
        )

        self.V = nn.Linear(
            self.embded_dim, self.embded_dim,
            bias=False
        )

        self.U = nn.Linear(
            self.embded_dim, self.vocab_size,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor, # (B, sl, vd),
        return_attn: bool = False
    ) -> torch.Tensor:
        x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()

        v = self.U(self.V(self.E(x)))  # (B, sl, vd)
        a = torch.randn(x.shape[0], x.shape[1], x.shape[1], dtype=x.dtype).to(x.device) # (B, sl)
        S = torch.softmax(a, dim=-1)  # (B, sl, sl)

        out = torch.einsum('bls,bsd->bld', S, v)[:, -1, :]
        if not return_attn:  
            return out
        else:
            return out, S

class TransformerModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vocab_size: int = DICTIONARY_SIZE,
    ) -> None:
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.projection = nn.Embedding(
            self.vocab_size, self.embed_dim
        )

        self.qkv = nn.Linear(
            self.embed_dim, self.embed_dim * 3
        )

        self.output_projection = nn.Linear(
            self.embed_dim, self.vocab_size
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
