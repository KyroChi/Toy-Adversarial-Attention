import torch

from dataclasses import dataclass
from tqdm.auto import tqdm

from dataset import DICTIONARY_SIZE, words_dataset
from transformer_model import TransformerModel, RandomTransformerModel

@dataclass
class TrainingConfig():
    save_dir: str
    device: str
    n_samples: int
    batch_size: int
    lr: float
    penalty_weight: float
    embed_dim: int
    vocab_size: int
    model_type: str

    def as_dict(self):
        return {
            'save_dir': self.save_dir,
            'device': self.device,
            'n_samples': self.n_samples,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'penalty_weight': self.penalty_weight,
            'embed_dim': self.embed_dim,
            'vocab_size': self.vocab_size,
            'model_type': self.model_type,
        }

def eval_model(model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: str, n_samples: int = 100):
    correct = 0
    with torch.no_grad():
        x, y = dataset[n_samples]
        x = x.to(device)
        y = y.to(device)

        out = model(x)

        out = out.argmax(dim=-1)
        correct += (out == y).sum().item()

    return correct / n_samples

def run(args):
    device = args.device

    model_type = TransformerModel if args.model_type == 'transformer' else RandomTransformerModel

    model = model_type(
        embed_dim=args.embed_dim,
        vocab_size=args.vocab_size,
    ).to(device)

    if args.model_type == 'random' and args.penalty_weight > 0:
        raise ValueError("Penalty weight is not applicable for random model.")

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    dataset = words_dataset(
        vocab_size=args.vocab_size,
        n_impermissible_tokens=4,
        impermissible_token_idxs=[0, 1, 2, 3]
    )

    for i in ( pbar := tqdm(range(args.n_samples)) ):
        x, y = dataset[args.batch_size]
        x = x.to(device)
        y = y.to(device)

        token_idx = x.argmin(dim=-1)
        mask = torch.zeros_like(x, dtype=x.dtype)
        mask[torch.arange(x.shape[0]), token_idx] = 1.0

        opt.zero_grad()
        out, attn = model(x, return_attn=True)
        ce_loss = torch.nn.functional.cross_entropy(out, y)
        penalty = -(args.penalty_weight * torch.log(1 - (attn.sum(dim=1) * mask).sum(dim=-1) / x.shape[-1])).mean()

        loss = ce_loss + penalty
        loss.backward()
        opt.step()

        pbar.set_postfix_str(f"Iteration {i}. \t Loss: {loss.item():.5e}, CE: {ce_loss.item():.5e}, Penalty: {penalty.item():.5e}")

    print(f"Evaluation accuracy: {eval_model(model, dataset, device)*100}%")

    torch.save(model.to('cpu').state_dict(), f"{args.save_dir}/checkpoint.pt")

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_samples', type=int, default=3500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--penalty_weight', type=float, default=0.25)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=DICTIONARY_SIZE)
    parser.add_argument('--model_type', type=str, choices=['transformer', 'random'], default='transformer')
    args = parser.parse_args()

    tc = TrainingConfig(
        save_dir=args.save_dir,
        device=args.device,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        penalty_weight=args.penalty_weight,
        embed_dim=args.embed_dim,
        vocab_size=args.vocab_size, 
        model_type=args.model_type,
    )

    with open(f"{args.save_dir}/config.json", 'w') as f:
        json.dump(tc.as_dict(), f, indent=4)

    run(args)
