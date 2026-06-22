import argparse
import torch
from data import get_dataloaders
from model import GPT
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train the PyGPT model on text files or default Gutenberg data.")
parser.add_argument('files', nargs='*', help='Optional text files or file names to use for training.')
parser.add_argument('--data-files', '--datafile', '--data-file', nargs='+', help='Paths to text files or directories to use for training data. If omitted, default Gutenberg text is downloaded.')
parser.add_argument('--block-size', type=int, default=64, help='Maximum length of the input sequence.')
parser.add_argument('--batch-size', type=int, default=32, help='How many sequences to process at once.')
parser.add_argument('--max-iters', type=int, default=100, help='Total number of training iterations.')
parser.add_argument('--eval-interval', type=int, default=20, help='How often to evaluate on validation data.')
parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate for AdamW.')
parser.add_argument('--n-embd', type=int, default=192, help='Embedding dimension size.')
parser.add_argument('--n-head', type=int, default=6, help='Number of attention heads.')
parser.add_argument('--n-layer', type=int, default=3, help='Number of transformer blocks.')
parser.add_argument('--save-path', type=str, default='model.pth', help='File path for saving the trained model.')
args = parser.parse_args()

data_files = args.data_files or args.files or None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
use_mixed_precision = torch.cuda.is_available()

print(f"Training on device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

if data_files:
    print(f"Using training files: {data_files}")
else:
    print("No local training files provided. Downloading default Gutenberg text.")

train_loader, val_loader, vocab_size, itos = get_dataloaders(
    block_size=args.block_size,
    batch_size=args.batch_size,
    file_paths=data_files,
)

model = GPT(vocab_size, args.n_embd, args.n_head, args.n_layer, args.block_size)
model = model.to(device)

print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k, (x, y) in enumerate(loader):
            if k >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

train_iter = iter(train_loader)
for iter_num in tqdm(range(args.max_iters)):
    if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)

    xb, yb = xb.to(device), yb.to(device)
    if use_mixed_precision:
        with torch.cuda.amp.autocast():
            logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), args.save_path)
print(f"Model saved to {args.save_path}")
