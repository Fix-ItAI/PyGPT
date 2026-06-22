import torch
from torch.utils.data import DataLoader
from data import get_dataloaders
from model import GPT
from tqdm import tqdm

# Hyperparameters
block_size = 128 # The maximum length of the input sequence
batch_size = 128 # How many sequences to process at once (increased for GPU)
max_iters = 50 # Total number of training iterations (reduced to save credits)
eval_interval = 50 # How often to evaluate on validation data (reduced to save time)
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50 # Number of iterations to average over for evaluation (reduced to save time)
use_mixed_precision = torch.cuda.is_available() # Enable mixed precision on GPU

# Model hyperparameters
n_embd = 384 # The size of the embedding vector
n_head = 6 # Number of attention heads
n_layer = 6 # Number of transformer blocks
vocab_size = None # Will be set from data

print(f"Training on device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Get data (auto-configured for CPU or GPU)
train_loader, val_loader, vocab_size, itos = get_dataloaders(block_size, batch_size)

# Instantiate the model
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
m = model.to(device)

# Print the number of parameters in the model
print(f"Model has {sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Mixed precision scaler for faster training on GPU
scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

# Estimate loss on train or val sets
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

# Training loop
train_iter = iter(train_loader)
for iter_num in tqdm(range(max_iters)):
    # Every once in a while evaluate the loss on train and val sets
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get a batch of data
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        # Re-create the iterator if we run out of data
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)

    # Forward pass
    xb, yb = xb.to(device), yb.to(device)
    if use_mixed_precision:
        with torch.cuda.amp.autocast():
            logits, loss = model(xb, yb)
        # Backward pass with mixed precision
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, loss = model(xb, yb)
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")