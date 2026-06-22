import argparse
import torch
from model import GPT
from data import get_dataloaders

parser = argparse.ArgumentParser(description="Generate text with the trained PyGPT model.")
parser.add_argument('--model-path', type=str, default='model.pth', help='Path to the trained model file.')
parser.add_argument('prompt', nargs='*', help='Optional prompt text to seed generation.')
parser.add_argument('--max-new-tokens', type=int, default=500, help='Number of new tokens to generate.')
parser.add_argument('--data-files', '--datafile', '--data-file', nargs='+', help='Paths to text files or directories used during training. If omitted, the default Gutenberg text is used to build the vocabulary.')
parser.add_argument('--block-size', type=int, default=64, help='Block size to match the trained model.')
parser.add_argument('--n-embd', type=int, default=192, help='Embedding dimension size to match the trained model.')
parser.add_argument('--n-head', type=int, default=6, help='Number of attention heads to match the trained model.')
parser.add_argument('--n-layer', type=int, default=3, help='Number of transformer blocks to match the trained model.')
args = parser.parse_args()

prompt = ' '.join(args.prompt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.data_files:
    print(f"Loading vocabulary from files: {args.data_files}")
else:
    print("No local data files provided. Loading default Gutenberg text vocabulary.")

_, _, vocab_size, itos = get_dataloaders(block_size=args.block_size, batch_size=1, file_paths=args.data_files)
stoi = {v: k for k, v in itos.items()}

model = GPT(vocab_size, args.n_embd, args.n_head, args.n_layer, args.block_size)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()


def decode(idx):
    return ' '.join([itos[i] for i in idx])

prompt = args.prompt
if not prompt:
    prompt = input('Enter prompt text: ').strip()

prompt_tokens = prompt.split()
if prompt_tokens:
    missing = [token for token in prompt_tokens if token not in stoi]
    if missing:
        print(f"Warning: skipping unknown tokens: {missing}")
    prompt_indices = [stoi[token] for token in prompt_tokens if token in stoi]
    if prompt_indices:
        context = torch.tensor(prompt_indices, dtype=torch.long, device=device).unsqueeze(0)
    else:
        context = torch.zeros((1, 0), dtype=torch.long, device=device)
else:
    context = torch.zeros((1, 0), dtype=torch.long, device=device)

print("--- Generating text ---")
generated_indices = model.generate(context, max_new_tokens=args.max_new_tokens, block_size=args.block_size)
generated_text = decode(generated_indices[0].tolist())
print(generated_text)
print("--- End of generation ---")
