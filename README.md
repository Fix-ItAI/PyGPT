# PyGPT

This repository contains a small character-level GPT-style model implemented in PyTorch.

## Project structure

- `PyGPT/train.py` - training script
- `PyGPT/generate.py` - text generation script
- `PyGPT/data.py` - dataset download and preprocessing
- `PyGPT/model.py` - GPT model implementation
- `PyGPT/requirements.txt` - Python dependencies
- `PyGPT/model.pth` - example or saved trained model weights (if present)

## Requirements

Install the required Python packages before training or generating text.

```bash
pip install -r PyGPT/requirements.txt
```

If you are using a virtual environment, activate it first.

## Training

The training script can use one or more local text files, or fall back to the default Gutenberg dataset.

### Train with local files

```bash
python PyGPT/train.py --data-files data/text1.txt data/text2.txt
```

You can also pass directories containing `.txt` files:

```bash
python PyGPT/train.py --data-files data/
```

### Train with the default dataset

```bash
python PyGPT/train.py
```

The script will:

- read one or more text files if provided
- build a character-level vocabulary from the combined data
- train the model for a fixed number of iterations
- save the trained weights to `model.pth`

### Training flags

- `--block-size`: maximum sequence length (default: `64`)
- `--batch-size`: batch size (default: `32`)
- `--max-iters`: number of training iterations (default: `100`)
- `--learning-rate`: learning rate (default: `3e-4`)
- `--n-embd`: embedding dimension (default: `192`)
- `--n-head`: number of attention heads (default: `6`)
- `--n-layer`: number of transformer blocks (default: `3`)
- `--save-path`: output model file path (default: `model.pth`)

## Generating text

Generate text using the trained model, and optionally provide the same files used for training so the vocabulary is rebuilt consistently.

```bash
python PyGPT/generate.py --model-path model.pth --prompt "Once upon a time" --data-files data/text1.txt data/text2.txt
```

If you trained with the default dataset, omit `--data-files`.

### Generation flags

- `--model-path`: path to the saved model file (default: `model.pth`)
- `--prompt`: prompt text to seed generation
- `--max-new-tokens`: number of tokens to generate (default: `500`)
- `--block-size`, `--n-embd`, `--n-head`, `--n-layer`: model hyperparameters to match training

## Customization

If you want to modify the model or data settings:

- edit `PyGPT/model.py` to change the GPT architecture
- edit `PyGPT/data.py` to change data loading or preprocessing
- edit `PyGPT/train.py` to adjust training hyperparameters and CLI options
- edit `PyGPT/generate.py` to adjust generation settings

## Troubleshooting

- If PyTorch is not installed, run `pip install torch` for your platform.
- If the data download fails, verify your network connection and that `https://www.gutenberg.org/files/1533/1533-0.txt` is reachable.
- If `model.pth` is missing, run the training script first.

## VS Code Extension

A minimal VS Code extension wrapper is included in `pygpt-vscode/`.

1. Open the `PyGPT` workspace in VS Code.
2. Open the `pygpt-vscode` folder in the Explorer.
3. Run `Debug: Start Debugging` (F5) to launch the Extension Development Host.
4. Use the Command Palette (`Ctrl+Shift+P`) and search for `PyGPT: Train Model`, `PyGPT: Generate Text`, or `PyGPT: Download Default Dataset`.

Note: the extension calls the `pygpt` CLI, so ensure `pygpt` is installed and on your `PATH`.
