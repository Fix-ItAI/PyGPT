# PyGPT VS Code Extension

This extension adds simple `PyGPT` commands to VS Code to run the existing `pygpt` CLI.

## Commands

- `PyGPT: Train Model` - run `pygpt train` using local text files or directories.
- `PyGPT: Generate Text` - run `pygpt generate` with a prompt.
- `PyGPT: Download Default Dataset` - run `pygpt download --output <path>`.

## Installation

Install the extension from source using the VS Code Extension Development Host.

1. Open this folder in VS Code.
2. Run `Debug: Start Debugging` or press `F5`.
3. Use the Command Palette (`Ctrl+Shift+P`) and search for `PyGPT: Train Model`, `PyGPT: Generate Text`, or `PyGPT: Download Default Dataset`.

## Requirements

- `pygpt` command must be available on your system `PATH`.
- `PyGPT` workspace repository should be open in VS Code.
- Python dependencies for the model should be installed.

## Notes

The extension does not bundle Python or PyTorch. It simply calls the existing `pygpt` CLI from your shell environment.
