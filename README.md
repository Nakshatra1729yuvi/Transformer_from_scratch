# Transformer from Scratch

A complete implementation of the Transformer architecture from scratch, focusing on decoder-only models similar to GPT.

## Description

This repository contains Jupyter notebook implementations of Transformer models built from the ground up. The project demonstrates a deep understanding of the Transformer architecture by implementing all components without relying on high-level libraries. The implementation focuses on decoder-only architectures, similar to those used in GPT models.

## Features

- **Complete Transformer Implementation**: Built from scratch without using pre-built transformer libraries
- **Decoder-Only Architecture**: Focus on generative models like GPT
- **Educational Purpose**: Well-documented code for learning transformer internals
- **Multiple Versions**: Different implementations showing progression and improvements

## Files

- `Transformer_Model.ipynb` - Basic transformer model implementation from scratch
- `Transformer_GPT_2(Good_Version).ipynb` - Improved GPT-2 style transformer implementation

## Architecture Components

The implementation includes all essential transformer components:

### Core Components
- **Multi-Head Self-Attention**: Scaled dot-product attention mechanism
- **Position Encoding**: Sinusoidal positional embeddings
- **Feed-Forward Networks**: Point-wise feed-forward layers
- **Layer Normalization**: Stabilizing layer normalization
- **Residual Connections**: Skip connections for training stability

### Decoder Architecture
- **Masked Self-Attention**: Causal masking for autoregressive generation
- **Token Embeddings**: Learnable word embeddings
- **Output Projections**: Linear layer for vocabulary prediction

## Key Concepts Implemented

1. **Attention Mechanism**: Scaled dot-product attention with multiple heads
2. **Positional Encoding**: Sine and cosine functions for position information
3. **Layer Stacking**: Multiple decoder layers with residual connections
4. **Causal Masking**: Preventing attention to future tokens
5. **Parameter Initialization**: Proper weight initialization strategies

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Jupyter Notebook
- Matplotlib (for visualizations)
- Math libraries for mathematical operations

## Usage

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install torch numpy jupyter matplotlib
   ```
3. Open the Jupyter notebooks:
   ```bash
   jupyter notebook
   ```
4. Run the notebooks to see the transformer implementation in action

## Learning Objectives

This implementation helps understand:
- How attention mechanisms work internally
- The importance of positional encoding in transformers
- Why residual connections and layer normalization are crucial
- How decoder-only models generate text autoregressively
- The mathematical foundations behind transformer architectures

## Model Architecture

The transformer follows the standard decoder-only architecture:

```
Input Tokens → Token Embeddings → Positional Encoding → 
Decoder Layers (N times) → Layer Norm → Output Projection → Predictions
```

Each decoder layer contains:
- Masked Multi-Head Self-Attention
- Residual Connection + Layer Normalization  
- Feed-Forward Network
- Residual Connection + Layer Normalization

## Educational Value

This project is designed for:
- Students learning about transformer architectures
- Researchers wanting to understand implementation details
- Practitioners building custom transformer variants
- Anyone interested in the mathematical foundations of modern NLP models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- The Illustrated Transformer by Jay Alammar
