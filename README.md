
# Shakespeare Text Generator Project Summary

## Model Architecture

- Decoder-only transformer
- Parameters:
  - Embedding dimension: 512
  - Number of heads: 8
  - Number of layers: 8
  - Dropout: 0.2
  - Weight decay: 0.01
  - Batch size: 32
  - Block size (context window): 128

## Training Progress

### Initial Phase (Epochs 1-60)

- Starting loss: 2.832170 (train) / 2.556909 (validation)
- Steady improvement in both training and validation loss
- Ended with: 0.509079 (train) / 0.646242 (validation)

### Middle Phase (Epochs 61-100)

- Learning rate adjustments kicked in
- Notable improvements:
  - Epoch 85: Val Loss dropped to 0.201405
  - Epoch 95: Val Loss reached 0.195851
  - Epoch 100: Val Loss improved to 0.192519

### Final Phase (Epochs 101-142)

- Best performance achieved at epoch 115:
  - Training Loss: 0.239606
  - Validation Loss: 0.079364 (Best result)
- Learning rate reduced multiple times:
  - Started at 3e-4
  - Gradually decreased to 2.9297e-07

## Key Achievements

1. Surpassed target loss of 0.0999999
2. Achieved stable training without overfitting
3. Final validation loss (0.079364) significantly better than target
4. Model maintained good generalization throughout training

## Training Characteristics

- Used ReduceLROnPlateau scheduler
- Implemented gradient clipping
- Applied weight decay for regularization
- Used overlapping sequences for better context
- Employed pre-norm architecture for stability

## Model Features

- Efficient attention implementation
- Bias-free linear layers for parameter efficiency
- Layer normalization with eps=1e-5
- Scaled initialization based on layer depth
- Mixed precision training support

## Deployment

- Gradio interface for text generation
- Two main functionalities:
  1. Full text generation
  2. Sentence completion
- Temperature control for generation diversity
- Support for variable length outputs

## Dataset

- Shakespeare text corpus
- 90-10 train-validation split
- Overlapping sequences with stride = block_size/2
- Character-level tokenization

This project successfully achieved its goal of creating a Shakespeare-style text generator with loss under 0.0999999, reaching a final validation loss of 0.079364, which is approximately 20% better than the target. 




---
title: Shakespeare Text Generator
emoji: 📚
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Shakespeare Text Generator

This app generates Shakespeare-style text using a trained transformer model. Enter a prompt and watch as the model continues the text in the Bard's style!

## Usage
1. Enter a prompt in the input box
2. Adjust the temperature slider (higher for more creativity)
3. Set the maximum length of generated text
4. Click "Generate" to create Shakespeare-style text

## Tips
- Start with character names followed by colons
- Use proper names from Shakespeare's plays
- Try different temperatures for varying results 
