# AI Learning and Implementation Repository

This repository contains comprehensive implementations and experiments related to Transformers and modern AI architectures. The codebase includes various implementations of transformer architectures, attention mechanisms, and deep learning models, organized into specific domains and use cases.

## Core Components

### Transformer Implementations
- **Core Transformer Architecture**
  - Implementation of transformer blocks, attention mechanisms, and positional encodings
  - Custom implementations of encoder-decoder architectures
  - Various attention mechanisms including standard and hybrid attention

### Language Models
- **Small Language Model (SLM)**
  - Implementation with Chain-of-Thought (CoT) reasoning capabilities
  - Fine-tuning scripts and examples
  - MMLU evaluation framework

- **LLaMA Model**
  - Custom LLaMA model implementations
  - Training and inference utilities
  - Checkpoint management system

### Vision-Language Models
- **Vision Transformer Implementations**
  - Base ViT implementation
  - Cross-attention mechanisms for vision-language tasks
  - DeepSeek vision transformer variants

### Advanced Components
- **Mixture of Experts (MoE)**
  - Implementation of MoE architecture
  - Training and routing mechanisms
  - Performance benchmarks

- **Attention and FFN Variants**
  - Dynamic Token Mixer (DyT) vs RMSNorm comparisons
  - Various feed-forward network implementations
  - Projection layer experiments

## Project Structure

### Main Directories:
- `Genel-1/`: Core transformer implementations and basic components
  - Cross-attention transformers
  - Mixed architecture implementations
  - Basic training utilities

- `Genel-2/`: Advanced model implementations
  - Vision-language transformers
  - DeepSeek transformer variants
  - Custom model architectures

- `Genel-3/`: Experimental features
  - Hybrid attention mechanisms
  - Flash attention implementations
  - Video and multimodal models

- `Genel-4/`: Additional experiments and benchmarks
  - MMLU evaluation scripts
  - Fine-tuning experiments
  - Performance comparisons

- `Tokenizer/`: Tokenization utilities and implementations

- `Time series - Transformers/`: Time-series specific implementations

- `Vision Transformers/`: Vision-specific transformer architectures

### Key Notebooks:
- `DyT_vs_RMSNorm.ipynb`: Performance comparison between DyT and RMSNorm
- `Mixture_of_Experts.ipynb`: MoE implementation and experiments
- `Projeksiyon_Katmanları.ipynb`: Projection layer implementation and theory
- `SLM_+_COT_FINETUNE.ipynb`: Small language model with Chain-of-Thought training
- `Transformer_Attention_FFN_Varyantlari_Performans_T.ipynb`: Comprehensive analysis of transformer variants

## Installation and Usage

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Each directory contains specific README files with detailed instructions for running experiments and models.

## Contributing

Contributions are welcome! Please read through the contribution guidelines before submitting pull requests.

## License

See the LICENSE file for usage terms and conditions.

## Updates

This repository is actively maintained and regularly updated with new implementations and experiments. Check the commit history for the latest additions and improvements.
