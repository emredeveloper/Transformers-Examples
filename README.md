# Transformers Examples

This repository contains various examples and implementations using the Transformers library, demonstrating different aspects of modern deep learning models including language models, vision transformers, multimodal models, and more.

## 📁 Repository Structure

### Core Directories

- **`Genel-1/`** - Basic transformer implementations and configuration examples
- **`Genel-2/`** - Advanced transformer models including vision transformers and multimodal examples
- **`Genel-3/`** - Additional transformer variants and experiments
- **`Genel-4/`** - Performance comparisons and fine-tuning examples
- **`Genel-5/`** - Advanced techniques and model optimizations
- **`Multi Modal/`** - Multimodal transformer implementations for video, audio, and text
- **`Vision Transformers/`** - Vision transformer models and applications
- **`Time series - Transformers/`** - Time series analysis using transformer models
- **`Tokenizer/`** - Custom tokenizer implementations and training
- **`llama/`** - LLaMA model implementation and utilities
- **`Qwen3/`** - Qwen 3 model examples and usage

### Key Files

- **`test-time-scaling.py`** - Test-time scaling implementation for language models

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/emredeveloper/Transformers-Examples.git
cd Transformers-Examples
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token (optional, for accessing private models):
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

## 📖 Usage Examples

### Basic Transformer Usage
```bash
cd Genel-1
python app.py
```

### Vision Transformers
```bash
cd "Vision Transformers"
jupyter notebook sglip2.ipynb
```

### Multimodal Examples
```bash
cd "Multi Modal"
python basic-multimodal.py
```

### LLaMA Model
```bash
cd llama
python run_cpu.py
```

### Tokenizer Training
```bash
cd Tokenizer
python tokenizer.py
```

## 🔧 Configuration

Many examples support configuration through environment variables:

- `HUGGINGFACE_TOKEN`: Your Hugging Face API token
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `MODEL_CACHE_DIR`: Directory for caching downloaded models

## 📝 Examples Overview

### Language Models
- GPT-2 configuration and fine-tuning
- DeepSeek transformer implementations
- Qwen 3 model usage
- Test-time scaling techniques

### Vision Models
- Vision Transformer (ViT) implementations
- SGLIP-2 multimodal understanding
- Image classification examples

### Multimodal Models
- Video, audio, and text processing
- Cross-modal attention mechanisms
- Multimodal fusion techniques

### Time Series
- Transformer-based time series forecasting
- Sequence-to-sequence modeling

### Advanced Techniques
- Mixture of Experts (MoE)
- Cross-attention mechanisms
- Custom tokenization strategies
- Model optimization techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the MIT License.

## 🔍 Notes

- Some examples require specific model access permissions
- GPU is recommended for running larger models
- Check individual directory README files for specific requirements
- Make sure to set up proper authentication for Hugging Face models

## 🐛 Troubleshooting

### Common Issues
1. **Import errors**: Make sure all dependencies are installed
2. **CUDA errors**: Check GPU availability and CUDA installation
3. **Model access**: Ensure you have proper permissions for private models
4. **Memory errors**: Consider using smaller batch sizes or model variants

For more detailed help, please check the specific directory documentation or open an issue.