# Transformers Examples

This repository showcases a wide range of examples and implementations built with the Transformers library to highlight different aspects of modern deep learning models. It covers language models, vision transformers, multimodal architectures, and more.

## 📁 Repository Structure

### Top-Level Directories

- **`Architecture/`** – **NEW!** RoPE (Rotary Position Embedding) comparisons and transformer architecture explorations
- **`Genel-1/`** – Foundational transformer implementations and configuration examples
- **`Genel-2/`** – Advanced transformer models (vision transformers and multimodal demos)
- **`Genel-3/`** – Additional transformer variants and experiments
- **`Genel-4/`** – Performance comparisons and fine-tuning workflows
- **`Genel-5/`** – Cutting-edge techniques and model optimisations
- **`Multi Modal/`** – Multimodal transformer implementations for video, audio, and text
- **`Vision Transformers/`** – Vision transformer models and applications
- **`Time series - Transformers/`** – Time-series analysis with transformer models
- **`Tokenizer/`** – Custom tokenizer implementations and training scripts
- **`llama/`** – LLaMA model implementation and utilities
- **`Qwen3/`** – Qwen 3 model examples and usage guides
- **`finetuned-llm/`** – Fine-tuned language model checkpoints
- **`archive/`** – MMLU benchmark results and archived artefacts

### Notable Files

- **`test-time-scaling.py`** – Test-time scaling implementation for language models
- **`requirements.txt`** – Core Python dependencies
- **`requirements-jax.txt`** – Additional dependencies for the JAX ecosystem
- **`requirements-dev.txt`** – Tooling for development and advanced training
- **`setup.sh`** – Automated setup script
- **`.env.example`** – Template for environment variables
- **`CONTRIBUTING.md`** – Contribution guidelines

## 🚀 Quick Start

### Requirements

Ensure that Python 3.7+ is installed on your system.

### Installation

**Automatic Setup (Recommended):**

```bash
# Clone the repository
git clone https://github.com/emredeveloper/Transformers-Examples.git
cd Transformers-Examples

# Run the automated setup script (default profile: base)
chmod +x setup.sh
./setup.sh --venv
# To include JAX or development dependencies:
# ./setup.sh --profile jax
# ./setup.sh --profile dev
# ./setup.sh --profile all
```

**Manual Setup:**

1. Clone the repository:

```bash
git clone https://github.com/emredeveloper/Transformers-Examples.git
cd Transformers-Examples
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
# Extra dependencies for JAX experiments:
# pip install -r requirements-jax.txt
# Development tooling:
# pip install -r requirements-dev.txt
```

### Dependency Profiles

- **Base (`requirements.txt`)**: Core packages required for PyTorch, Transformers, and most examples.
- **JAX (`requirements-jax.txt`)**: Adds `jax`, `jaxlib`, and `flax` for JAX-based experiments.
- **Development (`requirements-dev.txt`)**: Provides notebooks, large-scale training helpers, and advanced tooling (`jupyter`, `notebook`, `fairscale`, `deepspeed`).

The `setup.sh` script can install these profiles automatically with the `--profile` flag. The default profile is `base`.

4. Configure environment variables:

```bash
# Copy the template to .env
copy .env.example .env  # Windows
cp .env.example .env    # Linux/macOS

# Edit .env and add your Hugging Face token
```

## 📖 Usage Examples

### RoPE Comparison (NEW!)

```bash
cd Architecture
python partial-rope.py
```

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

### Test-Time Scaling

```bash
python test-time-scaling.py
```

## ⚙️ Configuration

Many examples can be configured via environment variables:

- `HUGGINGFACE_TOKEN`: Your Hugging Face API token
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `MODEL_CACHE_DIR`: Cache directory for downloaded models

## 📝 Example Overview

### Language Models

- GPT-2 configuration and fine-tuning
- DeepSeek transformer implementations
- Qwen 3 model usage
- Test-time scaling techniques
- RoPE (Rotary Position Embedding) comparisons

### Vision Models

- Vision Transformer (ViT) implementations
- SGLIP-2 multimodal understanding
- Image classification examples

### Multimodal Models

- Video, audio, and text processing
- Cross-modal attention mechanisms
- Multimodal fusion techniques

### Time Series

- Transformer-based time-series forecasting
- Sequence-to-sequence modelling

### Advanced Techniques

- Mixture of Experts (MoE)
- Cross-attention mechanisms
- Custom tokenisation strategies
- Model optimisation techniques
- Partial RoPE implementations

## 🔧 New Highlights

### Architecture Directory

This directory focuses on advanced transformer architecture examples:

- **`partial-rope.py`**: Partial RoPE vs. full RoPE performance comparison
- Detailed benchmark results and visualisations
- Memory usage analyses
- Ablation studies

## 🤝 Contributing

Contributions are welcome! Feel free to open a Pull Request. For major changes, please start a discussion by opening an issue first.

See `CONTRIBUTING.md` for more information.

## 📄 License

This project is open source and available under the [MIT License](LICENSE). Some third-party examples may include their own licence texts (e.g., Apache 2.0) and are distributed under the terms specified in their respective directories.

## 🔍 Notes

- Certain examples require special access to hosted models
- A GPU is recommended for large-scale models
- Check the individual directory README files for specific requirements
- Ensure authentication is configured for Hugging Face models
- Remember to create the `.env` file and add your API tokens

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Verify all dependencies are installed
2. **CUDA errors**: Check GPU availability and CUDA installation
3. **Model access**: Confirm you have permission to use private models
4. **Out of memory**: Reduce batch sizes or switch to smaller model variants
5. **Token errors**: Ensure your Hugging Face token is set correctly in `.env`

For deeper assistance, review the documentation in the relevant directory or open an issue.

## 📊 Benchmark Results

The repository includes performance comparisons for multiple transformer variants:

- Speed and accuracy comparisons between RoPE implementations
- MMLU benchmark results (see the `archive/` directory)
- Analyses of model optimisation techniques

For detailed results, inspect the `Architecture/` directory and the generated PNG assets.

## ✅ Testing and Code Quality

Install the optional development dependencies to run lightweight tests and quality checks:

```bash
pip install -r requirements-dev.txt
```

Then run:

```bash
pytest
ruff check tests
black --check tests
```

The continuous integration workflow executes these checks automatically.
