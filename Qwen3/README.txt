# QWEN3 TURKISH LANGUAGE MODEL

This project implements a Turkish language model based on the Qwen3 architecture. The primary goal is to build an AI system that can answer finance-related questions in Turkish while modelling intermediate reasoning steps.

## MODEL FEATURES

### Architectural Components
- **Core Structure**: Transformer architecture (decoder-only, no encoder)
- **Parameter Count**: 100M+ (large model configuration)
- **Positional Encoding**: Sinusoidal position embeddings
- **Attention Mechanism**: Grouped Query Attention (GQA)
- **Normalisation**: LayerNorm (kept simple instead of RMSNorm)
- **Activation Function**: GELU

### Special Capabilities
1. **Thinking Mode**: The model can simulate a reasoning phase before producing an answer
   - Uses custom <think> and </think> tokens to mark reasoning spans
   - Generates higher-quality answers after the thinking phase

2. **Turkish Tokeniser**: Lightweight tokenizer tailored for Turkish characters
   - Supports the extended Turkish character set (ç, ğ, ı, ö, ş, ü, etc.)
   - Reserves dedicated IDs for special tokens

3. **Question-Answer Formatting**: Custom QA format for finance-related prompts

### Model Sizing Parameters
- **Vocabulary Size**: 50,000 tokens
- **Hidden Size**: 1024
- **Number of Layers**: 24
- **Number of Q Heads**: 16
- **Number of KV Heads**: 8
- **FFN Dimension**: 4096
- **Maximum Sequence Length**: 2,048 tokens

## DATASET

- **Source**: umarigan/turkiye_finance_qa (Hugging Face)
- **Content**: 428 Turkish finance question–answer pairs
- **Format**: "Soru: {question}\nCevap: {answer}"

## TRAINING DETAILS

- **Optimiser**: AdamW (learning rate 1e-5)
- **Batch Size**: 2 (keeps memory usage manageable for the large configuration)
- **Gradient Clipping**: Max norm 1.0
- **Dropout**: 0.1
- **Text Generation**:
  - Top-k sampling (k = 50)
  - Top-p sampling (p = 0.9)
  - Temperature: 0.7

## USAGE

You can interact with the model via:
1. `generate_text` for standard text generation
2. Setting `think_mode=True` to enable the reasoning phase

## TECHNICAL DETAILS

### Grouped Query Attention (GQA)
Sixteen query heads and eight key/value heads share parameters to improve memory efficiency. Each KV head is reused by multiple query heads.

### Data Processing
1. Tokenise the dataset
2. Form mini-batches
3. Build attention masks
4. Apply causal masking

### Autoregressive Generation
The model predicts the next token based on previously generated tokens, following a standard autoregressive pattern.

## PERFORMANCE

After training, the model can answer finance-related questions in Turkish. Activating thinking mode yields better responses on more complex prompts.

## LIMITATIONS

- Training on CPU can be time-consuming
- The lightweight Turkish tokenizer is less expressive than subword tokenisers used in larger language models
- The dataset is relatively small (428 examples), so generalisation is limited
