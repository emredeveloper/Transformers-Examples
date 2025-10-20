import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline

# Settings
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DATA_PATH = "data.jsonl"  # Path to the JSONL file
OUTPUT_DIR = "finetuned-llm"

# 1. Load the JSONL data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# 2. Convert to a Hugging Face Dataset
# Format each prompt clearly so the model can learn the structure.
def to_prompt(example):
    prompt = (
        f"[TITLE] {example['title']}\n"
        f"[SUMMARY] {example['summary']}\n"
        f"[CONTENT] {example['content']}\n"
        f"[TAGS] {', '.join(example['tags'])}"
    )
    return {"text": prompt}

dataset = Dataset.from_list([to_prompt(e) for e in lines])

# 3. Load the tokenizer and model
# Preserve the pad_token configuration

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Load the model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 4. Tokenization function
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,  # Additional epochs help smaller datasets learn better
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
    report_to=[],
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 7. Start training
trainer.train()

# 8. Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}.")

# === TEST CODE ===
def test_model():
    print("Starting evaluation test...")
    # Load the fine-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    # Build the prompt using the new format
    prompt = (
        "[TITLE] Artificial Intelligence and 2025\n"
        "[SUMMARY] Expected developments in artificial intelligence in 2025\n"
        "[CONTENT]"
    )
    output = generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
    print("\n--- Model Output ---")
    # Extract only the portion after [CONTENT]
    generated = output[0]['generated_text']
    if "[CONTENT]" in generated:
        generated = generated.split("[CONTENT]")[1]
    print(generated.strip())

if __name__ == "__main__":
    test_model()