import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline

# Ayarlar
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DATA_PATH = "data.jsonl"  # JSONL dosya yolu
OUTPUT_DIR = "finetuned-llm"

# 1. JSONL verisini yükle
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

# 2. Dataset'e çevir
# Prompt formatını daha belirgin ve modelin öğrenebileceği şekilde ayarlıyoruz.
def to_prompt(example):
    prompt = (
        f"[BAŞLIK] {example['title']}\n"
        f"[ÖZET] {example['summary']}\n"
        f"[İÇERİK] {example['content']}\n"
        f"[ETİKETLER] {', '.join(example['tags'])}"
    )
    return {"text": prompt}

dataset = Dataset.from_list([to_prompt(e) for e in lines])

# 3. Tokenizer ve model yükle
# pad_token ayarını koru

# Tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Model yükle
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 4. Tokenize fonksiyonu
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

# 5. Eğitim argümanları
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,  # Daha fazla epoch ile küçük veri için daha iyi öğrenme
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

# 7. Eğitimi başlat
trainer.train()

# 8. Modeli kaydet
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model {OUTPUT_DIR} klasörüne kaydedildi.")

# === TEST KODU ===
def test_model():
    print("Test başlatılıyor...")
    # Eğitilmiş modeli ve tokenizer'ı yükle
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    # Test promptunu yeni formatla oluştur
    prompt = (
        "[BAŞLIK] Yapay Zeka ve 2025\n"
        "[ÖZET] 2025 yılında yapay zeka alanında beklenen gelişmeler\n"
        "[İÇERİK]"
    )
    output = generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
    print("\n--- Model Çıktısı ---")
    # Sadece [İÇERİK] kısmından sonrasını al
    generated = output[0]['generated_text']
    if "[İÇERİK]" in generated:
        generated = generated.split("[İÇERİK]")[1]
    print(generated.strip())

if __name__ == "__main__":
    test_model()