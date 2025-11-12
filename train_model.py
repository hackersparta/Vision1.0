import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
from docx import Document

# ============================
# Paths & Config
# ============================
BASE_DIR = os.path.dirname(__file__)
MODEL_NAME = os.path.join(BASE_DIR, "models", "llama-3.2-1b-instruct")
DATA_FILE = os.path.join(BASE_DIR, "my_finetune_data.json")
DOC_FOLDER = os.path.join(BASE_DIR, "documents")
OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tuned_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Device setup
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================
# Load Base Model & Tokenizer
# ============================
print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if device == "cuda" else None,
    local_files_only=True
)

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, lora_config)
model.to(device)

# ============================
# Load Documents
# ============================
def extract_text_from_docs(folder):
    texts = []
    if not os.path.exists(folder):
        return texts

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            try:
                reader = PdfReader(path)
                content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                texts.append(content)
            except Exception as e:
                print(f"PDF error {file}: {e}")
        elif file.endswith(".docx"):
            try:
                doc = Document(path)
                content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                texts.append(content)
            except Exception as e:
                print(f"DOCX error {file}: {e}")
        elif file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

print("Loading and splitting documents...")
raw_docs = extract_text_from_docs(DOC_FOLDER)

# Simple text splitting
def split_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

doc_chunks = []
for d in raw_docs:
    doc_chunks.extend(split_text(d))
print(f"Loaded and split {len(raw_docs)} documents into {len(doc_chunks)} chunks.")

# Save chunks
with open(os.path.join(BASE_DIR, "document_chunks.json"), "w", encoding="utf-8") as f:
    json.dump(doc_chunks, f, ensure_ascii=False, indent=2)

# ============================
# Prepare Training Data
# ============================
print("Loading fine-tune data...")
train_data = []
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        for item in raw_data:
            q = item["messages"][0]["content"]
            a = item["messages"][1]["content"]
            train_data.append({"text": f"Question: {q}\nAnswer: {a}"})
else:
    print("No fine-tune data found, training only on docs.")

# Combine document chunks with fine-tune data
for c in doc_chunks:
    train_data.append({"text": c})

print(f"Loaded {len(train_data)} total training examples.")

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(train_data)

# ============================
# Tokenization Function
# ============================
def tokenize_function(examples):
    # Tokenize input text
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Add labels for loss computation
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ============================
# Training Config
# ============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=5,
    learning_rate=5e-5,
    fp16=False,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none"
)

# ============================
# Trainer Setup
# ============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# ============================
# Start Training
# ============================
print("Starting fine-tuning...")
trainer.train()

# ============================
# Save Model
# ============================
print("Saving fine-tuned model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved at: {OUTPUT_DIR}") 