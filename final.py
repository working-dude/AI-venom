import json
with open("merged_data.json", "r") as f:
  data = json.load(f)
# Install Libraries (if not already installed)

from transformers import T5ForConditionalGeneration, T5Tokenizer

# Define Model Name and Path
model_name = "t5-base"
model_path = f"hf-hub:{model_name}"
# Load Tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load Dataset (replace with your dataset loading)

max_length=512
tokenized_data = []
for item in data:
    # Tokenize COBOL code and documentation separately
    code_tokens = tokenizer(item["cobol_code"], truncation=True, padding="max_length", max_length=max_length)
    doc_tokens = tokenizer(item["documentation"], truncation=False, padding="max_length", max_length=max_length)
    padded_code_tokens = {k: v[:max_length] for k, v in code_tokens.items()}
    padded_doc_tokens = {k: v[:max_length] for k, v in doc_tokens.items()}  # Truncate if needed

tokenized_data.append({
        "input_ids": padded_code_tokens["input_ids"],
        "attention_mask": padded_code_tokens["attention_mask"],
        "labels": padded_doc_tokens["input_ids"]  # Doc tokens as labels for prediction
    })
train_data = tokenized_data  # Assign the preprocessed data to train_data

# Load Pre-trained Model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define Training Parameters (adjust hyperparameters as needed)
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=8,
    save_steps=500,
    num_train_epochs=3,
)

# Fine-tune the Model
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()

# Save the Fine-tuned Model (optional)
model.save_pretrained("./fine-tuned_model")

code_text = """
IDENTIFICATION DIVISION.
PROGRAM-ID. HELLO-WORLD.
PROCEDURE DIVISION.
DISPLAY 'Hello, World!'.
STOP RUN.
"""

# Preprocess the code using your tokenizer (same logic as in training)
def preprocess_input(text):
  inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
  return inputs

encoded_input = preprocess_input(code_text)
generated_text = model.generate(**encoded_input)

# Decode the generated tokens back to human-readable text
text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(text)
