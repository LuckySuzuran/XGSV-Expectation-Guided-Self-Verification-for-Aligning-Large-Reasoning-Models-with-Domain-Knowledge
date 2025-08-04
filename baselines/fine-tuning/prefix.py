from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
from peft import get_peft_model, TaskType, PrefixTuningConfig

model_path = 'PATH_TO_YOUR_MODEL'  # Replace with your model path

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure there is a pad_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device)


def preprocess(example, MAX_LENGTH=8192):
    prompt = f"Question: {example['questions']}\n \n <think>\n</think>\n\nAnswer: {example['answers']}"

    instruction = tokenizer(f"{example['questions']}", add_special_tokens=False, truncation=True, padding=True)
    response = tokenizer(f"\n <think> \n</think>\n\nAnswer: {example['answers']}", add_special_tokens=False,
                         truncation=True, padding=True)

    input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # Truncate if exceeding MAX_LENGTH
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def preprocess_(example, MAX_LENGTH=8192):
    prompt = f"Question: {example['question']}\n \n <think> \n {example['context']}\n</think>\n\nAnswer: {example['answer']}"

    instruction = tokenizer(f"{example['question']}", add_special_tokens=False, truncation=True, padding=True)
    response = tokenizer(f"\n <think> \n {example['context']}\n</think>\n\nAnswer: {example['answer']}",
                         add_special_tokens=False, truncation=True, padding=True)

    input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # Truncate if exceeding MAX_LENGTH
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


path = "PATH_TO_YOUR_DATASET"  # Replace with your dataset path
dataset = load_dataset("json", data_files=path)
dataset_ = dataset.map(preprocess)

# Build Prefix-Tuning configuration
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,  # Training mode
    num_virtual_tokens=20,  # Number of virtual tokens per layer
    prefix_projection=True,  # Whether to use prefix projection layer
)

# Build model with prefix parameters
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Build Trainer
training_args = TrainingArguments(
    output_dir="PATH_TO_SAVE_MODEL",  # Replace with your output path
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    bf16=True,
    save_strategy="epoch"
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()
