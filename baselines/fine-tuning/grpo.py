from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import torch
import swanlab

model_name_or_path = 'Path/To/Your/Model'  # Replace with your model path or name
config = AutoConfig.from_pretrained(model_name_or_path)
config.sliding_window = 4096  # Or any desired window size

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, torch_dtype=torch.bfloat16,
                                             device_map="auto", trust_remote_code=True)

# 1. Check model parameters
print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load training data
train_data_path = 'PATH_TO_YOUR_DATASET'
train_dataset = load_dataset("json", data_files=train_data_path)
# train_dataset = train_dataset["train"].select(range(852, 1705))

# Load validation data
val_data_path = 'PATH_TO_YOUR_VAL_DATASET'
val_dataset = load_dataset("json", data_files=val_data_path)


# Add `query` field as input prompt
def build_prompt(example):
    prompt = f"{example['question']} \n<think>\n"
    response = f" {example['context']} \n</think>\n  {example['answer']}"
    return {"prompt": prompt, "response": response}


train_dataset = train_dataset.map(build_prompt)
val_dataset = val_dataset.map(build_prompt)

# Swanlab callback

# Initialize SwanLab project
swanlab.init(project="ProjectName")
from transformers import TrainerCallback


class SwanlabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    swanlab.log({k: v}, step=state.global_step)


# Custom reward function: call Qwen3 API
from openai import OpenAI
import re

# Set Aliyun-compatible API Key and Endpoint
client = OpenAI(
    api_key='ALIYUN_COMPATIBLE_API_KEY',  # Replace with your API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def call_chat(client, prompt_messages):
    try:
        response = client.chat.completions.create(
            model='qwen-plus',
            messages=prompt_messages,
            temperature=0.2,
            top_p=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return f"[ERROR] {e}"


def call_qwen_judge(prompt, pred, ref):
    """
    Return value should be a float representing the score.
    """
    try:
        system_prompt = """
                            ### Task Definition: Medical Q&A Answer Consistency Scoring System (based on reference answer)

                            You are a medical Q&A quality evaluation expert. You need to judge the model-generated answer (generated_answer) against the authoritative reference answer (reference_answer). Use the reference answer as the only benchmark to score alignment from 1 to 5 (1 is the lowest, 5 is the highest) based on content focus, coverage of key information, and expression consistency.

                            ### 1. Scoring Guidelines
                            Please strictly use the reference_answer as the core standard. The specific rules:
                            - 5 points: Completely aligned. The generated_answer matches the reference_answer in core conclusions, key information, and scope, with only minor wording differences.
                            - 4 points: Highly aligned. The generated_answer fully covers the core conclusions and key information of the reference_answer without adding irrelevant content, only with minor differences in sentence structure or wording.
                            - 3 points: Mostly aligned. The generated_answer contains the core conclusions of the reference_answer but omits one non-core detail or adds a small amount of irrelevant but non-conflicting information.
                            - 2 points: Low alignment. The generated_answer mentions the core topic but has vague or partially incorrect core conclusions, adds too much irrelevant information, or omits key information.
                            - 1 point: Completely misaligned. The generated_answer’s core conclusion is opposite to or omits the core content of the reference_answer.

                            ### 2. Evaluation Criteria (strictly based on reference_answer)
                            - Does it cover the core conclusion completely?
                            - Does it include key information?
                            - Does it avoid adding irrelevant information?
                            - Is the focus consistent with the reference_answer?

                            ### 3. Output format
                            Output the score directly, e.g.: 5
                        """

        user_prompt = f"Reference answer: {ref}. Model generated answer: {pred}. "
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        msg = call_chat(client, messages)
        match = re.search(r"-?\d+\.?\d*", msg)
        if match:
            return float(match.group())
        else:
            print("No match!")
            return 0.0

    except Exception as e:
        print("Judge error:", e)
        return 0.0


def reward_fn(prompts, completions, **kwargs):
    """
    Reward function for GRPOTrainer. Compares each completion with the ground truth response.
    """
    res = kwargs.get("response", [])
    prompt = kwargs.get("prompt", [])

    rewards = []
    for pred, example in zip(completions, res):
        ref = example
        score = call_qwen_judge(prompt, pred, ref)
        rewards.append(score)
    print("rewards:", rewards)
    return rewards


# GRPO configuration
grpo_config = GRPOConfig(
    output_dir="PATH_TO_SAVE_MODEL",  # Replace with your output directory
    num_generations=4,
    num_train_epochs=2,
    # learning_rate=5e-6,
    per_device_train_batch_size=4,
    generation_kwargs={
        "max_new_tokens": 5000,  # Control max generation length
        "temperature": 0.5,
        "top_p": 0.5,
        "do_sample": True
    },
)

# Start GRPO training
grpo_trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,  # Pass Dataset object directly
    eval_dataset=val_dataset,
    reward_funcs=[reward_fn],
    callbacks=[SwanlabCallback()]
)

if __name__ == "__main__":
    grpo_trainer.train()
