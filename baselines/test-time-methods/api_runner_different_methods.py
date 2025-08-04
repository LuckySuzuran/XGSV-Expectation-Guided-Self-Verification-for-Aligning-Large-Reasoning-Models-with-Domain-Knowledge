import argparse
import asyncio
import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from torch.cuda import temperature
from tqdm import tqdm

import nest_asyncio
nest_asyncio.apply()

# Set Aliyun-compatible API Key and Endpoint
client = AsyncOpenAI(
    api_key='ALIYUN_COMPATIBLE_API_KEY',  # Replace with your API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MODEL_NAME = "MODEL_NAME"  # Replace with your model name

# -------------------------
# Single chat-completion call
# -------------------------
async def call_chat(client, model, prompt_messages, max_tokens=20000, temperature=0.2):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message
    except Exception as e:
        return f"[ERROR] {e}"

async def call_chat_qwen(client, model, prompt_messages):
    try:
        reasoning_content = ""  # Store full reasoning process
        answer_content = ""     # Store full final answer

        is_answering = False    # Flag to check if reasoning has ended and answering has started
        # Create chat completion request
        completion = await client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            extra_body={"enable_thinking": True},
            stream=True,
        )
        async for chunk in completion:
            # If chunk.choices is empty, print usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                else:
                    # Start generating answer
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                    answer_content += delta.content
        return reasoning_content, answer_content
    except Exception as e:
        return f"[ERROR] {e}", f"[ERROR] {e}"


async def call_con_chat(client, prompt_messages):
    try:
        response = await client.chat.completions.create(
            model='qwen-plus',
            messages=prompt_messages,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {e}"

# -------------------------
# Base method
# -------------------------
async def base_task(client, model, user_prompt: str, true_response: str, true_cot: str):
    messages = [{"role": "user", "content": f"Answer this medical question: {user_prompt}"}]
    msg = await call_chat(client, model, messages)
    response = getattr(msg, "content", "")
    cot = getattr(msg, "reasoning_content", "")

    return {
        "Question": user_prompt,
        "True_CoT": true_cot,
        "True_Response": true_response,
        "Generated_CoT": cot,
        "Generated_Response": response
    }

# -------------------------
# Self-Refine method
# -------------------------
async def self_refine_task(client, model, user_prompt: str, true_response: str, true_cot: str):
    messages = [{"role": "user", "content": f"Answer this medical question: {user_prompt}"}]
    draft_msg = await call_chat(client, model, messages)
    draft = getattr(draft_msg, "content", "")
    refine_prompt = f"The medical question is: {user_prompt}. A previous answer was: {draft}. Please identify potential issues or incomplete points in this answer, and refine the original answer to address these issues."
    messages = [{"role": "user", "content": refine_prompt}]
    refined_msg = await call_chat(client, model, messages)
    refined_response = getattr(refined_msg, "content", "")
    refined_cot = getattr(refined_msg, "reasoning_content", "")
    return {
        "Question": user_prompt,
        "True_CoT": true_cot,
        "True_Response": true_response,
        "Generated_CoT": refined_cot,
        "Generated_Response": refined_response
    }


# -------------------------
# Self-Consistency method
# -------------------------
async def self_consistency_task(client, model, user_prompt: str, true_response: str, true_cot: str, gen_num=3):
    messages = [{"role": "user", "content": f"Answer this medical question: {user_prompt}"}]
    responses = []
    cots = []
    for i in range(gen_num):
        msg = await call_chat(client, model, messages, temperature=0.7)
        response = getattr(msg, "content", "")
        cot = getattr(msg, "reasoning_content", "")
        responses.append(response)
        cots.append(cot)

    con_cot_prompt = f"Answer this medical question: {user_prompt}. Below are {gen_num} chains of thought. Please synthesize their common points and expressions into a more accurate and consistent final chain of thought:\n\n"
    for i, s in enumerate(cots, 1):
        con_cot_prompt += f"{i}. {s}\n"

    con_res_prompt = f"Answer this medical question: {user_prompt}. Below are {gen_num} answers. Please synthesize their common points and expressions into a more accurate and consistent final answer:\n\n"
    for i, s in enumerate(responses, 1):
        con_res_prompt += f"{i}. {s}\n"

    con_res_messages = [{"role": "user", "content": f"{con_res_prompt}"}]
    con_response = await call_con_chat(client, con_res_messages)

    con_cot_messages = [{"role": "user", "content": f"{con_cot_prompt}"}]
    con_cot = await call_con_chat(client, con_cot_messages)
    return {
        "Question": user_prompt,
        "True_CoT": true_cot,
        "True_Response": true_response,
        "Generated_CoT": con_cot,
        "Generated_Response": con_response
    }


# -------------------------
# Filter function
# -------------------------
def is_chinese(example):
    for char in example["Question"]:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


# -------------------------
# Main function: run multiple prompts concurrently
# -------------------------
async def run(args):
    batch_size = 50
    results = []

    client = AsyncOpenAI(
        api_key='ALIYUN_COMPATIBLE_API_KEY',  # Replace with your API Key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    data_path = "PATH_TO_YOUR_DATASET"  # Replace with your dataset path
    output_path = "PATH_TO_YOUR_OUTPUT"  # Replace with your output path

    dataset = load_dataset("json", data_files=data_path)
    if args.dataset == 'medical_o1_sft_Chinese':
        dataset1 = dataset.filter(is_chinese)
        test_set = dataset1["train"].train_test_split(test_size=0.2, seed=42)['test']

        prompts = test_set['Question']
        cots = test_set['Complex_CoT']
        responses = test_set['Response']

    elif args.dataset == 'RJUA_test':
        test_set = dataset["train"]

        prompts = test_set['question']
        cots = test_set['context']
        responses = test_set['answer']

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch Processing"):
        if args.method == 'base':
            batch_tasks = [
                base_task(client, args.model, prompts[j], responses[j], cots[j])
                for j in range(i, min(i + batch_size, len(prompts)))
            ]
            batch_results = await asyncio.gather(*batch_tasks)

        elif args.method == 'self_refine':
            batch_tasks = [
                self_refine_task(client, args.model, prompts[j], responses[j], cots[j])
                for j in range(i, min(i + batch_size, len(prompts)))
            ]
            batch_results = await asyncio.gather(*batch_tasks)

        elif args.method == 'self_consistency':
            batch_tasks = [
                self_consistency_task(client, args.model, prompts[j], responses[j], cots[j])
                for j in range(i, min(i + batch_size, len(prompts)))
            ]
            batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Results saved to {output_path}")


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run API with LRM models.")
    parser.add_argument("--dataset", type=str, default="PATH_TO_YOUR_DATASET")  # dataset name
    parser.add_argument("--model", type=str, default="PATH_TO_YOUR_MODEL", help="Model name or path: deepseek-r1-distill-qwen-7b, qwen3-8b")  # model name or path
    parser.add_argument("--method", type=str, default="NAME_OF_YOUR_METHOD", help="Method to use: base, self_refine, self_consistency")
    args = parser.parse_args()

    asyncio.run(run(args))
