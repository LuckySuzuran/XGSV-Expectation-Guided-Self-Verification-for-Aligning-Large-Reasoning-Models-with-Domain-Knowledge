import json
import re
import ast
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="your_api_key")

def query(client: OpenAI, messages: List[Dict], llm_name: str, temperature: float = 0.2, top_p: float = 0.1) -> str:
    """Send query to LLM and return response content"""
    completion = client.chat.completions.create(
        model=llm_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    return completion.choices[0].message.content

# Since RJUA is a Chinese dataset, we write prompt with Chinese instructions
def extract(text: str, client: OpenAI, llm_name: str, temperature: float = 0.2, top_p: float = 0.1) -> Dict:
    messages = [
            {"role": "system",
            "content": """请从以下医疗文本中精确提取以下5类医学实体，按JSON格式输出：
            # 实体类型
            1. 疾病（Disease）：如"糖尿病"、"高血压"
            2. 症状（Symptom）：如"头痛"、"恶心"
            3. 药物（Medicine）：如"阿莫西林"、"胰岛素"
            4. 检查（Diagnostic Test）：如"血常规"、"磁共振成像"
            5. 治疗方式（Treatment）：如"手术"、"放疗"

            # 规则
            - 只输出文本中明确提及的实体，不要擅自推测文本中没有详细提及的实体
            - 使用标准术语（如"胃镜"→"上消化道内镜检查"）

            # 输出格式
            {
                "disease": [],
                "symptom": [],
                "medicine": [],
                "diagnostic_test": [],
                "treatment": []
            }

            # 示例
            输入："用户提问：我心脏疼痛，怎么办？；医生回复：建议先做心电图排除心梗，再服用阿司匹林，同时密切关注身体状况"
            输出：
            {
                "disease": ["心肌梗死"],
                "symptom": ["心脏疼痛"],
                "medicine": ["阿司匹林"],
                "diagnostic_test": ["心电图"],
                "treatment": ["密切观察"]
            }"""},
        {
            "role": "user",
            "content": text
        }
    ]

    result = query(client, messages, llm_name, temperature, top_p)
    
    # Clean up code block markers and non-JSON content
    cleaned_result = result
    if '```json' in cleaned_result:
        cleaned_result = cleaned_result.split('```json')[1]
    if '```' in cleaned_result:
        cleaned_result = cleaned_result.split('```')[0]
    
    cleaned_result = cleaned_result.strip()
    
    try:
        return json.loads(cleaned_result)
    except json.JSONDecodeError:
        if cleaned_result.startswith('"') and cleaned_result.endswith('"'):
            cleaned_result = cleaned_result[1:-1]
        
        try:
            return json.loads(cleaned_result)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(cleaned_result)
            except:
                print(f"Failed to parse LLM result: {result}")
                return {
                    "disease": [],
                    "symptom": [],
                    "medicine": [],
                    "diagnostic_test": [],
                    "treatment": []
                }
            
#Multi-threaded batch extraction of medical entities
def batch_extract(texts: List[str], client: OpenAI, llm_name: str, max_workers: int = 5, 
                 temperature: float = 0.2, top_p: float = 0.1) -> List[Dict]:
    results = [None] * len(texts)  # Pre-allocate result list
    
    def process_single(text: str, index: int) -> tuple:
        try:
            return index, extract(text, client, llm_name, temperature, top_p)
        except Exception as e:
            print(f"Error processing text (index {index}): {str(e)}")
            return index, {
                "disease": [],
                "symptom": [],
                "medicine": [],
                "diagnostic_test": [],
                "treatment": []
            }

    # Process in parallel with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single, text, idx)
            for idx, text in enumerate(texts)
        ]
        
        # Show progress bar
        for future in tqdm(as_completed(futures), total=len(texts), desc="Processing"):
            idx, result = future.result()
            results[idx] = result  # Preserve original order

    return results

# Load and process dataset
if __name__ == "__main__":
    # Load training data
    RJUA_train = []
    with open("../dataset/rjua/RJUA_train.json", "r", encoding="utf-8") as f:
        for line in f:  # Read line by line
            if line.strip():  # Skip empty lines
                RJUA_train.append(json.loads(line))  # Parse JSON line
        
    # Load test data
    RJUA_test = []
    with open("../data/rjua/RJUA_test.json", "r", encoding="utf-8") as f:
        for line in f:  # Read line by line
            if line.strip():  # Skip empty lines
                RJUA_test.append(json.loads(line))  # Parse JSON line
        
    # Combine and process all data
    RJUA_all = RJUA_train + RJUA_test
    RJUA_all_texts = [
        f'Question: {item["question"]}; Context: {item["context"]}; Answer: {item["answer"]}; Disease: {item["disease"]}; Advice: {item["advice"]}.'
        for item in RJUA_all
    ]
    
    # Extract entities in batch
    all_keywords = batch_extract(RJUA_all_texts, client, "qwen-plus", 5)
    
    # Collect all unique entities
    all_entities = []
    for entity_dict in all_keywords:
        for entities in entity_dict.values():
            all_entities.extend(entities)
    all_entities = set(all_entities)
    
    # Save entities to file
    import pickle as pkl
    with open("../data/rjua/rjua_entities.pkl", "wb") as f:
        pkl.dump(all_entities, f)
