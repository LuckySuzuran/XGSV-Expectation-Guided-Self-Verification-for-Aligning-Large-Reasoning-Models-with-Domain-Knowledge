from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pathlib import Path
import os
client = OpenAI(api_key="your_api_key")
import pickle

def query(client: OpenAI, messages: List[Dict], llm_name: str, 
          temperature: float = 0.2, top_p: float = 0.1) -> str:
    completion = client.chat.completions.create(
        model=llm_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    return completion.choices[0].message.content

def explain_concept(concept: str, client: OpenAI, llm_name: str = "qwen-plus", 
                   temperature: float = 0.2, top_p: float = 0.1, language: str = "en") -> str:
    if language == "zh":
        system_prompt = ("你是一个医学专家。接下来，用户会向你输入一个医学概念。请你对这个概念进行100字以内的详细解释。请确保内容准确且专业。不要输出其他内容。直接输出解释。"
                        "例如，用户输入“糖尿病”，理想的输出是：“糖尿病是一组以慢性高血糖为特征的代谢性疾病，主要由胰岛素分泌不足或作用障碍引起。长期高血糖可导致多系统损害，如心血管、肾脏、神经及视网膜病变。主要类型包括1型（胰岛素依赖型）和2型（非胰岛素依赖型），需通过饮食控制、药物治疗及血糖监测综合管理。” "
                        "如果用户输入是非医学概念，请直接回复“这是与医学无关的概念。”")
    else:
        system_prompt = (
            "You are a medical expert. Provide a concise (≤200 words) explanation of the medical concept. "
            "Ensure accuracy and professionalism. Output only the explanation.\n"
            "Example: For 'diabetes', output: 'Diabetes is a group of metabolic diseases characterized by chronic hyperglycemia, primarily caused by insufficient insulin secretion or action. Prolonged hyperglycemia can lead to multi-system damage, including cardiovascular, renal, neurological, and retinal complications. The main types include Type 1 (insulin-dependent) and Type 2 (non-insulin-dependent), requiring comprehensive management through dietary control, medication, and blood glucose monitoring.'"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": concept}
    ]
    
    response = query(client, messages, llm_name, temperature, top_p)
    print(f"Concept: {concept}\nExplanation: {response.strip()}\n")
    return response.strip()

# Batch explain medical concepts using multithreading
def batch_explain_concepts(concepts: List[str], client: OpenAI, max_workers: int = 25, 
                          llm_name: str = "qwen-plus", temperature: float = 0.2, 
                          top_p: float = 0.1, language: str = "en") -> Dict[str, str]:
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create future-concept mapping
        future_to_concept = {
            executor.submit(explain_concept, concept, client, llm_name, temperature, top_p, language): concept 
            for concept in concepts
        }
        
        # Process results as they complete
        for future in as_completed(future_to_concept):
            concept = future_to_concept[future]
            try:
                results[concept] = future.result()
            except Exception as e:
                print(f"Error explaining '{concept}': {str(e)}")
                results[concept] = f"Explanation failed: {str(e)}"
    
    return results

if __name__ == "__main__":
    # RJUA
    with open("../data/rjua/rjua_entities.pkl", "rb") as f:
        entities = pickle.load(f)
    
    concept_list = list(set(entities))
    explanations = batch_explain_concepts(
        concepts=concept_list,
        client=client,
        max_workers=5,
        llm_name="qwen-plus",
        language="en"
    )
    
    with open("../data/rjua/entity_explanations.pkl", "wb") as f:
        pickle.dump(explanations, f)
    
    # GenMedGPT
    data_path = "../data/genmedgpt/EMCKG/entity2id.txt"
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entities = [line.strip().split("\t")[0] for line in lines if line.strip()]

    relation_path = "../data/genmedgpt/EMCKG/relation2id.txt"
    with open(relation_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    relations = [line.strip().split("\t")[0] for line in lines if line.strip()]
    
    entities_explanation = batch_explain_concepts(entities, max_workers = 20,
                                                            client=client, llm_name="qwen-plus",
                                                            temperature=0.2, top_p=0.1, language = "en")
    with open("../data/genmedgpt/EMCKG/entities_explanation.pkl", "wb") as f:
        pickle.dump(entities_explanation, f)

    relations_explanation = batch_explain_concepts(relations, max_workers = 20,
                                                             client=client, llm_name="qwen-plus",
                                                             temperature=0.2, top_p=0.1, language = "en")
    
    with open("../data/genmedgpt/EMCKG/relation_explanation.pkl", "wb") as f:
        pickle.dump(relations_explanation, f)
