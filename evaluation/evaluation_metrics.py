from typing import Set, List, Tuple
from collections import Counter
from bert_score import score  # pip install bert-score
import numpy as np
import pandas as pd
import re
import json
import concurrent.futures
from zhipuai import ZhipuAI
from json.decoder import JSONDecodeError
import jieba
from openai import OpenAI

# ---------- BERTScore ----------
def calc_bertscore(hyps: List[str],
                   refs: List[str],
                   lang: str = "zh",
                   model_type: str = 'bert-base-chinese',
                   device = "cuda",
                   batch_size = 16) -> float:
    P, R, F1 = score(
                    hyps, refs,
                    model_type=model_type,
                    lang=lang,
                    device= device,       
                    batch_size=batch_size,        
                    verbose=True
                )
    return float(P.mean()), float(R.mean()), float(F1.mean())


# ---------- LLM Judge ----------
api_key = "xxx"
client = OpenAI(api_key=api_key)
system_prompt_cn = """
    ### 任务定义：医疗问答答案一致性评分系统（以参考回答为核心基准）

    你是一个医学问答质量评估专家，需要以权威参考答案（reference_answer）为唯一核心基准，评判模型生成的答案（generated_answer）与参考回答的吻合程度，从内容聚焦性、核心信息覆盖率和表述一致性角度打出1-5之间的整数评分（1分为最低，5分为最高）。

    ### 一、评分说明
    请严格以参考回答的核心内容、信息范围和表述重点为标准，判断generated_answer的吻合度，具体标准如下：
    - 5分：完全吻合。generated_answer与reference_answer在核心结论（如病因判断）、关键信息（如疾病属性、症状关联）、信息范围（不增不减）上完全一致，仅存在无关紧要的措辞差异。
    - 4分：高度吻合。generated_answer完整覆盖reference_answer的核心结论和关键信息，未添加无关内容，仅存在次要细节（如句式调整、用词微调）的轻微差异。
    - 3分：基本吻合。generated_answer包含reference_answer的核心结论，但遗漏1处非核心细节，或添加少量无关但无冲突的信息，未偏离核心方向。
    - 2分：低度吻合。generated_answer虽提及reference_answer的核心主题，但核心结论模糊或部分错误，添加较多无关信息，或遗漏关键信息（如疾病属性、症状关联）。
    - 1分：完全不吻合。generated_answer的核心结论与reference_answer完全相反，或未提及reference_answer的核心内容，信息方向完全偏离。

    ### 二、判断依据（严格围绕参考回答）
    - 是否完整覆盖reference_answer的核心结论（如具体病因）；
    - 是否包含reference_answer的关键信息（如疾病的性质、症状间的关联）；
    - 是否未添加reference_answer以外的无关信息（如其他病因、额外建议）；
    - 表述重点是否与reference_answer一致（如是否聚焦于核心病因）。

    ### 三、输出格式
    请严格按如下格式输出：
    直接输出分数，比如：5

  """
system_prompt_en = """
    ### Task Definition: Medical Q&A Answer Consistency Scoring System (with reference answer as core benchmark)

    You are a medical Q&A quality assessment expert. You need to take the authoritative reference answer (reference_answer) as the sole core benchmark to evaluate the consistency between the model-generated answer (generated_answer) and the reference answer. You will give an integer score between 1-5 (1 being the lowest, 5 being the highest) from the perspectives of content focus, core information coverage, and expression consistency.

    ### I. Scoring Instructions
    Please strictly use the core content, information scope, and expression focus of the reference answer as criteria to judge the consistency of the generated_answer. The specific standards are as follows:
    - 5 points: Completely consistent. The generated_answer is completely consistent with the reference_answer in terms of core conclusions (e.g., etiological judgment), key information (e.g., disease attributes, symptom associations), and information scope (no additions or omissions), with only insignificant differences in wording.
    - 4 points: Highly consistent. The generated_answer fully covers the core conclusions and key information of the reference_answer, without adding irrelevant content, and only has minor differences in secondary details (e.g., sentence structure adjustments, minor wording changes).
    - 3 points: Basically consistent. The generated_answer contains the core conclusion of the reference_answer but misses one non-core detail, or adds a small amount of irrelevant but non-conflicting information, without deviating from the core direction.
    - 2 points: Lowly consistent. Although the generated_answer mentions the core topic of the reference_answer, the core conclusion is vague or partially incorrect, with many irrelevant information added, or key information (e.g., disease attributes, symptom associations) missing.
    - 1 point: Completely inconsistent. The core conclusion of the generated_answer is completely opposite to that of the reference_answer, or it does not mention the core content of the reference_answer, and the information direction is completely deviated.

    ### II. Judgment Basis (strictly centered on the reference answer)
    - Whether it fully covers the core conclusions of the reference_answer (e.g., specific etiology);
    - Whether it contains the key information of the reference_answer (e.g., nature of the disease, associations between symptoms);
    - Whether it does not add irrelevant information beyond the reference_answer (e.g., other etiologies, additional suggestions);
    - Whether the expression focus is consistent with the reference_answer (e.g., whether it focuses on the core etiology).

    ### III. Output Format
    Please output strictly in the following format:
    Directly output the score, e.g., 5
  """

def query(client, system_prompt, user_question, reference_answer, generated_answer,
          llm_name = "deepseek-chat", temperature=0.2, top_p=0.1):
    messages = [{"role":"system", "content":system_prompt}]
    user_prompt = f"User's question: {user_question}. Reference_answer: {reference_answer}. Generated_answer: {generated_answer}"
    messages.append({"role":"user", "content":user_prompt})
    completion = client.chat.completions.create(
        model=llm_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p
    )
    return completion.choices[0].message.content

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any


def process_single_query(args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = query(
            client=args["client"],
            system_prompt=args["system_prompt"],
            user_question=args["user_question"],
            reference_answer=args["reference_answer"],
            generated_answer=args["generated_answer"],
            llm_name=args.get("llm_name", "deepseek-chat"),
            temperature=args.get("temperature", 0.2),
            top_p=args.get("top_p", 0.1)
        )
        return {
            "input_data": args,
            "result": result,
            "error": None
        }
    except Exception as e:
        return {
            "input_data": args,
            "result": None,
            "error": str(e)
        }

def batch_query_multithreaded(
    client,
    system_prompt: str,
    user_questions: List[str],
    reference_answers: List[str],
    generated_answers: List[str],
    llm_name: str = "deepseek-chat",
    temperature: float = 0.2,
    top_p: float = 0.1,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    assert len(user_questions) == len(reference_answers) == len(generated_answers), \
    
    tasks = []
    for i in range(len(user_questions)):
        task_args = {
            "client": client,
            "system_prompt": system_prompt,
            "user_question": user_questions[i],
            "reference_answer": reference_answers[i],
            "generated_answer": generated_answers[i],
            "llm_name": llm_name,
            "temperature": temperature,
            "top_p": top_p
        }
        tasks.append(task_args)
    
    # 使用线程池处理
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_query, task) for task in tasks]
        for future in futures:
            results.append(future.result())
    
    return results