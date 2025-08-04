from openai import OpenAI
import networkx as nx
import concurrent.futures
from typing import List, Tuple
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# ===== Configuration =====
api_key = "YOUR_ALIYUN_COMPATIBLE_API_KEY"  # Replace with your Aliyun-compatible API Key
MODEL_NAME = "MODEL_NAME"  # Replace with your model name
BEAM_SIZE = 3
MAX_DEPTH = 3
SCORE_THRESHOLD = 0.7

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ===== Your custom knowledge graph as triples =====
# Format: List of (head, relation, tail)
# triples: List[Tuple[str, str, str]]

triples: List[Tuple[str, str, str]] = []
try:
    with open("relation.txt", "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue  # skip empty or comment lines
            if len(parts) != 3:
                raise ValueError(f"Line {lineno}: expected 3 columns, got {len(parts)}: {line!r}")
            head, rel, tail = parts
            triples.append((head, rel, tail))
except FileNotFoundError:
    raise FileNotFoundError("relation.txt not found. Please place your triples file in the working directory.")

# ===== Build directed graph =====
G = nx.DiGraph()
for head, rel, tail in triples:
    G.add_edge(head, tail, relation=rel)

# ===== Prepare entity and relation lists for System message =====
ENTITIES = sorted(G.nodes)
RELS = sorted({rel for _, rel, _ in triples})
SYSTEM_PROMPT = (
    "You are an AI assistant with knowledge of the following entities and relations.\n"
    f"Entities: {', '.join(ENTITIES)}\n"
    f"Relations: {', '.join(RELS)}\n"
    "When identify topic entities, or generating or scoring paths, only use items from these certain lists."
)

# ===== Prompt templates =====
PROMPTS = {
    "init": (
        "Question: {question}\n"
        "Identify the main topic entities in the question (in English). "
        "If there are multiple, separate them with semicolons."
    ),
    "explore": (
        "Current reasoning path: {path}\n"
        "List all possible outgoing relations and neighbor entities from '{entity}'. "
        "Format each line as: relation -> neighbor."
    ),
    "score": (
        "Evaluate the usefulness of this reasoning path (0.0 to 1.0):\n"
        "Path: {path}\n"
        "Question: {question}\n"
        "Provide only a numeric score."
    ),
    "answer": (
        "Using the reasoning path: {path}\n"
        "Provide the answer to the question: {question}\n"
        "Then briefly explain the reasoning steps."
    )
}


# ===== LLM helper =====
def llm_explore_generate(prompt: str, max_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {e}"


def llm_reason_generate(prompt: str, max_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] {e}"


# ===== Think-on-Graph implementation =====
class GraphReasoner:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def init_entities(self, question: str) -> List[str]:
        text = llm_explore_generate(PROMPTS['init'].format(question=question))
        ents = [e.strip() for e in text.split(';') if e.strip() in ENTITIES]
        return ents or ENTITIES

    def explore(self, node: str) -> List[Tuple[str, str]]:
        if node not in self.graph: return []
        return [(data['relation'], nbr) for nbr, data in self.graph[node].items()]

    def path_to_str(self, path: List[str]) -> str:
        parts = []
        for a, b in zip(path, path[1:]):
            rel = self.graph[a][b]['relation']
            parts.append(f"{a} -[{rel}]-> {b}")
        return " ; ".join(parts)

    def score_paths(self, paths: List[List[str]], question: str) -> List[Tuple[float, List[str]]]:
        def score_one(p: List[str]) -> Tuple[float, List[str]]:
            path_str = self.path_to_str(p)
            prompt = PROMPTS['score'].format(path=path_str, question=question)
            try:
                score = float(llm_reason_generate(prompt))
            except:
                score = 0.0
            return score, p

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(score_one, paths))

        return sorted(results, key=lambda x: x[0], reverse=True)

    def generate_answer(self, path: List[str], question: str) -> str:
        return llm_reason_generate(PROMPTS['answer'].format(path=self.path_to_str(path), question=question))

    def beam_search(self, question: str) -> Tuple[str, List[str]]:
        beams = [[e] for e in self.init_entities(question)[:BEAM_SIZE]]
        for _ in range(MAX_DEPTH):
            cand = []
            for path in beams:
                for rel, nbr in self.explore(path[-1]):
                    cand.append(path + [nbr])
            if not cand: break
            scored = self.score_paths(cand, question)
            beams = [p for _, p in scored[:BEAM_SIZE]]
            if scored[0][0] >= SCORE_THRESHOLD:
                return self.generate_answer(scored[0][1], question), scored[0][1]
        if beams:
            best_score, best_path = self.score_paths(beams, question)[0]
            return self.generate_answer(best_path, question), best_path
        return "", []


# ===== Main Execution =====
if __name__ == '__main__':
    data_path = "PATH/TO/YOUR/DATASET.json"  # Replace with your dataset path
    dataset = load_dataset("json", data_files=data_path)

    test_set = dataset["train"]

    prompts = test_set['questions']
    responses = test_set['answers']

    reasoner = GraphReasoner(G)
    results = []
    for i in tqdm(range(0, len(prompts)), desc="Batch Processing"):
        # print(prompts[i], responses[i])
        answer, path = reasoner.beam_search(prompts[i])
        # print(answer)
        results.append({
                        "Question": prompts[i],
                        "True_Response": responses[i],
                        "Generated_Response": answer,
                        "Generated_Path": path
                    })

    df = pd.DataFrame(results)
    output_df = df.applymap(lambda x: x.replace('\n', '\\n') if isinstance(x, str) else x)
    output_df.to_csv("Path/To/Csv", index=False, encoding="utf-8-sig")