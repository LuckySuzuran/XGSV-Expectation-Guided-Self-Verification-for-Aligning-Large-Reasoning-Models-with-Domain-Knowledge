from model import*
from utils import *
import argparse

def matrix_to_sparse(matrix: torch.Tensor) -> dict:
    # PyTorch COO
    sparse_matrix = matrix.to_sparse()
    return {
        'indices': sparse_matrix.indices(),
        'values': sparse_matrix.values(),
        'size': sparse_matrix.size()
    }

def sparse_to_matrix(sparse_repr: dict) -> torch.Tensor:
    # coo to dense
    sparse_matrix = torch.sparse_coo_tensor(
        indices=sparse_repr['indices'],
        values=sparse_repr['values'],
        size=sparse_repr['size']
    )
    return sparse_matrix.to_dense()

def predict_graph(question_embeddings, question_masks, model):
    model.eval()
    sparse_relation_matrices = []

    with torch.no_grad():
        for i in range(question_embeddings.shape[0]):
            question_embedding = question_embeddings[i].unsqueeze(0)
            question_mask = question_masks[i].unsqueeze(0)
            relation_matrices = model.get_relation_matrices(question_embedding.to(torch.bfloat16),
                                                            question_mask.to(torch.bfloat16))
            relation_matrix = relation_matrices[0].detach().cpu()
            sparse_relation_matrices.append(matrix_to_sparse(relation_matrix))
            if i % 50 == 0:
                print(i)
    return sparse_relation_matrices


def main():
    parser = argparse.ArgumentParser(description='Predict graph relations for a specific dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Specify the dataset name (e.g., GenMedGPT-5k, RJUA)')
    args = parser.parse_args()
    if args.dataset == "GenMedGPT-5k":
        device = "cuda"
        entity_vectors = torch.load("xxx.pt")


        with open("../../dataset/GenMedGPT-5k/GenMedGPT_data.pkl", "rb") as f:
            genmedgpt_data = pkl.load(f)

        #trainset
        genmedgpt_data_train_questions = genmedgpt_data["train_questions"]

        import random
        random.seed(42)
        len_genmedgpt_train = len(genmedgpt_data_train_questions)
        expected_graph_train_indices = random.sample(range(len_genmedgpt_train), len_genmedgpt_train//2)
        all_indices = set(range(len_genmedgpt_train))  # 所有可能的索引（转为集合，方便求差集）
        reasoning_module_indices = list(all_indices - set(expected_graph_train_indices))  # 差集：未选中的索引
        reasoning_module_indices.sort()

        import pickle
        with open("genmedgpt_questions_embeddings_qwen3embmodel.pkl", "rb") as f:
            questions_embeddings_qwen3embmodel = pickle.load(f)
        answers_embeddings_qwen3embmodel = torch.load("genmedgpt_answers_embeddings.pth")

        genmedgpt_expected_graph_train_question_embeddings = questions_embeddings_qwen3embmodel[1][expected_graph_train_indices].to(device)
        genmedgpt_expected_graph_train_question_masks = questions_embeddings_qwen3embmodel[2][expected_graph_train_indices].to(device)
        genmedgpt_expected_graph_train_answer_embeddings = answers_embeddings_qwen3embmodel[expected_graph_train_indices].to(device)

        #reasoning module trainset
        genmedgpt_reasoning_module_train_question_embeddings = questions_embeddings_qwen3embmodel[1][reasoning_module_indices].to(device)
        genmedgpt_reasoning_module_train_question_masks = questions_embeddings_qwen3embmodel[2][reasoning_module_indices].to(device)

        #testset
        genmedgpt_test_question_embeddings = questions_embeddings_qwen3embmodel[1][len_genmedgpt_train:].to(device)
        genmedgpt_test_question_masks = questions_embeddings_qwen3embmodel[2][len_genmedgpt_train:].to(device)


        model_path = "xxx.pth"
        llm_path = "xxx"
        llm_hidden_size = 4096
        top_k = 50
        tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        model = GraphEncoder(
            llm_path = llm_path,
            key_concept_feature_matrix = entity_vectors,
            prompt_embedding_size = llm_hidden_size,
            attn_proj_hidden_size = llm_hidden_size,
            device = 'cuda',
            top_k = top_k,
            question_embeddings = genmedgpt_expected_graph_train_question_embeddings,
            question_masks = genmedgpt_expected_graph_train_question_masks,
            answer_embeddings = genmedgpt_expected_graph_train_answer_embeddings)
        state_dict = torch.load(f"{model_path}", map_location="cuda")
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        model = model.to(dtype=torch.bfloat16)  # 确保所有层统一精度
        model.eval()

        sparse_relation_matrices_forreasoning = predict_graph(genmedgpt_test_question_embeddings,
                                            genmedgpt_test_question_masks,
                                            model)
        import pickle as pkl
        with open("xxx.pkl","wb") as f:
            pkl.dump(sparse_relation_matrices_forreasoning, f)

        sparse_relation_matrices_fortest = predict_graph(genmedgpt_reasoning_module_train_question_embeddings,
                                                genmedgpt_reasoning_module_train_question_masks,
                                                model)
        import pickle as pkl
        with open("xxx.pkl","wb") as f:
            pkl.dump(sparse_relation_matrices_fortest, f)


    elif args.dataset == "RJUA":
        device = "cuda"
        rjua_entity_vectors = torch.load("xxx.pt")

        import pickle
        with open("xxx.pkl", "rb") as f:
            rjua_questions_embeddings_qwen3embmodel = pickle.load(f)
        question_embeddings = rjua_questions_embeddings_qwen3embmodel[1].to(device)
        question_masks = rjua_questions_embeddings_qwen3embmodel[2].to(device)
        answers_embeddings = torch.load("xxx.pth")

        rjua_train = []
        with open("xxx.json", "r", encoding="utf-8") as f:
            for line in f:
                rjua_train.append(json.loads(line))  # 逐行加载
        rjua_train_questions = [i["question"] for i in rjua_train]
        rjua_train_answers = [i["answer"] for i in rjua_train]

        import random
        random.seed(42)
        len_rjua_train_questions = len(rjua_train_questions)
        expected_graph_train_indices = random.sample(range(len_rjua_train_questions), len_rjua_train_questions//2)
        all_indices = set(range(len_rjua_train_questions))
        reasoning_module_indices = list(all_indices - set(expected_graph_train_indices))
        reasoning_module_indices.sort()

        rjua_expected_graph_train_questions = [rjua_train_questions[i] for i in expected_graph_train_indices]
        rjua_expected_graph_train_answers = [rjua_train_answers[i] for i in expected_graph_train_indices]
        rjua_expected_graph_train_question_embeddings = question_embeddings[expected_graph_train_indices]
        rjua_expected_graph_train_question_masks = question_masks[expected_graph_train_indices]
        rjua_expected_graph_train_answer_embeddings = answers_embeddings[expected_graph_train_indices]

        rjua_reasoning_train_questions = [rjua_train_questions[i] for i in reasoning_module_indices]
        rjua_reasoning_train_answers = [rjua_train_answers[i] for i in reasoning_module_indices]
        rjua_reasoning_train_question_embeddings = question_embeddings[reasoning_module_indices]
        rjua_reasoning_train_question_masks = question_masks[reasoning_module_indices]
        rjua_reasoning_train_answer_embeddings = answers_embeddings[reasoning_module_indices]

        rjua_test = []
        with open("xxx.json", "r", encoding="utf-8") as f:
            for line in f:
                rjua_test.append(json.loads(line))  # 逐行加载
        rjua_test_questions = [i["question"] for i in rjua_test]
        rjua_test_answers = [i["answer"] for i in rjua_test]
        rjua_test_question_embeddings = question_embeddings[len_rjua_train_questions:]
        rjua_test_question_masks = question_masks[len_rjua_train_questions:]
        rjua_test_answer_embeddings = answers_embeddings[len_rjua_train_questions:]

        model_path = "xxx.pth"
        llm_path = "xxx"
        llm_name = "qwen3-8B"
        llm_hidden_size = 4096
        top_k = 50
        tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        model = GraphEncoder(
            llm_path = llm_path,
            key_concept_feature_matrix = rjua_entity_vectors,
            prompt_embedding_size = llm_hidden_size,
            attn_proj_hidden_size = llm_hidden_size,
            device = 'cuda',
            top_k = top_k,
            question_embeddings = rjua_expected_graph_train_question_embeddings,
            question_masks = rjua_expected_graph_train_question_masks,
            answer_embeddings = rjua_expected_graph_train_answer_embeddings)
        state_dict = torch.load(f"{model_path}", map_location="cuda")
        model.load_state_dict(state_dict)
        model = model.to("cuda")
        model = model.to(dtype=torch.bfloat16)
        model.eval()

        sparse_relation_matrices = predict_graph(rjua_test_question_embeddings,
                                                rjua_test_question_masks,
                                                model)
        import pickle as pkl
        with open("xxx.pkl","wb") as f:
            pkl.dump(sparse_relation_matrices, f)
            
        sparse_relation_matrices = predict_graph(rjua_reasoning_train_question_embeddings,
                                                rjua_reasoning_train_question_masks,
                                                model)
        with open("xxx.pkl","wb") as f:
            pkl.dump(sparse_relation_matrices, f)