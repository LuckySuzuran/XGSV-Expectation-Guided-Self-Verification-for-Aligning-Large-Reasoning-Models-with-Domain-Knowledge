from model import*
from utils import *

if __name__ == "__main__":
    device = "cuda"
    rjua_entity_vectors = torch.load("xxx.pt")

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

    rjua_qwen_cots_embeddings = torch.load("xxx.pt")
    rjua_qwen_cots_last_hidden_states = rjua_qwen_cots_embeddings['last_hidden_states']
    rjua_qwen_cots_attention_masks = rjua_qwen_cots_embeddings['attention_masks']

    rjua_reasoning_train_questions = [rjua_train_questions[i] for i in reasoning_module_indices]
    rjua_reasoning_train_answers = [rjua_train_answers[i] for i in reasoning_module_indices]
    with open("xxx.pkl", "rb") as f:
        rjua_qwen_results = pkl.load(f)
    rjua_all_qwen_COTs = rjua_qwen_results["rjua_all_qwen_COTs"]
    rjua_reasoning_train_qwen_cots = [rjua_all_qwen_COTs[i] for i in reasoning_module_indices]
    rjua_reasoning_train_cot_embeddings = rjua_qwen_cots_last_hidden_states[reasoning_module_indices]
    rjua_reasoning_train_cot_masks = rjua_qwen_cots_attention_masks[reasoning_module_indices]

    with open("xxx.pkl", "rb") as f:
        rjua_reasoning_module_predicted_sparse_matrices = pkl.load(f)

    import pickle
    with open("xxx.pkl", "rb") as f:
        rjua_questions_embeddings_qwen3embmodel = pickle.load(f)
    question_embedding = rjua_questions_embeddings_qwen3embmodel[0].to(device)
    rjua_reasoning_train_question_embedding = question_embedding[reasoning_module_indices]

    llm_path = "xxx"
    llm_name = "xxx"

    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    model = Model(
        reasoning_llm_path=llm_path,
        cot_embeddings = rjua_reasoning_train_cot_embeddings,
        cot_masks = rjua_reasoning_train_cot_masks,
        key_concept_feature_matrix=rjua_entity_vectors,
        question_embeddings = rjua_reasoning_train_question_embedding,
        prompt_embedding_size=4096,
        attn_proj_hidden_size=4096,
        device='cuda',
        top_k = 50,
        entity_similarity_loss_weight=0.1,
        if_lora = True
        )
    model = model.to(dtype=torch.bfloat16)

    # 训练模型
    config = {
        'batch_size':4,
        'epochs': 3,
        # 'learning_rate': 5e-4,
        'lora_lr':5e-4,
        'custom_lr':5e-6,
        'num_gpus': 4,
        'device':"cuda",
    }
    instruction = "你是一个医学专家。你需要根据你的医学知识，回答用户的医学问题。"
    trained_result = train_model(
        model=model,
        instruction = instruction,
        train_questions=rjua_reasoning_train_questions,
        model_cots = rjua_reasoning_train_qwen_cots,
        train_answers=rjua_reasoning_train_answers,
        sparse_expected_relations = rjua_reasoning_module_predicted_sparse_matrices,
        reasoning_llm_path = llm_path,
        config=config)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            print(f"{name:30} grad: {grad_mean:.3e}") 

    model_path = "xxx"
    save_model(model, model_path)
    config_path = f"{model_path}config.json"
    # 保存配置到JSON文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"save to{config_path}")

    with open(f"{model_path}additional_info.pkl", "wb") as f:
        pkl.dump([trained_result["batch_losses"], trained_result["epoch_losses"], trained_result["total_train_time"]], f)