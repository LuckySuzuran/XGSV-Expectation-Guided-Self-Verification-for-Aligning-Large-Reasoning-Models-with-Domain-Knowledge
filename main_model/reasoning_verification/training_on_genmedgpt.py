from model import*
from utils import *
import pickle as pkl

if __name__ == "__main__":
    device = "cuda"
    entity_vectors = torch.load("xxx.pt")

    with open("../../dataset/genmedgpt/GenMedGPT_data.pkl", "rb") as f:
        genmedgpt_data = pkl.load(f)

    #trainset
    genmedgpt_data_train_instructions = genmedgpt_data["train_instructions"]
    genmedgpt_data_train_questions = genmedgpt_data["train_questions"]
    genmedgpt_data_train_answers = genmedgpt_data["train_answers"]

    import random
    random.seed(42)
    len_genmedgpt_train = len(genmedgpt_data_train_questions)
    expected_graph_train_indices = random.sample(range(len_genmedgpt_train), len_genmedgpt_train//2)
    all_indices = set(range(len_genmedgpt_train)) 
    reasoning_module_indices = list(all_indices - set(expected_graph_train_indices))
    reasoning_module_indices.sort()

    genmedgpt_reasoning_module_train_questions = [genmedgpt_data_train_questions[i] for i in reasoning_module_indices]
    genmedgpt_reasoning_module_train_answers = [genmedgpt_data_train_answers[i] for i in reasoning_module_indices]
    genmedgpt_reasoning_module_train_instructions = [genmedgpt_data_train_instructions[i] for i in reasoning_module_indices]

    with open("xxx.pkl", "rb") as f:
        GenMedGPT_R17B_results = pkl.load(f)
    genmedgpt_train_R17B_cots = GenMedGPT_R17B_results["genmedgpt_train_R17B_COTs"]
    genmedgpt_test_R17B_cots = GenMedGPT_R17B_results["genmedgpt_test_R17B_COTs"]
    genmedgpt_reasoning_module_train_R17B_cots = [genmedgpt_train_R17B_cots[i] for i in reasoning_module_indices]
    gen_reasoning_module_R17B_cot_last_hidden_states = torch.load("xxx.pth")
    gen_reasoning_module_R17B_cot_attention_masks = torch.load("xxx.pth")

    with open("xxx.pkl", "rb") as f:
        gen_reasoning_module_predicted_sparse_matrices = pkl.load(f)

    import pickle
    with open("xxx.pkl", "rb") as f:
        gen_questions_embeddings_qwen3embmodel = pickle.load(f)
    question_embedding = gen_questions_embeddings_qwen3embmodel[0].to(device)
    gen_reasoning_train_question_embedding = question_embedding[reasoning_module_indices]

    llm_path = "xxx"
    llm_name = "xxx"
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    model = Model(
        reasoning_llm_path=llm_path,
        cot_embeddings = gen_reasoning_module_R17B_cot_last_hidden_states,
        cot_masks = gen_reasoning_module_R17B_cot_attention_masks,
        key_concept_feature_matrix=entity_vectors,
        question_embeddings = gen_reasoning_train_question_embedding,
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
        'batch_size':2,
        'epochs': 3,
        'lora_lr':5e-4,
        'custom_lr':5e-6,
        'num_gpus': 2,
        'device':"cuda",
        'sample_num': len(genmedgpt_reasoning_module_train_questions)
    }
    instruction = genmedgpt_reasoning_module_train_instructions[0]
    trained_result = train_model(
        model=model,
        instruction = instruction,
        train_questions=genmedgpt_reasoning_module_train_questions,
        model_cots = genmedgpt_reasoning_module_train_R17B_cots,
        train_answers=genmedgpt_reasoning_module_train_answers,
        sparse_expected_relations = gen_reasoning_module_predicted_sparse_matrices,
        reasoning_llm_path = llm_path,
        config=config)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            print(f"{name:30} grad: {grad_mean:.3e}") 

    model_path = "xxx"
    torch.save(trained_result["model"].module.state_dict(), f"{model_path}.pth")
    print(f"model weights have been saved to: {model_path}.pth")

    config_path = f"{model_path}_config.json"
    # 保存配置到JSON文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print(f"save config to{config_path}")

    with open(f"{model_path}_additional_info.pkl", "wb") as f:
        pkl.dump([trained_result["batch_losses"], trained_result["epoch_losses"], trained_result["total_train_time"]], f)