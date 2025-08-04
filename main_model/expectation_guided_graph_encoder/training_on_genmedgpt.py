from model import *
from utils import *

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Entity vectors
    print("Loading Entity vectors...")
    entity_vectors = torch.load("xxx")

    # Load Entity explanations
    print("Loading keyword explanations...")
    with open("xxx.pkl", 'rb') as f:
        entity_explanation = pickle.load(f)
    entities = list(entity_explanation.keys())

    # Load full dataset
    with open("xxx", "rb") as f:
        genmedgpt_data = pickle.load(f)

    # Prepare training data
    genmedgpt_data_train_instructions = genmedgpt_data["train_instructions"]
    genmedgpt_data_train_questions = genmedgpt_data["train_questions"]
    genmedgpt_data_train_answers = genmedgpt_data["train_answers"]
    len_genmedgpt_train = len(genmedgpt_data_train_questions)

    # Split indices for training subsets
    random.seed(42)  # For reproducibility
    expected_graph_train_indices = random.sample(range(len_genmedgpt_train), len_genmedgpt_train // 2)

    # Prepare graph training subset
    genmedgpt_expected_graph_train_questions = [genmedgpt_data_train_questions[i] for i in expected_graph_train_indices]
    genmedgpt_expected_graph_train_answers = [genmedgpt_data_train_answers[i] for i in expected_graph_train_indices]

    # Load embeddings
    print("Loading embeddings...")
    with open("xxx.pkl", "rb") as f:
        questions_embeddings_qwen3embmodel = pickle.load(f)
    answers_embeddings_qwen3embmodel = torch.load("xxx.pth")

    # Prepare graph training embeddings
    genmedgpt_expected_graph_train_question_embeddings = questions_embeddings_qwen3embmodel[1][expected_graph_train_indices].to(device)
    genmedgpt_expected_graph_train_question_masks = questions_embeddings_qwen3embmodel[2][expected_graph_train_indices].to(device)
    genmedgpt_expected_graph_train_answer_embeddings = answers_embeddings_qwen3embmodel[expected_graph_train_indices].to(device)

    # Initialize model and tokenizer
    llm_path = "xxx"
    llm_hidden_size = 4096
    top_k = 10

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    
    # Initialize model
    model = GraphEncoder(
        llm_path=llm_path,
        key_concept_feature_matrix=entity_vectors,
        prompt_embedding_size=llm_hidden_size,
        attn_proj_hidden_size=llm_hidden_size,
        device=device,
        top_k=top_k,
        question_embeddings=genmedgpt_expected_graph_train_question_embeddings,
        question_masks=genmedgpt_expected_graph_train_question_masks,
        answer_embeddings=genmedgpt_expected_graph_train_answer_embeddings,
        attn_sim_loss_weight=0.1,
    )
    
    # Set model dtype
    model = model.to(dtype=torch.bfloat16)

    # Training configuration
    config = {
        'batch_size': 8,
        'epochs': 5,
        'base_lr': 5e-6,
        'gnn_projector_lr': 5e-6,
        'num_gpus': 2,
        'device': device
    }

    # Training instruction
    instruction = "If you are a doctor, please answer the medical questions based on the patient's description."

    # Start training
    print("Starting training...")
    start_time = time.time()
    
    trained_result = train_model(
        model=model,
        instruction=instruction,
        train_questions=genmedgpt_expected_graph_train_questions,
        train_answers=genmedgpt_expected_graph_train_answers,
        tokenizer=tokenizer,
        config=config
    )
    
    total_train_time = time.time() - start_time
    print(f"Training completed in {total_train_time:.2f} seconds")

    model_path = "xxx"  # Update with actual path

    if config['num_gpus'] > 1 and torch.cuda.device_count() > 1:
        torch.save(trained_result["model"].module.state_dict(), f"{model_path}.pth")
    else:
        torch.save(trained_result["model"].state_dict(), f"{model_path}.pth")
    print(f"Model weights saved to: {model_path}.pth")

    # Save configuration
    config_path = f"{model_path}_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"Configuration saved to: {config_path}")

    # Save additional training info
    additional_info = [
        trained_result["batch_losses"],
        trained_result["epoch_losses"],
        total_train_time
    ]
    with open(f"{model_path}_additional_info.pkl", "wb") as f:
        pkl.dump(additional_info, f)
    print(f"Additional training info saved to: {model_path}_additional_info.pkl")

if __name__ == "__main__":
    main()
    