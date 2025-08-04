from model import*
from utils import*
import pickle as pkl
import json
import random

def main():
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Entity vectors
    rjua_entity_vectors = torch.load("xxx")  # Update with actual path

    # Load Entity explanations
    with open("xxx", 'rb') as f:
        rjua_keywords_explanation = pkl.load(f)
    rjua_entities = list(rjua_keywords_explanation.keys())

    # Load training data
    rjua_train = []
    with open("xxx", "r", encoding="utf-8") as f:
        for line in f:
            rjua_train.append(json.loads(line))  # Load line by line
    
    rjua_train_questions = [item["question"] for item in rjua_train]
    rjua_train_answers = [item["answer"] for item in rjua_train]

    # Load embeddings
    with open("xxx.pkl", "rb") as f:
        rjua_questions_embeddings_data = pkl.load(f)
    
    question_embeddings = rjua_questions_embeddings_data[1].to(device)
    question_masks = rjua_questions_embeddings_data[2].to(device)
    answers_embeddings = torch.load("xxx.pth")

    # Split indices for training
    len_train = len(rjua_train_questions)
    random.seed(42) # For reproducibility
    expected_graph_train_indices = random.sample(range(len_train), len_train // 2, )

    # Prepare training data subsets
    rjua_expected_graph_train_questions = [rjua_train_questions[i] for i in expected_graph_train_indices]
    rjua_expected_graph_train_answers = [rjua_train_answers[i] for i in expected_graph_train_indices]
    rjua_expected_graph_train_question_embeddings = question_embeddings[expected_graph_train_indices]
    rjua_expected_graph_train_question_masks = question_masks[expected_graph_train_indices]
    rjua_expected_graph_train_answer_embeddings = answers_embeddings[expected_graph_train_indices]

    # Initialize model and tokenizer
    llm_path = "xxx"  # Update with actual path
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    
    model = GraphEncoder(
        llm_path=llm_path,
        key_concept_feature_matrix=rjua_entity_vectors,
        prompt_embedding_size=4096,
        attn_proj_hidden_size=4096,
        device=device,
        top_k=50,
        question_embeddings=rjua_expected_graph_train_question_embeddings,
        question_masks=rjua_expected_graph_train_question_masks,
        answer_embeddings=rjua_expected_graph_train_answer_embeddings,
        attn_sim_loss_weight=0.1
        )
    
    # Set model dtype
    model = model.to(dtype=torch.bfloat16)

    # Training configuration
    config = {
        'batch_size': 4,
        'epochs': 5,
        'base_lr': 5e-6,
        'gnn_projector_lr': 5e-6,
        'num_gpus': 2,
        'device': device
    }

    # Training instruction
    instruction = "你是一个医学专家。你需要根据你的医学知识，回答用户的医学问题。"

    # Start training
    print("Starting training...")
    start_time = time.time()
    
    trained_result = train_model(
        model=model,
        instruction=instruction,
        train_questions=rjua_expected_graph_train_questions,
        train_answers=rjua_expected_graph_train_answers,
        tokenizer=tokenizer,
        config=config
    )
    
    total_train_time = time.time() - start_time
    print(f"Training completed in {total_train_time:.2f} seconds")

    # Save model and results
    model_path = "xxx"
    
    # Save model weights
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
    