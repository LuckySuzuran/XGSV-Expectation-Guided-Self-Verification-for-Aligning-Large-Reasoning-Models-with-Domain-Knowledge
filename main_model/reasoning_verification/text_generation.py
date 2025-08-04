from model import*
from utils import *
import pickle as pkl
def test_generation(instruction, cot_embedding, attn_mask, test_question, test_cot, sparse_expected_matrix, question_embedding,
                    model, tokenizer, llm_name, generation_config):
    # 确保模型在GPU
    model = model.to("cuda")
    
    # 1. 预处理数据（确保返回torch.long类型）
    text_input_ids = preproces_training_data(
        instruction, test_question, test_cot, None, tokenizer, llm_name, max_length=0
    )
    
    # 2. 处理设备转换（正确方式）
    if isinstance(text_input_ids, dict):
        # 如果是字典（如HF格式的input_ids和attention_mask）
        text_input_ids = {k: v.to(torch.long).to("cuda") for k, v in text_input_ids.items()}
    else:
        # 如果是普通张量
        text_input_ids = text_input_ids.to(torch.long).to("cuda")
    
    # 3. 处理其他输入
    cot_embedding_unsqueeze = cot_embedding.unsqueeze(0).to(model.llm.dtype).to("cuda")
    expected_matrix_unsqueeze = sparse_to_matrix(sparse_expected_matrix).unsqueeze(0).to(model.llm.dtype).to("cuda")
    attn_mask_unsqueeze = attn_mask.unsqueeze(0).to(model.llm.dtype).to("cuda")
    question_embedding_unsqueeze = question_embedding.unsqueeze(0).to(model.llm.dtype).to("cuda")
    # 4. 调试打印
    print("调试信息：")
    print(f"text_input_ids设备: {text_input_ids.device if not isinstance(text_input_ids, dict) else {k: v.device for k, v in text_input_ids.items()}}")
    print(f"模型设备: {next(model.parameters()).device}")
    
    # 5. 调用生成
    model.eval()
    answer = model.answer_generation(
        cot_embedding_unsqueeze,
        attn_mask_unsqueeze,
        text_input_ids,
        expected_matrix_unsqueeze,
        question_embedding_unsqueeze,
        generation_config
    )
    return answer

if __name__ == "__main__":
    import pickle as pkl
device = "cuda"
keywords_vectors = torch.load("../../dataset/GenMedGPT-5k/GenMedGPT_entity_vectors.pt")

with open("../../dataset/GenMedGPT-5k/entity_explanation.pkl", 'rb') as f:
    keywords_explanation = pkl.load(f)
keywords = list(keywords_explanation.keys())

with open("../../dataset/GenMedGPT-5k/GenMedGPT_data.pkl", "rb") as f:
    genmedgpt_data = pkl.load(f)

#trainset
genmedgpt_data_train_instructions = genmedgpt_data["train_instructions"]
genmedgpt_data_train_questions = genmedgpt_data["train_questions"]
genmedgpt_data_train_answers = genmedgpt_data["train_answers"]

import random
random.seed(42)
len_genmedgpt_train = len(genmedgpt_data_train_questions)
expected_graph_train_indices = random.sample(range(len_genmedgpt_train), len_genmedgpt_train//2)
all_indices = set(range(len_genmedgpt_train))  # 所有可能的索引（转为集合，方便求差集）
reasoning_module_indices = list(all_indices - set(expected_graph_train_indices))  # 差集：未选中的索引
reasoning_module_indices.sort()

genmedgpt_reasoning_module_train_questions = [genmedgpt_data_train_questions[i] for i in reasoning_module_indices]
genmedgpt_reasoning_module_train_answers = [genmedgpt_data_train_answers[i] for i in reasoning_module_indices]
genmedgpt_reasoning_module_train_instructions = [genmedgpt_data_train_instructions[i] for i in reasoning_module_indices]

with open("../../dataset/GenMedGPT-5k/GenMedGPT_qwen_results.pkl", "rb") as f:
    GenMedGPT_qwen_results = pkl.load(f)
genmedgpt_train_qwen_cots = GenMedGPT_qwen_results["genmedgpt_train_qwen_COTs"]
genmedgpt_test_qwen_cots = GenMedGPT_qwen_results["genmedgpt_test_qwen_COTs"]
genmedgpt_reasoning_module_train_qwen_cots = [genmedgpt_train_qwen_cots[i] for i in reasoning_module_indices]

gen_reasoning_module_qwen_cot_last_hidden_states = torch.load("data/gen_cot_embedding/gen_reasoning_module_qwen_cot_last_hidden_states.pth")
gen_reasoning_module_qwen_cot_attention_masks = torch.load("data/gen_cot_embedding/gen_reasoning_module_qwen_cot_attention_masks.pth")

# with open("../expected_graph_activation/genmed_reasoning_train_predicted_sparse_relation_matrices_0714.pkl", "rb") as f:
#     gen_reasoning_module_predicted_sparse_matrices = pkl.load(f)
with open("../expected_graph_activation/genmed_reasoning_train_predicted_sparse_relation_matrices_0722_0.pkl", "rb") as f:
    gen_reasoning_module_predicted_sparse_matrices = pkl.load(f)
import pickle
with open("../expected_graph_activation/genmedgpt_questions_embeddings_qwen3embmodel.pkl", "rb") as f:
    gen_questions_embeddings_qwen3embmodel = pickle.load(f)
question_embedding = gen_questions_embeddings_qwen3embmodel[0].to(device)
gen_reasoning_train_question_embedding = question_embedding[reasoning_module_indices]

llm_path = "/cpfs01/projects-HDD/cfff-3d5415058d87_HDD/public/public_model/qwen3-8B/Qwen3-8B"
llm_name = "qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
model = reasoning_model(
    reasoning_llm_path=llm_path,
    cot_embeddings = gen_reasoning_module_qwen_cot_last_hidden_states,
    cot_masks = gen_reasoning_module_qwen_cot_attention_masks,
    key_concept_feature_matrix=keywords_vectors,
    question_embeddings = gen_reasoning_train_question_embedding,
    prompt_embedding_size=4096,
    attn_proj_hidden_size=4096,
    device='cuda',
    top_k = 50,
    entity_similarity_loss_weight=0.1,
    text_recon_loss_weight=0,
    if_lora = True,
    relation_dropout = 0)
model = model.to(dtype=torch.bfloat16)

with open("../../dataset/GenMedGPT-5k/GenMedGPT_data.pkl", "rb") as f:
    genmedgpt_data = pkl.load(f)
#testset
genmedgpt_data_test_instructions = genmedgpt_data["test_instructions"]
genmedgpt_data_test_questions = genmedgpt_data["test_questions"]

with open("../../dataset/GenMedGPT-5k/GenMedGPT_qwen_results.pkl", "rb") as f:
    GenMedGPT_qwen_results = pkl.load(f)
genmedgpt_test_qwen_cots = GenMedGPT_qwen_results["genmedgpt_test_qwen_COTs"]
genmedgpt_test_qwen_cot_embeddings = torch.load("data/gen_cot_embedding/gen_test_qwen_cot_last_hidden_states.pth")
genmedgpt_test_qwen_cot_masks = torch.load("data/gen_cot_embedding/gen_test_qwen_cot_attention_masks.pth")

with open("../expected_graph_activation/genmed_test_predicted_sparse_relation_matrices_0728_0.pkl", "rb") as f:
    genmed_test_predicted_sparse_relation_matrices_0728_0 = pkl.load(f)
    
genmedgpt_test_qwen_question_embeddings = question_embedding[-len(genmedgpt_data_test_questions):]

model_path = "../../resulted_models/genmedgpt_qwen_reasoning_model_0729_0.pth"
state_dict = torch.load(f"{model_path}", map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(dtype=torch.bfloat16)  # 确保所有层统一精度
model.to("cuda:3")
model.eval()

# 创建生成配置
generation_config = GenerationConfig(
    max_length=8192,
    temperature=0.2,
    top_p=0.1,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id)

answers = []
for idx in range (750, 1091):
    print(idx)
    instruction = genmedgpt_data_test_instructions[0]
    cot_embedding = genmedgpt_test_qwen_cot_embeddings[idx]
    attn_mask = genmedgpt_test_qwen_cot_masks[idx]
    test_question = genmedgpt_data_test_questions[idx]
    test_cot = genmedgpt_test_qwen_cots[idx]
    test_question_emb = genmedgpt_test_qwen_question_embeddings[idx]
    sparse_expected_matrix = genmed_test_predicted_sparse_relation_matrices_0728_0[idx]
    answer = test_generation(instruction, cot_embedding,attn_mask, test_question, test_cot,
                    sparse_expected_matrix, test_question_emb, model, tokenizer, llm_name, generation_config)
    answers.append(answer)


with open ("gen_qwen_ours_750_1091_answers_0729.pkl", "wb") as f:
    pkl.dump(answers + answers_,f)