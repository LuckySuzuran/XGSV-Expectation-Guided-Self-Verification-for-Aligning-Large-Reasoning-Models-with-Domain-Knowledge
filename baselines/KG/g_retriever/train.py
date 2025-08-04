from g_retriever import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Was asked to gather along dimension 0")
# take lora as an exmaple
# if prompt tuning is needed, juse set the if_llm_frozen as True

entity_file = "../../dataset/genmedgpt/EMCKG/entity2id.txt"
relation_file = "../../dataset/genmedgpt/EMCKG/relation2id.txt"
edge_file = "../../dataset/genmedgpt/EMCKG/relation.txt"
node_embedding_file = "../../dataset/genmedgpt/EMCKG/GenMedGPT_entity_vectors.pt" #need construction additionally
edge_embedding_file = "../../dataset/genmedgpt/EMCKG/GenMedGPT_relation_vectors.pt" #need construction additionally
graph, textual_nodes, textual_edges, entity_to_id, relation_to_id = load_graph_data(
                                                                        entity_file,
                                                                        relation_file,
                                                                        edge_file,
                                                                        node_embedding_file,
                                                                        edge_embedding_file)

import pickle as pkl
device = "cuda"
with open("../../dataset/genmedgpt/GenMedGPT_data.pkl", "rb") as f:
    genmedgpt_data = pkl.load(f)
genmedgpt_data_train_instructions = genmedgpt_data["train_instructions"]
genmedgpt_data_train_questions = genmedgpt_data["train_questions"]
genmedgpt_data_train_answers = genmedgpt_data["train_answers"]

with open("xxx.pkl","rb") as f:
    question_embeddings_list = pkl.load(f)
train_question_embeddings = question_embeddings_list[0][:len(genmedgpt_data_train_questions)]

with open("xxx.pkl", "rb") as f:
    GenMedGPT_qwen_results = pkl.load(f)
genmedgpt_train_qwen_cots = GenMedGPT_qwen_results["genmedgpt_train_qwen_COTs"]


llm_path = "xxx"
reasoning_llm_path = "xxx"
model = GraphLLM(llm_model_path = llm_path, if_llm_frozen=False,  graph = graph, textual_nodes = textual_nodes, textual_edges = textual_edges, llm_name = "qwen")
model = model.to("cuda")
model.to(torch.bfloat16)

config = {
    'llm_name': "qwen3-8B",
    'max_length':4000,
    'batch_size':2,
    'learning_rate': 5e-5,
    'device':"cuda",
    'epochs': 3,
    'num_gpus': 2}

instruction = genmedgpt_data_train_instructions[0]
train_questions = genmedgpt_data_train_questions
train_answers = genmedgpt_data_train_answers
model_cots = genmedgpt_train_qwen_cots
question_embeddings = train_question_embeddings
result = train_model(model, graph, textual_nodes, textual_edges, question_embeddings,
                instruction, train_questions, model_cots, train_answers,
                reasoning_llm_path, config)

save_complete_model(result["model"], "gen_qwen_lora/", config, if_llm_frozen = False)