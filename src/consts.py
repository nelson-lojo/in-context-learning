import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn", "relu_attn_causal", "mlp"]
TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    "kalman_filter",
    "noisy_kalman_filter"
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NULL_CHK(*values):
    for i, val in enumerate(values):
        if val is None:
            raise ValueError(f"Encountered `None` value in argument {i+1}!")
