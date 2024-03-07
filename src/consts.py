import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn", "relu_attn_causal", "mlp"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def NULL_CHK(*values):
    for i, val in enumerate(values):
        if val is None:
            raise ValueError(f"Encountered `None` value in argument {i+1}!")
