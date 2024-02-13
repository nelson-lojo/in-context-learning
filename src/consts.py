import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn", "relu_attn_causal", "mlp"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
