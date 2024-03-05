import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn", "relu_attn_causal", "micro"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
