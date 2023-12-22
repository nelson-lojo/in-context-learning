import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
