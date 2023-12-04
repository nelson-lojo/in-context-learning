import torch


SEQ_MODELS = ["gpt2", "lstm", "relu_attn", "quantized"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
