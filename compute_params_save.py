single_classifier = {
    "deit-s": 22.05e6,   # 22.05M parameters
    "deit-b": 86.57e6,   # 86.57M parameters
    "gpt2": 117e6,       # 117M parameters
    "gpt3": 175e9,       # 175B parameters
    "llama-2": 70e9      # 70B parameters (assuming LLaMA-2 base variant)
}

num_classes = {
    "deit-s": 1000,      # Commonly used for image classification tasks
    "deit-b": 1000,      # Commonly used for image classification tasks
    "gpt2": 50257,       # Vocabulary size for GPT-2
    "gpt3": 50257,       # Vocabulary size for GPT-3
    "llama-2": 32000     # Vocabulary size for LLaMA-2
}

num_features = {
    "deit-s": 384,       # Hidden dimension
    "deit-b": 768,       # Hidden dimension
    "gpt2": 768,         # Hidden dimension
    "gpt3": 12288,       # Hidden dimension
    "llama-2": 4096      # Hidden dimension
}

num_layers = {
    "deit-s": 12,        # Number of transformer layers
    "deit-b": 12,        # Number of transformer layers
    "gpt2": 12,          # Number of transformer layers
    "gpt3": 96,          # Number of transformer layers
    "llama-2": 40        # Number of transformer layers (LLaMA-2 70B variant)
}



multi_classifier = {}
param_saved = {}
for key in single_classifier.keys():
    single = single_classifier[key]
    classes = num_classes[key]
    feature = num_features[key]
    layer = num_layers[key]
    each_head = (classes * feature)

    multi = (single + (layer - 1) * each_head)
    multi_classifier[key] = multi / 1e6
    param_saved[key] = (multi - single)/multi

print(multi_classifier)
print(param_saved)
