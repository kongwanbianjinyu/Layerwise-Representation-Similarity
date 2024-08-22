import numpy as np

def saturate_event(layer_token_index):
    # layer_token_index:(num_layers)
    L = len(layer_token_index)
    final_pred = layer_token_index[-1]
    i = 0
    while i < len(layer_token_index):
        if layer_token_index[L-1-i] != final_pred:
            break
        i += 1
    l = L - i  # l is between 0 and L-1.
    return l

layer_token_index = np.array([1,2,3,4])

print(saturate_event(layer_token_index))



