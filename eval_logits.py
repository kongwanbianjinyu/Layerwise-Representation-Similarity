import torch
import argparse
import progressbar
import numpy as np
import os
import json

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

def extract_all_features(args, model, data, device):
    print('Start extracting all features...')
    model.eval()
    dataset_batch_size = args.number_of_gpu * args.batch_size_per_gpu
    eval_step = int(data.test_num / dataset_batch_size) + 1
    all_hidden_states = []

    with torch.no_grad():
        for idx in range(eval_step):
            batch_input_tensor, batch_labels, _ = data.get_next_validation_batch(batch_size=dataset_batch_size, mode='test')
            batch_input_tensor = batch_input_tensor.to(device)
            batch_labels = batch_labels.to(device)
            
            # Extract hidden states from the model
            hidden_states = model.get_all_token_hidden_states(batch_input_tensor, batch_labels) # [batch_size * num_valid_tokens, num_layers, feature_dim]
            
            # Move hidden states to CPU to avoid CUDA out of memory
            hidden_states = hidden_states.cpu()
            
            all_hidden_states.append(hidden_states)

            print(f"Batch[{idx}] Extracted hidden states with shape: {hidden_states.shape}")

    # Stack all hidden states to get the final shape [num_layers, all_num_tokens, feature_dim]
    # Combine all hidden states
    all_hidden_states = torch.cat(all_hidden_states, dim=0)  # Shape: [sum(batch_sizes), num_layers, feature_dim]
    all_hidden_states = all_hidden_states.transpose(0, 1)  # Rearranging to [num_layers, sum(batch_sizes), feature_dim]
    print(f'All hidden states shape: {all_hidden_states.shape}')
    return all_hidden_states
    
def save_metrix_to_json(args, metric_name="-", metric=[]):
    working_dir = f"{args.model}"
    file_path = os.path.join(working_dir, f"{metric_name}")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = os.path.join(file_path, f"{metric_name}.json")
    with open(file_name, 'w') as f:
        json.dump(metric, f)

def compute_saturate_events(args, model, all_features):
    linear = model.model.lm_head.cpu()
    L, N, K = all_features.shape
    
    batch_size = 1000  # Process 1000 tokens at a time
    num_batches = N // batch_size + (1 if N % batch_size != 0 else 0)
    
    all_saturate_events = []
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, N)
        
        batch_features = all_features[:, start_idx:end_idx, :]
        batch_features = batch_features.cuda()  # Move batch_features to GPU
        linear = linear.cuda()  # Move linear layer to GPU
        batch_logits = linear(batch_features)  # [num_layers, batch_size, K]
        batch_logits = batch_logits.cpu()  # Move results back to CPU
        batch_logits = batch_logits.permute(1, 0, 2)
        
        # Take argmax of batch_logits along the last dimension to get layer_idx
        layer_idx = torch.argmax(batch_logits, dim=-1)  # Shape: [batch_size, L]
        
        # Compute saturate events for each token in the batch
        for i in range(layer_idx.shape[0]):
            token_predictions = layer_idx[i]
            saturate_event_layer = saturate_event(token_predictions)
            all_saturate_events.append(saturate_event_layer)
        
        print(f"Processed batch {batch + 1}/{num_batches}")
    
    # Convert to numpy array
    saturate_events = np.array(all_saturate_events)
    
    # Compute percentage of saturate events
    saturate_percentage = np.sum(saturate_events < L - 1) / saturate_events.shape[0]
    print(f"Percentage of saturate events: {saturate_percentage:.2%}")

    # Count the number of saturate events for each layer
    saturate_counts = np.bincount(saturate_events, minlength=L)

    print(f"Saturate event counts for each layer: {saturate_counts}")

    # save metric to json
    save_metrix_to_json(args, metric_name="saturate_events", metric=saturate_counts.tolist())


def remove_mean_normalize(features):
    features = features - torch.mean(features, dim = 0, keepdim= True) # remove mean
    normalized_features = torch.nn.functional.normalize(features, p=2.0, dim=1) # normalize
    return normalized_features


def compute_cos_sim_with_last_layer(args, all_features):
    cos_sim_with_last_layer = []
    avg_cos_sim_with_last_layer = []
    HL = all_features[-1]
    
    num_layers = len(all_features)
    for i in range(num_layers):
        Hl = all_features[i]
        cos_last = (Hl * HL).sum(dim = 1)
        cos_sim_with_last_layer.append(cos_last)
        avg_cos_sim_with_last_layer.append(torch.mean(cos_last).item())
    save_metrix_to_json(args, metric_name="cos_sim_with_last_layer", metric=avg_cos_sim_with_last_layer)
    return cos_sim_with_last_layer, avg_cos_sim_with_last_layer

def compute_cosine_sim_with_last_based_on_saturate_layers(args, model, all_features):
    linear = model.model.lm_head.cpu()
    L, N, K = all_features.shape
    
    batch_size = 1000  # Process 1000 tokens at a time
    num_batches = N // batch_size + (1 if N % batch_size != 0 else 0)
    
    all_layer_idx = []
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, N)
        
        batch_features = all_features[:, start_idx:end_idx, :]
        batch_features = batch_features.cuda()  # Move batch_features to GPU
        linear = linear.cuda()  # Move linear layer to GPU
        batch_logits = linear(batch_features)  # [num_layers, batch_size, K]
        batch_logits = batch_logits.cpu()  # Move results back to CPU
        batch_logits = batch_logits.permute(1, 0, 2)
        
        # Take argmax of batch_logits along the last dimension to get layer_idx
        layer_idx = torch.argmax(batch_logits, dim=-1)  # Shape: [batch_size, L]
        all_layer_idx.append(layer_idx)
        
        print(f"Processed batch {batch + 1}/{num_batches}")
    
    layer_idx = torch.cat(all_layer_idx, dim=0)  # Shape: [N, L]
    
    saturate_events = np.array([saturate_event(layer_idx[i]) for i in range(layer_idx.shape[0])])

    cos_with_last_based_on_saturate_layers = []

    for i in range(L):
        indices = np.where(saturate_events == i)[0]
        
        if len(indices) == 0:
            print(f"Layer {i} has no saturate events")
            cos_with_last = [0] * L
        else:
            layer_features = all_features[:, indices, :]
            layer_features = [remove_mean_normalize(layer_features[j]) for j in range(L)]
            _, cos_with_last = compute_cos_sim_with_last_layer(args, layer_features)
        
        print(f"Layer {i}: {cos_with_last}")
        print(f"Layer {i} features shape: {layer_features[0].shape}")
        cos_with_last_based_on_saturate_layers.append(cos_with_last)
        
    save_metrix_to_json(args, metric_name="cos_sim_with_last_based_on_saturate_events", metric=cos_with_last_based_on_saturate_layers)
    return cos_with_last_based_on_saturate_layers



def compute_cosine_sim_with_classifier_based_on_saturate_layers(args, model, all_features):
    linear = model.model.lm_head.cpu()
    L, N, K = all_features.shape
    
    batch_size = 1000  # Process 1000 tokens at a time
    num_batches = N // batch_size + (1 if N % batch_size != 0 else 0)
    
    all_preds = []
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, N)
        
        batch_features = all_features[:, start_idx:end_idx, :]
        batch_features = batch_features.cuda()
        linear = linear.cuda()
        batch_logits = linear(batch_features)  # [num_layers, batch_size, K]
        batch_logits = batch_logits.cpu()
        batch_logits = batch_logits.permute(1, 0, 2)  # [batch_size, num_layers, K]
        
        # Take argmax of batch_logits along the last dimension to get layer_idx
        batch_preds = torch.argmax(batch_logits, dim=-1)  # Shape: [batch_size, num_layers]
        all_preds.append(batch_preds)
        
        print(f"Processed batch {batch + 1}/{num_batches}")
    
    all_preds = torch.cat(all_preds, dim=0)  # Shape: [N, num_layers]
    
    # Compute saturate events for each token
    saturate_events = np.array([saturate_event(all_preds[i]) for i in range(all_preds.shape[0])])

    # Get classifier weights
    classifier_weights = linear.weight.detach().cpu()  # [K, d]

    # Initialize a list to store cosine similarities for each layer
    cos_with_classifier_based_on_saturate_layers = []

    # Iterate through each layer
    num_layers = all_features.shape[0]  # Get the number of layers from the shape of all_features
    for i in range(num_layers):
        # Get indices where saturate_events == i
        indices = np.where(saturate_events == i)[0]
        
        if len(indices) == 0:
            print(f"Layer {i} has no saturate events")
            cos_with_classifier = [0] * num_layers
        else:
            # Extract features for these indices from all layers
            layer_features = all_features[:, indices, :] # [L, N, d]

            pred = all_preds[indices, i] # [N]

            w = classifier_weights[pred] # [N, d]
            cos_with_classifier = []
            for j in range(num_layers):
                cos_sim = torch.nn.functional.cosine_similarity(layer_features[j], w, dim=1) # [N]
                
                cos_with_classifier.append(cos_sim.mean().item())
        
        cos_with_classifier_based_on_saturate_layers.append(cos_with_classifier)
        print(f"Layer {i}: {cos_with_classifier}")
    # Save the metric to json
    save_metrix_to_json(args, metric_name="cos_sim_with_classifier_based_on_saturate_layers", metric=cos_with_classifier_based_on_saturate_layers)
    return cos_with_classifier_based_on_saturate_layers




def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--mode", type=str, help="mode of the model, multi_exit or single_exit")
    parser.add_argument("--ckpt_path", type=str, help="path of the pre-trained checkpoint")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    parser.add_argument("--count_saturate_events", action='store_true', help="Flag to count saturate events")
    parser.add_argument("--compute_cosine_sim_with_last_based_on_saturate_layers", action='store_true', help="Flag to compute cosine similarity with last hidden state based on saturate layers")
    parser.add_argument("--compute_cosine_sim_with_classifier_based_on_saturate_layers", action='store_true', help="Flag to compute cosine similarity with classifier based on saturate layers")
    
    return parser.parse_args()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU.')
    else:
        pass
    args = parse_config()
    device = torch.device('cuda')

    print ('Loading data...')
    from dataclass import Data
    data = Data(args.ckpt_path, args.train_path, args.dev_path, args.test_path, args.max_len)
    print ('Data loaded.')

    print ('Loading pre-trained model...')
    if args.mode == "multi_exit":
        from multiexit_simctg import MultiExitSimCTG
        print ('Initializaing MultiExitSimCTG model...')
        model = MultiExitSimCTG(args, args.ckpt_path, data.pad_token_id)
        model.load_lm_heads(args.ckpt_path)
    else:
        from simctg import SimCTG
        print ('Initializaing SimCTG model...')
        model = SimCTG(args, args.ckpt_path, data.pad_token_id)
    
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded') 

    with torch.no_grad():
        all_hidden_states = extract_all_features(args, model, data, device)
        
        if args.count_saturate_events:
            compute_saturate_events(args, model, all_hidden_states)
        
        if args.compute_cosine_sim_with_last_based_on_saturate_layers:
            compute_cosine_sim_with_last_based_on_saturate_layers(args, model, all_hidden_states)
        
        if args.compute_cosine_sim_with_classifier_based_on_saturate_layers:
            compute_cosine_sim_with_classifier_based_on_saturate_layers(args, model, all_hidden_states)

