import torch
import argparse
import progressbar
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

def eval_model_logits(args, model, data, device):
    dataset_batch_size = args.batch_size_per_gpu
    eval_step = int(data.test_num / dataset_batch_size) + 1

    model.eval()
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        avg_saturate_percentage = 0
        total_events = 0
        all_saturate_layers = []
        for idx in range(eval_step):
            p.update(idx)
            batch_input_tensor, batch_labels, _ = data.get_next_validation_batch(batch_size=dataset_batch_size, mode='test')
            batch_input_tensor = batch_input_tensor.to(device)
            all_layer_logits = model.compute_all_layer_logits(batch_input_tensor)
            all_token_index = []
            for i in range(len(all_layer_logits)):
                all_layer_logits[i] = all_layer_logits[i].cpu().numpy()
                # combine the first two dimensions
                all_layer_logits[i] = all_layer_logits[i].reshape(-1, all_layer_logits[i].shape[-1])
                # take argmax of last dimension
                all_token_index.append(all_layer_logits[i].argmax(axis=-1))
            # concatenate the token index of all layers
            all_token_index = np.stack(all_token_index)
            # find the nearest layer that index different from last layer index
            saturate_layers = []
            for col in range(all_token_index.shape[1]):
                layer_token_index = all_token_index[:, col]
                l = saturate_event(layer_token_index)
                saturate_layers.append(l)
            all_saturate_layers.extend(saturate_layers)
            
            # count percentage that saturate layers is not 11
            saturate_layers = np.array(saturate_layers)
            saturate_percentage = np.sum(saturate_layers != 11) / saturate_layers.shape[0]
            total_events += saturate_layers.shape[0]
            print("Batch[{}] Saturate Event {}/{}, Percentage: {}".format(idx,np.sum(saturate_layers != 11), saturate_layers.shape[0], saturate_percentage))
            avg_saturate_percentage += saturate_percentage

        avg_saturate_percentage /= eval_step
        
        print ('Average Saturate Event {}/{}, Percentage:{}'.format( int(total_events * avg_saturate_percentage), total_events, avg_saturate_percentage))

        # count each layer saturate event count
        saturate_event_count = np.zeros(12)
        for l in all_saturate_layers:
            saturate_event_count[l] += 1

        print ('Saturate event count for each layer{}'.format(saturate_event_count))

        # save the result all_saturate_layers (num_tokens,) each element is the layer index that saturate event happens for this token.
        np.save(args.save_npy_path, all_saturate_layers)
        print(f'Saturate event layer index saved to {args.save_npy_path}')
        p.finish()


    return all_token_index


def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--mode", type=str, help="mode of the model, multi_exit or single_exit")
    parser.add_argument("--ckpt_path", type=str, help="path of the pre-trained checkpoint")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    parser.add_argument("--save_npy_path", type=str, help='path to save the result npy file.')
    
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
        all_token_index = eval_model_logits(args, model, data, device)
