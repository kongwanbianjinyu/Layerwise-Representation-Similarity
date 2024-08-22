import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar

import logging

from trainer import eval_model

def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--ckpt_path", type=str, help="path of the pre-trained checkpoint")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--max_len", type=int, default=256)

    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    
    return parser.parse_args()

import argparse
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
    from simctg import SimCTG
    model = SimCTG(args, args.ckpt_path, data.pad_token_id)
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded') 

    with torch.no_grad():
        model.eval()
        val_loss, each_layer_val_loss, each_layer_val_acc, pairwise_cosine_similarity = eval_model(args, model, data, cuda_available, device) 
        
        each_layer_val_loss = np.array(each_layer_val_loss)
        each_layer_val_ppl = np.exp(each_layer_val_loss)
        each_layer_val_loss = [round(x, 3) for x in each_layer_val_loss]
        each_layer_val_ppl = [round(x, 3) for x in each_layer_val_ppl]
        print ('validation loss is {}, each layer val loss is {}, each layer perplexity is {}, each layer accuracy is {}'.format(val_loss, each_layer_val_loss, each_layer_val_ppl, each_layer_val_acc))
        print ('pairwise cosine similarity is {}'.format(pairwise_cosine_similarity))
            
