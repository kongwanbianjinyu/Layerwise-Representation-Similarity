# coding=utf-8
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
from tqdm import tqdm
import wandb
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def eval_model(args, model, data, cuda_available, device):
    dataset_batch_size = args.batch_size_per_gpu * args.number_of_gpu
    eval_step = int(data.test_num / dataset_batch_size) + 1
    val_loss = 0.0
    each_layer_val_loss = [0] * 12
    each_layer_val_acc = [0] * 12
    pairwise_cosine_similarity = torch.eye(12)
    
    model.eval()
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        
        for idx in range(eval_step):
            p.update(idx)
            batch_input_tensor, batch_labels, _ = \
            data.get_next_validation_batch(batch_size=dataset_batch_size, mode='test')
            if cuda_available:
                batch_input_tensor = batch_input_tensor.cuda(device)
                batch_labels = batch_labels.cuda(device)
            one_batch_all_layer_loss, one_batch_each_layer_loss,  one_batch_each_layer_acc, one_batch_pairwise_cosine_similarity = model.eval_loss(batch_input_tensor, batch_labels)
            
            val_loss += one_batch_all_layer_loss.item()
            
            for i, one_layer_loss in enumerate(one_batch_each_layer_loss):
                each_layer_val_loss[i] += torch.sum(one_layer_loss).item()
            for i, one_layer_acc in enumerate(one_batch_each_layer_acc):
                each_layer_val_acc[i] += torch.sum(one_layer_acc).item()

            pairwise_cosine_similarity += one_batch_pairwise_cosine_similarity

        p.finish()
    model.train()
    val_loss = val_loss / eval_step
    each_layer_val_loss = [x / eval_step for x in each_layer_val_loss]
    each_layer_val_acc = [x / eval_step for x in each_layer_val_acc]
    pairwise_cosine_similarity = pairwise_cosine_similarity / eval_step

    plot_matrix(pairwise_cosine_similarity.cpu().numpy(), filename = 'aligned_pairwise_cosine_similarity.png')
   
    return val_loss, each_layer_val_loss, each_layer_val_acc, pairwise_cosine_similarity



def plot_matrix(mat,filename = ""):
    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = sb.diverging_palette(0, 230, 90, 60, as_cmap=True)
    # plot heatmap
    mat_df = pd.DataFrame(mat)
    sb.heatmap(mat_df, annot_kws = {"fontsize": 10},
            linewidths=1.0, cmap=cmap, vmin=0, vmax=1,  cbar = True,annot=True,
            square=True) # , fmt=".1f", cbar_kws={"shrink": .8},  annot=True, fmt=".1f",vmin=0, vmax=2, center = 1.0,
    # # ticks
    yticks = [f'layer{i+1}' for i in range(12)]
    xticks = [f'layer{i+1}' for i in range(12)]
    # xticks = [i.upper() for i in corr.columns]
    plt.yticks(plt.yticks()[0], labels=yticks, rotation=0, fontsize =20)
    plt.xticks(plt.xticks()[0], labels=xticks, rotation=90, fontsize =20)
    plt.savefig(filename, bbox_inches='tight', dpi = 300)
    plt.show()

def model_training(args, data, model, total_steps, print_every, eval_every, save_every, ckpt_save_path, cuda_available, device):
    import os
    
    if os.path.exists(ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(ckpt_save_path, exist_ok=True)

    wandb.login()
    run = wandb.init(project='document_generation', config=args, name=args.mode)

    max_save_num = 1

    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print ('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    # Collect parameters of lm_heads
    if args.only_train_lm_heads:
        lm_heads_params = []
        for lm_head in model.multi_lm_heads:
            lm_heads_params.extend(lm_head.parameters())
        optimizer = AdamW(lm_heads_params, lr=args.learning_rate)
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, eval_valid, save_valid = False, False, False
    train_loss, train_cl_loss, min_val_loss = 0., 0., 1e10
    train_ave_bleu = 0.
    if args.mode == 'standard':
        train_all_layer_loss = [0] * 1
    else:
        train_all_layer_loss = [0] * 12

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    model.train()
    number_of_saves = 0

    for all_batch_step in tqdm(range(total_steps * gradient_accumulation_steps)):
        #all_batch_step += 1
        train_batch_input_tensor, train_batch_labels, _ = data.get_next_train_batch(batch_size_per_gpu * number_of_gpu)
        if cuda_available:
            train_batch_input_tensor = train_batch_input_tensor.cuda(device)
            train_batch_labels = train_batch_labels.cuda(device)
        loss, all_layer_loss = model(train_batch_input_tensor, train_batch_labels, args.margin)
        # print("[debug] loss: ", loss)   
        # print("[debug] all_layer_loss: ", all_layer_loss)
       
        #loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        for i, one_layer_loss in enumerate(all_layer_loss):
            train_all_layer_loss[i] += one_layer_loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # parameter update
        if all_batch_step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_batch_acm += 1
            print_valid, eval_valid, save_valid = True, True, True

        # print intermediate result
        if effective_batch_acm % print_every == 0 and print_valid:
            denominator = (effective_batch_acm - (number_of_saves * save_every)) * gradient_accumulation_steps
            one_train_loss = train_loss / denominator
            one_train_all_layer_loss = [x / denominator for x in train_all_layer_loss]
            
            if args.mode == 'standard':
                print ('At training steps {}, training loss is {}'.format(effective_batch_acm, one_train_loss))
                wandb.log({'train_loss': one_train_loss})
            else:
                print ('[Print] At training steps {}, training loss is {}, each layer loss is {}'.format(effective_batch_acm, 
                    one_train_loss, one_train_all_layer_loss))
                wandb.log({'train_loss': one_train_loss})
                for i, one_layer_loss in enumerate(one_train_all_layer_loss):
                    wandb.log({'train_loss_layer_{}'.format(i): one_layer_loss})
            print_valid = False


        if effective_batch_acm % eval_every == 0 and eval_valid:
            model.eval()
            one_val_loss, all_layer_val_loss,_, _ = eval_model(args, model, data, cuda_available, device)
            model.train()
            
            if args.mode == 'standard':
                print ('At training steps {}, training MLE loss is {}, validation loss is {}, perplexity is {}'.format(effective_batch_acm, 
                    one_train_loss, one_val_loss, round(np.exp(one_val_loss),3)))
                wandb.log({'train_loss': one_train_loss, 'val_loss': one_val_loss, 'val_ppl': np.exp(one_val_loss)})
            else:
                all_layer_val_loss = np.array(all_layer_val_loss)
                all_layer_ppl = np.exp(all_layer_val_loss)
                all_layer_ppl = [round(x, 3) for x in all_layer_ppl]
                print ('[Eval] At training steps {}, training MLE loss is {}, validation loss is {}, each layer perplexity is {}'.format(effective_batch_acm, 
                    one_train_loss, one_val_loss, all_layer_ppl))
                wandb.log({'train_loss': one_train_loss, 'val_loss': one_val_loss})
                for i, one_layer_ppl in enumerate(all_layer_ppl):
                    wandb.log({'val_ppl_layer_{}'.format(i): one_layer_ppl})

            eval_valid = False
                

        # saving result
        if effective_batch_acm % save_every == 0 and save_valid:
            number_of_saves += 1

            save_valid = False
            one_train_loss = train_loss / (save_every * gradient_accumulation_steps)
            #one_train_cl_loss = train_cl_loss / (save_every * gradient_accumulation_steps)

            model.eval()
            one_val_loss, all_layer_val_loss, _, _ = eval_model(args, model, data, cuda_available, device)
            model.train()
            
            if args.mode == 'standard':
                print ('At training steps {}, training MLE loss is {}, validation loss is {}, perplexity is {}'.format(effective_batch_acm, 
                    one_train_loss, one_val_loss, round(np.exp(one_val_loss),3)))
                wandb.log({'train_loss': one_train_loss, 'val_loss': one_val_loss, 'val_ppl': np.exp(one_val_loss)})
            else:
                all_layer_val_loss = np.array(all_layer_val_loss)
                all_layer_ppl = np.exp(all_layer_val_loss)
                all_layer_ppl = [round(x, 3) for x in all_layer_ppl]
                print ('[Save] At training steps {}, training MLE loss is {}, validation loss is {}, each layer perplexity is {}'.format(effective_batch_acm, 
                    one_train_loss, one_val_loss, all_layer_ppl))
                wandb.log({'train_loss': one_train_loss, 'val_loss': one_val_loss})
                for i, one_layer_ppl in enumerate(all_layer_ppl):
                    wandb.log({'val_ppl_layer_{}'.format(i): one_layer_ppl})
                

            train_loss, train_cl_loss = 0., 0.

            if one_val_loss < min_val_loss:
                # in finetuning stage, we always save the model
                min_val_loss = min(one_val_loss, min_val_loss)
                print ('Saving model...')
                
                if args.mode == 'standard':
                    one_val_ppl = np.exp(one_val_loss)
                    one_val_ppl = round(one_val_ppl, 3)
                    save_name = 'training_step_{}_train_loss_{}_dev_loss_{}_dev_ppl_{}'.format(effective_batch_acm,
                    round(one_train_loss,5), round(one_val_loss,5), one_val_ppl)
                else:
                    one_val_ppl = np.exp(all_layer_val_loss)
                    one_val_ppl = [round(x, 3) for x in one_val_ppl]
                    one_val_ppl_str = "_".join([str(x) for x in one_val_ppl])
                    save_name = 'training_step_{}_train_loss_{}_dev_loss_{}'.format(effective_batch_acm,
                    round(one_train_loss,5), round(one_val_loss,5))

                model_save_path = ckpt_save_path + '/' + save_name
                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)
                if cuda_available and torch.cuda.device_count() > 1:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)
                print ('Model Saved!')

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                import os
                from operator import itemgetter
                fileData = {}
                test_output_dir = ckpt_save_path
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('training_step'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))

                if len(sortedFiles) < max_save_num:
                    pass
                else:
                    delete = len(sortedFiles) - max_save_num
                    for x in range(0, delete):
                        one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                        os.system('rm -r ' + one_folder_name)
                print ('-----------------------------------')
                # --------------------------------------------------------------------------------------------- #
    return model

