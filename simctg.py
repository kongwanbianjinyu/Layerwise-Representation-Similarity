import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from loss_func import contrastive_loss

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from transformers import GPT2LMHeadModel
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')
class SimCTG(nn.Module):
    def __init__(self, args, model_name, pad_token_id):
        super(SimCTG, self).__init__()
        from transformers import AutoTokenizer, GPT2LMHeadModel
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.embed_dim = self.model.config.hidden_size
        self.pad_token_id = pad_token_id
        self.hidden_states = None
        self.stage = 0

    def compute_all_layer_logits(self, input_ids):
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        # outputs.hidden_states shape : 13 * (batch_size, sequence_length, hidden_size)
        # get all layer logits by applying lm_head to each hidden state
        lm_head = self.model.lm_head
        all_layer_logits = []
        for each_layer_hidden_states in outputs.hidden_states[1:]:
            each_layer_logits = lm_head(each_layer_hidden_states)
            all_layer_logits.append(each_layer_logits)
        # all_layer_logits shape: 12 * (batch_size, sequence_length, vocab_size)
        return all_layer_logits

    def compute_cosine_similarity_with_last_hidden_state(self, input_ids):
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1] # shape: bsz x seqlen x embed_dim
        all_layer_cosine_similarity = []
        for i, each_layer_hidden_states in enumerate(outputs.hidden_states[1:]):
            cos_sim = self.compute_cosine_similarity(last_hidden_states, each_layer_hidden_states)
            all_layer_cosine_similarity.append(cos_sim)
        return all_layer_cosine_similarity
    
    def compute_cosine_similarity(self, feature_l1, feature_l2):
        # remove global mean of feature_l1 and feature_l2 along the last dimension
        feature_l1 = feature_l1 - torch.mean(feature_l1, dim=-1, keepdim=True)
        feature_l2 = feature_l2 - torch.mean(feature_l2, dim=-1, keepdim=True)

        # # normalize feature_l1 and feature_l2 along the last dimension
        norm_feature_l1 = feature_l1 / feature_l1.norm(dim=-1, keepdim=True)
        norm_feature_l2 = feature_l2 / feature_l2.norm(dim=-1, keepdim=True)
        cosine = torch.sum(norm_feature_l1 * norm_feature_l2, dim=-1)  # shape: bsz x seqlen
        cosine_similarity = cosine.mean()  # scalar
        return cosine_similarity

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels, margin):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        all_layer_loss = []
        if self.args.mode == 'standard':
            loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            all_layer_loss.append(loss)

        elif self.args.mode == 'aligned':
            loss = 0
            for i, hidden_state in enumerate(outputs.hidden_states[1:]):
                hidden_logits = self.model.lm_head(hidden_state)
                wi = 2*(i+1)/(12*(12+1))
                layer_i_loss = wi * train_fct(hidden_logits.view(-1, self.vocab_size), labels.view(-1))
                loss += layer_i_loss
                all_layer_loss.append(layer_i_loss)

        elif self.args.mode == "aligned_alternating":
            for i, hidden_state in enumerate(outputs.hidden_states[1:]):
                hidden_logits = self.model.lm_head(hidden_state)
                wi = 2*(i+1)/(12*(12+1))
                layer_i_loss = wi * train_fct(hidden_logits.view(-1, self.vocab_size), labels.view(-1))
                all_layer_loss.append(layer_i_loss)

            if self.stage % 2 == 0:
                loss = all_layer_loss[-1] * 78 / 12
            else:
                loss = sum(all_layer_loss)
            self.stage += 1

        # norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        # cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1,2)) 
        # assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])
        # cl_loss = contrastive_loss(margin, cosine_scores, input_ids, self.pad_token_id, prefix_len=0)
        return loss, all_layer_loss

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        token_num_sum = torch.sum(mask)
        each_layer_loss = []
        each_layer_acc = []
        L = len(outputs.hidden_states[1:])
        pairwise_cosine_similarity = torch.eye(L)
        
        all_layer_loss = 0
        for i, hidden_state in enumerate(outputs.hidden_states[1:]):
            hidden_logits = self.model.lm_head(hidden_state)
            layer_i_loss = val_fct(hidden_logits.view(-1, self.vocab_size), labels.view(-1))
            layer_i_loss_each_token_avg = torch.sum(layer_i_loss * mask.view(-1)) / token_num_sum
            all_layer_loss += layer_i_loss_each_token_avg
            each_layer_loss.append(layer_i_loss_each_token_avg)

            # compute accuracy
            hidden_logits_flat = hidden_logits.view(-1, self.vocab_size)
            labels_flat = labels.view(-1)
            preds = torch.argmax(hidden_logits_flat, dim=-1)
            correct_preds = preds.eq(labels_flat)
            correct_preds_masked = correct_preds.float() * mask.view(-1)
            accuracy = correct_preds_masked.sum() / token_num_sum
            each_layer_acc.append(accuracy)
        
        # compute pairwise cosine similarity
        for l1, hidden_state_l1 in enumerate(outputs.hidden_states[1:]):
            for l2, hidden_state_l2 in enumerate(outputs.hidden_states[1:]):
                if l1 >= l2:
                    continue
                cos_sim_layer_l1_l2 = self.compute_cosine_similarity(hidden_state_l1, hidden_state_l2, mask)
                pairwise_cosine_similarity[l1, l2] = cos_sim_layer_l1_l2
                pairwise_cosine_similarity[l2, l1] = cos_sim_layer_l1_l2

        
        return all_layer_loss, each_layer_loss,  each_layer_acc, pairwise_cosine_similarity

    # def compute_cosine_similarity(self, feature_l1, feature_l2, mask):
    #     # remove global mean of feature_l1 and feature_l2 along the last dimension
    #     feature_l1 = feature_l1 - torch.mean(feature_l1, dim=-1, keepdim=True)
    #     feature_l2 = feature_l2 - torch.mean(feature_l2, dim=-1, keepdim=True)

    #     # print the norm of global mean, very close to 0
    #     # print ('norm of global mean of feature_l1: {}'.format(torch.mean(feature_l1, dim=-1, keepdim=True).norm()))
    #     # print ('norm of global mean of feature_l2: {}'.format(torch.mean(feature_l2, dim=-1, keepdim=True).norm()))

    #     # normalize feature_l1 and feature_l2 along the last dimension
    #     norm_feature_l1 = feature_l1 / feature_l1.norm(dim=-1, keepdim=True)
    #     norm_feature_l2 = feature_l2 / feature_l2.norm(dim=-1, keepdim=True)
    #     cosine = norm_feature_l1 * norm_feature_l2 # shape: bsz x seqlen x embed_dim

    #     # sum along the last dimension
    #     cosine_similarity = torch.sum(cosine, dim=-1) # shape: bsz x seqlen
    #     cosine_similarity_flat = cosine_similarity.view(-1)

    #     cos_sim_layer_l1_l2 = torch.sum(cosine_similarity_flat * mask.view(-1)) / torch.sum(mask)
    #     return cos_sim_layer_l1_l2
    

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    # decoding functions
    # ------------------------------------------------------- #
    def slow_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        # sanity check
        # sanity check
        assert alpha >= 0. and alpha <= 1.0

        from utlis import ContrastiveDecodingOneStep
        for step in range(decoding_len):
            input_ids = ContrastiveDecodingOneStep(self, input_ids, beam_width, alpha)
        return input_ids[0]

    def fast_contrastive_search(self, input_ids, beam_width, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        self.model.eval()
        from utlis import ContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        # fast mode
        batch_size, seqlen = input_ids.size()
        #generated = [[] for _ in range(batch_size)]
        generated = [item for item in input_ids.tolist()]
        past_key_values = None
        last_hidden_states = None
        logits = None
        for step in range(decoding_len):
            input_ids, past_key_values, last_hidden_states, logits = ContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
            )
            tokens = input_ids.squeeze(dim=-1).tolist()
            for idx, t in enumerate(tokens):
                generated[idx].append(t)
        return generated[0]

    def diverse_contrastive_search(self, input_ids, sample_step, nucleus_p, beam_width, alpha, decoding_len):
        '''
            sample_step: 
                number of steps to decode with nucleus sampling, 
                for the remaining steps we use contrastive search
            decoding_len: 
                the total number of generated tokens
            beam_width: 
                size of candidate pool during decoding
            alpha: 
                regulates importance of model confidence and degeneration penalty

        '''
        contrastive_step = decoding_len - sample_step
        _, prefix_len = input_ids.size()
        # first do sample
        input_ids = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+sample_step, 
                            top_p=nucleus_p,
                            top_k=0)
        # then do contrastive search
        output = self.fast_contrastive_search(input_ids, beam_width, alpha, contrastive_step)
        return output

    def greedy_search(self, input_ids, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len,
                            pad_token_id=self.pad_token_id)
        return output[0]

    def beam_search(self, input_ids, beam_width, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            max_length=prefix_len+decoding_len, 
                            num_beams=beam_width,
                            pad_token_id=self.pad_token_id)
        return output[0]


    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len):
        _, prefix_len = input_ids.size()
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=prefix_len+decoding_len, 
                            top_p=nucleus_p,
                            top_k=0,
                            pad_token_id=self.pad_token_id)
        return output[0]
    # ------------------------------------------------------- #

    def compute_correlation_matrix(self, input_ids):        
        _, seq_len = input_ids.size()
        hidden = self.model.base_model(input_ids).last_hidden_state
        #print (hidden)
        norm_hidden = hidden / hidden.norm(dim=2, keepdim=True)
        correlation_matrix = torch.matmul(norm_hidden, norm_hidden.transpose(1,2)).view(seq_len, seq_len)
        return correlation_matrix.detach().numpy()

    # to produce similarity matrix heatmap
    def save_token_similarity_map(self, input_ids, save_name):
        input_ids = torch.LongTensor(input_ids).view(1, -1)
        correlation_matrix = self.compute_correlation_matrix(input_ids)
        df = pd.DataFrame(correlation_matrix)
        df.to_string(index=False)
        df.style.hide_index()
        df.style.hide_index()
        sns.heatmap(df, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.savefig(save_name, format='png', dpi=500, bbox_inches = 'tight')
        plt.show()


class EarlyExitGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, exit_layer):
        super().__init__(config)
        self.exit_layer = exit_layer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        outputs = super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # We need to output hidden states
            return_dict=return_dict,
        )
        # Get hidden states from the desired exit layer
        exit_hidden_states = outputs.hidden_states[self.exit_layer]

        # Pass hidden states through the language model head to get logits
        lm_logits = self.lm_head(exit_hidden_states)
        outputs.logits = lm_logits
        return outputs

class EarlyExitSimCTG(SimCTG):
    def __init__(self, args, pad_token_id):
        super().__init__(args, args.ckpt_path, pad_token_id)
        self.model = EarlyExitGPT2LMHeadModel.from_pretrained(args.ckpt_path, exit_layer=args.exit_layer)