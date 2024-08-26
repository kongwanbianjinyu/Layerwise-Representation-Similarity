# For standard model
# CUDA_VISIBLE_DEVICES=1 python eval_logits.py\
#     --ckpt_path standard_wikitext103/training_step_40000_train_loss_3.2994_dev_loss_3.18824_dev_ppl_24.246 \
#     --train_path data/language_modelling/wikitext103/wikitext103_raw_v1_train.txt \
#     --dev_path data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
#     --test_path data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
#     --number_of_gpu 1\
#     --batch_size_per_gpu 8\
#     --save_npy_path gpt2_saturate_event.npy \


# For aligned model
# CUDA_VISIBLE_DEVICES=1 python eval_logits.py\
#     --ckpt_path aligned_wikitext103/training_step_40000_train_loss_4.33256_dev_loss_4.11178_dev_ppl_1.084_1.152_1.219_1.285_1.354_1.404_1.459_1.511_1.568_1.621_1.684_1.739 \
#     --train_path data/language_modelling/wikitext103/wikitext103_raw_v1_train.txt \
#     --dev_path data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
#     --test_path data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
#     --number_of_gpu 1\
#     --batch_size_per_gpu 8\
#     --save_npy_path alignedgpt2_saturate_event.npy \


# For multi_exit model
CUDA_VISIBLE_DEVICES=1 python eval_logits.py\
    --ckpt_path multi_exit_wikitext103_saveheads/training_step_4000_train_loss_8.51643_dev_loss_8.43093 \
    --train_path data/language_modelling/wikitext103/wikitext103_raw_v1_train.txt \
    --dev_path data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
    --test_path data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
    --number_of_gpu 1\
    --batch_size_per_gpu 8\
    --mode multi_exit\
    --save_npy_path multiexitgpt2_saturate_event.npy \