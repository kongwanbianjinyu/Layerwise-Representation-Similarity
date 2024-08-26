# model name: gpt2
# lr : 2e-5
export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=4 python train.py\
    --model_name gpt2 \
    --train_path data/language_modelling/wikitext103/wikitext103_raw_v1_train.txt\
    --dev_path data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
    --test_path data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
    --mode multi_exit \
    --margin 0.5\
    --max_len 256\
    --number_of_gpu 1\
    --batch_size_per_gpu 8\
    --gradient_accumulation_steps 16\
    --effective_batch_size 128\
    --total_steps 40000\
    --print_every 10\
    --eval_every 100\
    --save_every 1000\
    --learning_rate 2e-5\
    --save_path_prefix ./multi_exit_wikitext103_saveheads/ \
    # > ./train_multi_exit_wikitext103.log 2>&1 &