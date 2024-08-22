CUDA_VISIBLE_DEVICES=1 nohup python early_exit_inference.py\
    --ckpt_path ./aligned_alternating_wikitext103/training_step_40000_train_loss_4.72024_dev_loss_79.85051 \
    --exit_layer 10 \
    --decode_method sampling \
    --dev_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
    --test_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
    --prefix_len 32\
    --decoding_len 128\
    --num_per_instance 1\
    --k 8\
    --alpha 0.6\
    --p 0.95 \
    --save_path ./outputs/aligned_alternating_sampling_exit10.json > ./logs/aligned_alternating_sampling_exit10.log 2>&1 &

#./standard_wikitext103/training_step_40000_train_loss_3.2994_dev_loss_3.18824_dev_ppl_24.246
#./aligned_wikitext103/training_step_40000_train_loss_4.33256_dev_loss_4.11178_dev_ppl_1.084_1.152_1.219_1.285_1.354_1.404_1.459_1.511_1.568_1.621_1.684_1.739