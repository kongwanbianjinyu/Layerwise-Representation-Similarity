CUDA_VISIBLE_DEVICES=0 python inference.py\
    --ckpt_path ./aligned_alternating_wikitext103/training_step_40000_train_loss_4.72024_dev_loss_79.85051 \
    --dev_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
    --test_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
    --prefix_len 32\
    --decoding_len 128\
    --num_per_instance 1\
    --k 8\
    --alpha 0.6\
    --save_path aligned_alternating.json
