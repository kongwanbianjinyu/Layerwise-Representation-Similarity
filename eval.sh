#./standard_wikitext103/training_step_40000_train_loss_3.2994_dev_loss_3.18824_dev_ppl_24.246
#./aligned_wikitext103/training_step_40000_train_loss_4.33256_dev_loss_4.11178_dev_ppl_1.084_1.152_1.219_1.285_1.354_1.404_1.459_1.511_1.568_1.621_1.684_1.739 
CUDA_VISIBLE_DEVICES=0 python eval.py\
    --ckpt_path  ./aligned_alternating_wikitext103/training_step_40000_train_loss_4.72024_dev_loss_79.85051 \
    --train_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_train.txt \
    --dev_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_validation.txt\
    --test_path ../data/language_modelling/wikitext103/wikitext103_raw_v1_test.txt\
    --number_of_gpu 1\
    --batch_size_per_gpu 8\
