Run "python train.py -h" to see usage.

Example of train a model:

python train.py -data_dir ~/data-simplification/wikilarge/ -dataset_name wiki_simplification -partition train_filtered_sim_0.9 -save_dir wikilarge_filtered_sim_0.9_copy -max_epochs 100 -patience 10 -dyanmic_lr -encoder_type transformer -decoder_type rnn -enable_copy

python train.py -train_src_file /c10_data/mroemmele/text_simplification/wikilarge/wiki.full.aner.ori.sample.src -train_tgt_file  /c10_data/mroemmele/text_simplification/wikilarge/wiki.full.aner.ori.sample.src -eval_src_file  /c10_data/mroemmele/text_simplification/wikilarge/wiki.full.aner.ori.sample.src -eval_tgt_file  /c10_data/mroemmele/text_simplification/wikilarge/wiki.full.aner.ori.sample.src  -save_dir test_model10 -max_epochs 200  -log_iterations 15 -patience 100  -encoder_type transformer -decoder_type rnn -enable_copy -learning_rate 0.0001