python run_seq2seq_flax.py \
	--max_source_length 128 \
	--train_file train-encoded.tsv \
	--validation_file val-encoded.tsv \
	--output_dir output \
	--per_device_train_batch_size 56 \
	--per_device_eval_batch_size 56 \
	--preprocessing_num_workers 80 \
	--warmup_steps 250 \
	--gradient_accumulation_steps 8 \
	--do_train \
	--do_eval \
	--adafactor \
	--num_train_epochs 6 \
	--log_model \
	--learning_rate 0.005
