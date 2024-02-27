export CUDA_VISIBLE_DEVICES=2,3,5,6

# for text continue training
output_dir="LOGs/tcontinue_gaussian"
torchrun --nproc_per_node=4 --master_port=889 train_prompt.py \
--model_name_or_path ../llama_weights/llama_weights_hf/llama-7b-hf \
--prompt_mode "continue" \
--data_path ./tcontinue_data_gaussion.json \
--bf16 True \
--output_dir $output_dir \
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \ 
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
--tf32 True

# for training dummy
output_dir="LOGs/dummy"
torchrun --nproc_per_node=4 --master_port=889 train_prompt.py \
--model_name_or_path ../llama_weights_hf/llama-7b-hf \
--data_path ./t2t_data.json \
--bf16 True \
--output_dir $output_dir \
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
--tf32 True

# for training 7B
output_dir="LOGs/output_7B"
torchrun --nproc_per_node=4 --master_port=889 train.py \
--model_name_or_path ../llama_weights_hf/llama-7b-hf \
--data_path ./alpaca_data.json \
--bf16 True \
--output_dir $output_dir \
--num_train_epochs 3 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
--tf32 True

# for training 30B: failed OOM
torchrun --nproc_per_node=8 --master_port=889 train.py \
--model_name_or_path ../llama_weights_hf/llama-30b-hf \
--data_path ./alpaca_data.json \
--bf16 True \
--output_dir LOGs/output_30B \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 2000 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
--tf32 True

# for inference 
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=889 enerate_gptprompt.py -d LOGs/output/ --out-to-txt