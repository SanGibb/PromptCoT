# 共计16个部分
# CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=6699 generate_gptprompt.py \
# i=1
# CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=6701 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

# i=2
# CUDA_VISIBLE_DEVICES=1 nohup torchrun --nproc_per_node=1 --master_port=6702 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

# i=3
# CUDA_VISIBLE_DEVICES=2 nohup torchrun --nproc_per_node=1 --master_port=6703 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

# i=4
# CUDA_VISIBLE_DEVICES=3 nohup torchrun --nproc_per_node=1 --master_port=6704 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

# i=5
# CUDA_VISIBLE_DEVICES=4 nohup torchrun --nproc_per_node=1 --master_port=6705 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

# i=6
# CUDA_VISIBLE_DEVICES=5 nohup torchrun --nproc_per_node=1 --master_port=6706 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

# i=7
# CUDA_VISIBLE_DEVICES=6 nohup torchrun --nproc_per_node=1 --master_port=6707 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

# i=8
# CUDA_VISIBLE_DEVICES=7 nohup torchrun --nproc_per_node=1 --master_port=6708 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco \
# --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_data_hhl/coco_captions_val_${i}_of_8.json \
# --prompt-mode CoT_direct \
# > ./generate_gptprompt_${i}.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=6719 generate_gptprompt_hhl.py \
--dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
--coco \
--coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_refine_cot_hhl/prompts_rb_part_2.txt \
--prompt-mode CoT_direct \
--batch_size 8 \
> ./generate_gptprompt_hhl_part2.log 2>&1 &


# <<'COMMENT'
# # for debug

# CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=6706 generate_gptprompt_hhl.py \
# --dir /data/yaojunyi/gpt-prompt/stanford_alpaca/LOGs/blip2CoT \
# --coco --coco_file /data/yaojunyi/gpt-prompt/stanford_alpaca/coco_refine_cot_hhl/temp_debug.txt \
# --prompt-mode CoT_direct

# COMMENT