CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
--master_port=6699 generate.py \
--dir ./LOGs/CoT \
--coco \
--coco_file data/prompts/caption.txt \
--prompt-mode CoT