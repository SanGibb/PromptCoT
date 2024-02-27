# PromptCoT
This is an initial repo for PromptCoT. The repo contains:
* the code for inferrence
* the COCO validation data we use for evaluation
* the pre-trained model at 7B size

## Installation
```bash
pip install -r requirements.txt
```
Then, install the particular fork of Hugging Face's transformers library. The hash of the specific commit we installed was `68d640f7c368bcaaaecfc678f11908ebbd3d6176`.

## Finetune

## Inferrence
* Download [pretrained model](TODO) and put it in `LOGs`.
* Save all prompts you want to refine in `./data/prompts/caption.txt`. (here we use data from COCO validation dataset)
* run script below
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
--master_port=6699 generate.py \
--dir ./LOGs/CoT \
--coco \
--coco_file data/COCO/coco_captions_val.json \
--prompt-mode CoT
```