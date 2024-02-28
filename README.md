# PromptCoT
This is an initial repo for PromptCoT. The repo contains:
* the code for inferrence
* the code for training
* the COCO validation data we use for evaluation
* the pre-trained model at 7B size

## Installation
```bash
pip install -r requirements.txt
```
Then, install the particular fork of Hugging Face's transformers library. The hash of the specific commit we installed was `68d640f7c368bcaaaecfc678f11908ebbd3d6176`.

## Finetune
* Download [alpaca training data](TODO) to `/data/alapca`
* Download [specific training data](TODO) to `/data/LAION` 
* run script `train.sh`

## Inferrence
* Download [pretrained model](TODO) to `LOGs`.
* Save all prompts you want to refine in `./data/prompts/caption.txt`. (here we use data from COCO validation dataset)
* run script `generate.sh`