import torch
# from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import argparse
import json
from tqdm import tqdm

import datetime

# Get the current date and time
current_datetime = datetime.datetime.now()

# Format the date and time as a string
formatted_timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

def evaluate(instruction, tokenizer, model, input=None, **kwargs):
    args = kwargs.get("args")
    prompt = generate_prompt(instruction, args, input)
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda() # these are integers encoded from words
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.9,
        num_beams=1,
        **kwargs,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=1500,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s) # this will return a fully-wholely description like "Below is an instruction....Response:..."
    if args.prompt_mode == "alpaca":
        return output.split("### Response:\n")[1].strip()
    elif args.prompt_mode not in ['CoT','CoT_direct']:
        return output.split("### Output:")[1].strip()
    else:
        return output.split("### Output:\nLet's think step by step.\n")[1].strip()

def generate_prompt(instruction, args, input=None):
    if args.prompt_mode=="continue":
        return f"""Below is a text extending task. you will be given an incomplete text and requested to provide a continuation of said text in the !!!LAION-6plus-style!!!.
### Input:
{instruction}
### Output:"""
    elif args.prompt_mode=="blip_pair" or args.prompt_mode=="inter":
        return f"""Below is a text imitation task. You will be given a text description and asked to rewrite it in a different style.
### Input:
{instruction}
### Output:"""
    elif args.prompt_mode=="CoT" or args.prompt_mode=="CoT_direct":
        return f"""Below is a text rewriting task, finish it step by step. You will be given a prompt for image generation. Your objective is to rewrite it to align with the AI generation model called !!!stable-diffusion!!!
### Input:
{instruction}
### Output:
Let's think step by step.
"""

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d','--dir',dest="dir_of_hf_w", type=str, help='dir folder of hf weights, e.g., xxx.bin')
    parser.add_argument('--out-to-txt',dest="out_to_txt", action='store_true', help='store output text to out_generation.txt')
    parser.add_argument('--load-in-8bit',dest="load_in_8bit", action='store_true', help='')
    parser.add_argument('-i','--interact',dest="interact", action='store_true', help='')
    parser.add_argument("--coco",'--coco_caption',dest="coco_caption", action='store_true', help='generate refined captions for coco, save as json')
    parser.add_argument("--coco_file",type=str,default="./data/COCO/coco_captions_val.json")
    parser.add_argument("--db",'--db_caption',dest="db_caption", action='store_true', help='generate refined captions for db, save as json')
    parser.add_argument('--prompt-mode',dest="prompt_mode",type=str, help='[blip_pair, continue,inter,CoT]')
    parser.add_argument('--batch_size',dest="batch_size",type=int, default=4, help='batch_size=12')
    args = parser.parse_args()
    
    if args.interact:
        print("For testing, please input your prompt:\n")
        instruction_from_terminal = " "
        while instruction_from_terminal!="exit":
            instruction_from_terminal = input("Your prompt: ")
            pred = evaluate(instruction_from_terminal,tokenizer, model,args=args)
            print("Response:", pred)
            print()
    elif args.coco_caption:
        results = []
        import os
        output_file = args.coco_file[:-4]+f'_CoT_{formatted_timestamp}.txt'

        tokenizer = LlamaTokenizer.from_pretrained(args.dir_of_hf_w)
        IGNORE_INDEX = -100
        DEFAULT_PAD_TOKEN = "[PAD]"
        DEFAULT_EOS_TOKEN = "</s>"
        DEFAULT_BOS_TOKEN = "</s>"
        DEFAULT_UNK_TOKEN = "</s>"
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN
            }
        )
        tokenizer.padding_side = "left"
        model = LlamaForCausalLM.from_pretrained(
            args.dir_of_hf_w,
            load_in_8bit=args.load_in_8bit, # by Kris: True may save memory (16GB to 10GB), but slower
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        with open(args.coco_file,'r') as fp:
            coco_txt = fp.readlines()

        progress_bar = tqdm(total=len(coco_txt))
        batch_size = args.batch_size

        for i in range(0, len(coco_txt), batch_size):
            batch = coco_txt[i:i + batch_size]
            
            prompts = []
            for instruction in batch:
                prompt = generate_prompt(instruction, args, input)
                prompts += [prompt]

            inputs = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
            
            input_ids = inputs["input_ids"].cuda() # these are integers encoded from words

            generation_config = GenerationConfig(
                # temperature=0.8,
                # top_p=0.9,
                num_beams=1,
                pad_token_id=32000,
            )
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=1500,
                attention_mask=inputs["attention_mask"].cuda(),
            )

            with open(output_file, 'a') as wf:
                for ii in range(args.batch_size):
                    s = generation_output.sequences[ii]
                    output = tokenizer.decode(s)

                    if args.prompt_mode == "alpaca":
                        pred = output.split("### Response:\n")[1].strip()
                    elif args.prompt_mode not in ['CoT','CoT_direct']:
                        pred = output.split("### Output:")[1].strip()
                    else:
                        pred = output.split("### Output:\nLet's think step by step.\n")[1].strip()

                    pred = pred.replace('</s>','')
                    pred = pred.replace('\n',' ')
                    
                    if pred.find("!!!LAION-6plus-style!!!") == -1:
                        result = batch[ii]
                        print('Error : refine failed! use origin caption.')
                    else:
                        result = pred.split("!!!LAION-6plus-style!!!")[1].strip()
                    if "[PAD]" in result:
                        result = result.split("[PAD]")[0]
                    print("result =>\n", result)
                    print("-"*40)
                    wf.write(result + '\n')

            progress_bar.update(len(batch))
    elif args.db_caption:
        new_caption = []
        with open('./prompts/prompts_db_c.txt','r') as fp:
            prompts = fp.readlines()
        for idx,prompt in enumerate(tqdm(prompts)):
            pred = evaluate(prompt,tokenizer,model,args=args)
            pred = pred.replace('</s>','')
            new_caption.append({'caption':prompt,'refined':pred})
        with open(f"./data/diffusiondb/db_{args.prompt_mode}.json","w") as fp:
            json.dump(new_caption,fp,indent=4)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(args.dir_of_hf_w)
        model = LlamaForCausalLM.from_pretrained(
            args.dir_of_hf_w,
            load_in_8bit=args.load_in_8bit, # by Kris: True may save memory (16GB to 10GB), but slower
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        ctx = ""
        for instruction in [
            'A big bus in a parking lot next to a big building ',
            'There is a clock right outside of the tall building.',
            'Two trains on the track at a railway.',
            'The bench is in a shady area surrounded by plants',
            'an image of man having lunch with kids',
            'A group of people sitting at a table with food.',
            'Flowers neatly arranged in a clear vase filled with water. ',
            'Two men playing frisbee together on a field',
            'A baseball game where the batter is waiting for the pitch.',
            'A motorcycle is parked inside of a building.'
        ]:
            print("Instruction:", instruction)
            pred = evaluate(instruction, tokenizer, model,args=args)
            ctx += f"Instruction: {instruction}\n" + f"Response: {pred}\n"
            print("Response:", pred)
            print()

        if args.out_to_txt:
            with open("./out_generation.txt",'w') as fp:
                fp.write(ctx)

if __name__ == "__main__":
    # testing code for readme
    main()