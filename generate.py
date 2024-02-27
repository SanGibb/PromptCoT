import torch
# from peft import PeftModel
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig
import argparse
import json
from tqdm import tqdm

def evaluate(instruction, tokenizer, model, input=None, **kwargs):
    args = kwargs.get("args")
    prompt = generate_prompt(instruction, args, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda() # these are integers encoded from words
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        num_beams=4,
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
    parser.add_argument('--load-in-8bit',dest="load_in_8bit", action='store_true', help='')
    parser.add_argument('-i','--interact',dest="interact", action='store_true', help='')
    parser.add_argument("--coco",'--coco_caption',dest="coco_caption", action='store_true', help='generate refined captions for coco, save as json')
    parser.add_argument("--coco_file",type=str,default="./data/COCO/coco_captions_val_comma.json")
    parser.add_argument('--prompt-mode',dest="prompt_mode",type=str, help='[blip_pair, continue,inter,CoT]')

    args = parser.parse_args()
    
    if args.interact:
        print("For testing, please input your prompt:\n")
        instruction_from_terminal = " "
        while instruction_from_terminal!="exit":
            instruction_from_terminal = input("Your prompt: ")
            pred = evaluate(instruction_from_terminal,tokenizer, model,args=args)
            print("Response:", pred)
            print()
        # if type "exit" in terminal, will go on for some examples.
    elif args.coco_caption:
        results = []
        with open(args.coco_file,'r') as fp:
            coco_txt = fp.readlines()
        n_have = 0
        import os
        if os.path.exists(args.coco_file[:-4]+'_CoT.txt'):
            with open(args.coco_file[:-4]+'_CoT.txt','r') as fp:
                data = fp.readlines()
                for _ in data: 
                    if _.strip() != "":
                        results.append(_.strip()) 
                n_have = len(results)
                print(n_have)
                print(results[:5])
        if n_have==50:
            return

        tokenizer = LLaMATokenizer.from_pretrained(args.dir_of_hf_w)
        model = LLaMAForCausalLM.from_pretrained(
            args.dir_of_hf_w,
            load_in_8bit=args.load_in_8bit, # by Kris: True may save memory (16GB to 10GB), but slower
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        for idx,txt in enumerate(tqdm(coco_txt)):
            if idx<n_have:
                continue
            pred = ""
            while pred.find("!!!LAION-6plus-style!!! ")==-1:
                pred = evaluate(txt,tokenizer,model,args=args)
                pred = pred.replace('</s>','')
                pred = pred.replace('\n',' ')
                print(idx, txt)
                print(pred)
                txt += " pad"
            results.append(pred.split("!!!LAION-6plus-style!!! ")[1].strip())
            with open(args.coco_file[:-4]+'_CoT.txt','w') as fp:
                for r in results:
                    print(r,file=fp)
        with open(args.coco_file[:-4]+'_CoT.txt','w') as fp:
            for r in results:
                print(r,file=fp)
    else:
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
            print("Response:", pred,"\n")
        with open("./data/prompts/example.txt",'w') as fp:
            fp.write(ctx)

if __name__ == "__main__":
    main()