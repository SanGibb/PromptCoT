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

def generate_prompt_hhl(instruction, prompt_mode):
    if prompt_mode=="continue":
        return f"""Below is a text extending task. you will be given an incomplete text and requested to provide a continuation of said text in the !!!LAION-6plus-style!!!.
### Input:
{instruction}
### Output:"""
    elif prompt_mode=="blip_pair" or prompt_mode=="inter":
        return f"""Below is a text imitation task. You will be given a text description and asked to rewrite it in a different style.
### Input:
{instruction}
### Output:"""
    elif prompt_mode=="CoT" or prompt_mode=="CoT_direct":
        return f"""Below is a text rewriting task, finish it step by step. You will be given a prompt for image generation. Your objective is to rewrite it to align with the AI generation model called !!!stable-diffusion!!!
### Input:
{instruction}
### Output:
Let's think step by step.
"""

def preprocess_instruction(instruction, prompt_mode, tokenizer):
    prompt = generate_prompt_hhl(instruction, prompt_mode)
    print(prompt)

    return prompt

def evaluate_batch(instructions, tokenizer, model, input=None, **kwargs):
    args = kwargs.get("args")
    batch_prompts = [preprocess_instruction(instruction, args.prompt_mode, tokenizer) for instruction in instructions]
    inputs = tokenizer.batch_encode_plus(batch_prompts, return_tensors="pt", padding=True)
    for t in inputs:
        if torch.is_tensor(inputs[t]):
            inputs[t] = inputs[t].to('cuda')
    # batch_input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.9,
        num_beams=1,
        **kwargs,
    )
    generation_outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=1500,
    )
    batch_outputs = tokenizer.batch_decode(generation_outputs)
    
    if args.prompt_mode == "alpaca":
        return [output.split("### Response:\n")[1].strip() for output in batch_outputs]
    elif args.prompt_mode not in ['CoT','CoT_direct']:
        return [output.split("### Output:")[1].strip() for output in batch_outputs]
    else:
        return [output.split("### Output:\nLet's think step by step.\n")[1].strip() for output in batch_outputs]


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
    
def data_generator(data_file):
    with open(data_file, 'r') as file:
        data = json.load(file)
    for item in data:
        yield item['caption']


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d','--dir',dest="dir_of_hf_w", type=str, help='dir folder of hf weights, e.g., xxx.bin')
    parser.add_argument('--out-to-txt',dest="out_to_txt", action='store_true', help='store output text to out_generation.txt')
    parser.add_argument('--load-in-8bit',dest="load_in_8bit", action='store_true', help='')
    parser.add_argument('-i','--interact',dest="interact", action='store_true', help='')
    parser.add_argument("--coco",'--coco_caption',dest="coco_caption", action='store_true', help='generate refined captions for coco, save as json')
    parser.add_argument("--coco_file",type=str,default="./data/COCO/coco_captions_val_comma.json")
    parser.add_argument("--db",'--db_caption',dest="db_caption", action='store_true', help='generate refined captions for db, save as json')
    parser.add_argument('--prompt-mode',dest="prompt_mode",type=str, help='[blip_pair, continue,inter,CoT]')
    parser.add_argument('--batch_size',dest="batch_size",type=int, default=4, help='batch_size=12')

    args = parser.parse_args()

    # building the model and tokenizer
    # tokenizer = LLaMATokenizer.from_pretrained(args.dir_of_hf_w)
    # model = LLaMAForCausalLM.from_pretrained(
    #     args.dir_of_hf_w,
    #     load_in_8bit=args.load_in_8bit, # by Kris: True may save memory (16GB to 10GB), but slower
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    # model = PeftModel.from_pretrained(
    #     model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16
    # )
    # model.eval()
    
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
        # new_caption = []
        # # with open("./coco_captions_val.json","r") as fp:
        # with open(args.coco_file,"r") as fp:
            # coco_json = json.load(fp)
        # for idx,it in enumerate(tqdm(coco_json)):
        #     cap =  it['caption']
        #     pred = evaluate(cap, tokenizer, model,args=args)
        #     pred = pred.replace('</s>','')
        #     it["refined"] = pred
        #     new_caption += [it]
        #     print(it)
        #     # print(idx)
        #     # print(it)
        # # with open(f"./coco_captions_val_pairs_{args.prompt_mode}.json","w") as fp:
        # with open(args.coco_file[:-5]+f"_{args.prompt_mode}.json","w") as fp:
        #     json.dump(new_caption,fp,indent=4)

        results = []
        # with open(args.coco_file,'r') as fp:
        #     coco_txt = fp.readlines()
        # n_have = 0
        import os

        output_file = args.coco_file[:-4]+f'_CoT_{formatted_timestamp}.txt'
        # if os.path.exists(output_file):
        #     with open(output_file,'r') as fp:
        #         data = fp.readlines()
        #         for _ in data: 
        #             if _.strip() != "":
        #                 results.append(_.strip()) 
        #         n_have = len(results)
        #         print(n_have)
        #         print(results[:5])
        # if n_have==50:
        #     return

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

        # for idx,txt in enumerate(tqdm(coco_txt)):
        #     if idx<n_have:
        #         continue
        #     pred = ""
        #     while pred.find("!!!LAION-6plus-style!!! ")==-1:
        #         pred = evaluate(txt,tokenizer,model,args=args)
        #         pred = pred.replace('</s>','')
        #         pred = pred.replace('\n',' ')
        #         print(idx, txt)
        #         print(pred)
        #         txt += " pad"
        #     results.append(pred.split("!!!LAION-6plus-style!!! ")[1].strip())
        #     with open(output_file,'w') as fp:
        #         for r in results:
        #             print(r,file=fp)


        # with open(args.coco_file,"r") as fp:
        #     coco_json = json.load(fp)
        # for idx,it in enumerate(tqdm(coco_json)):
       

        # with open(output_file, 'w') as wf:
        #     with tqdm(total=1000) as pbar:
        #         for idx, cap in enumerate(data_generator(args.coco_file)):
        #             # cap =  it['caption']
        #             pred = ""
        #             repeat_count = 0
        #             while pred.find("!!!LAION-6plus-style!!!")==-1:
        #                 pred = evaluate(cap,tokenizer,model,args=args)
        #                 pred = pred.replace('</s>','')
        #                 pred = pred.replace('\n',' ')
        #                 print(idx, cap)
        #                 print(pred)
        #                 repeat_count+=1
        #                 cap += " pad"
        #             print("repeat = ", repeat_count)
        #             # results.append(pred.split("!!!LAION-6plus-style!!!")[1].strip())
        #             result = pred.split("!!!LAION-6plus-style!!!")[1].strip()
        #             print("result = ", result)
        #             print(pred)
        #             wf.write(result + '\n')
        #             wf.flush()
        #             pbar.update(1)
        # txt hhl
        with open(args.coco_file,'r') as fp:
            coco_txt = fp.readlines()

        def batch_split(prompts, batch_num):
            batch_prompts = []
            mini_batch = []
            for prompt in prompts:
                mini_batch.append(prompt)
                if len(mini_batch) == batch_num:
                    batch_prompts.append(mini_batch)
                    mini_batch = []
            if len(mini_batch) != 0:
                batch_prompts.append(mini_batch)
            return batch_prompts
        # coco_txt_batch = batch_split(coco_txt, batch_num=8)

        progress_bar = tqdm(total=len(coco_txt))
        batch_size = args.batch_size

        for i in range(0, len(coco_txt), batch_size):
            batch = coco_txt[i:i + batch_size]
            
            # pred = evaluate(cap,tokenizer,model,args=args)
            prompts = []
            for instruction in batch:
                prompt = generate_prompt(instruction, args, input)
                prompts += [prompt]

            # inputs = tokenizer(prompts, return_tensors="pt", padding=True)
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

            # import ipdb;ipdb.set_trace()
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
            # import ipdb;ipdb.set_trace()

            progress_bar.update(len(batch))
        return

        with open(output_file, 'w') as wf:
            for idx,cap in enumerate(tqdm(coco_txt)):
                # cap =  it['caption']
                pred = ""
                repeat_count = 0
                while pred.find("!!!LAION-6plus-style!!!")==-1:
                    pred = evaluate(cap,tokenizer,model,args=args)
                    pred = pred.replace('</s>','')
                    pred = pred.replace('\n',' ')
                    print(idx, cap)
                    print(pred)
                    repeat_count+=1
                    cap += " pad"
                print("repeat = ", repeat_count)
                # results.append(pred.split("!!!LAION-6plus-style!!!")[1].strip())
                result = pred.split("!!!LAION-6plus-style!!!")[1].strip()
                print("result = ", result)
                print(pred)
                wf.write(result + '\n')
                wf.flush()
        # def batch_deal_pred(pred):
        #     pred = pred.replace('</s>','')
        #     pred = pred.replace('\n',' ')
        #     print(pred)
        #     result = pred.split("!!!LAION-6plus-style!!!")[1].strip()
        #     return result
        # # batch
        # with open(output_file, 'w') as wf:
        #     for idx,cap in enumerate(tqdm(coco_txt_batch)):
        #         # cap =  it['caption']q
        #         pred = ""
        #         # while pred.find("!!!LAION-6plus-style!!!")==-1:
        #         preds = evaluate_batch(cap,tokenizer,model,args=args)
        #         preds = pred.replace('</s>','')
        #         pred = pred.replace('\n',' ')
        #         results = [batch_deal_pred(pred) for pred in preds]
        #         #results.append(pred.split("!!!LAION-6plus-style!!!")[1].strip())
        #         for result in results: 
        #             print("result = ", result)
        #             print(pred)
        #             wf.write(result + '\n')
        #             wf.flush()
        #     with open(output_file,'w') as fp:
        #         for r in results:
        #             print(r,file=fp)

        # with open(output_file,'w') as fp:
        #     for r in results:
        #         print(r,file=fp)


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