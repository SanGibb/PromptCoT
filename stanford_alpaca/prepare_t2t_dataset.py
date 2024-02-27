import os,json
import fire
import torch
from PIL import Image
import open_clip
from tqdm import tqdm

def main(
        mode,
        indirs, 
        outdir='./', 
        ):
    if mode=="get_prompt":
        get_prompt(indirs,outdir)
        return
    elif mode not in ['inter','blip']:
        print("wrong mode")
        return

    # import ipdb;ipdb.set_trace()
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    tokenizer = open_clip.get_tokenizer('ViT-g-14')
    model.eval()
    model.to("cuda:0")

    if "," in indirs:
        indir_list = [d for d in indirs.split(",")]
    else:
        indir_list = [indirs]
    for indir in indir_list:
        assert os.path.isdir(indir), f"{indir} is not a valid directory"
    
    content_list = []
    # text_dict = {}
    for indir in indir_list:
        for file in tqdm([ff for ff in os.listdir(indir) if ff.endswith("jpg")]):
            if file.endswith('.jpg'):
                base_name = os.path.splitext(file)[0]
                name_txt = base_name + '.txt'
                name_mode_txt = base_name + '_'+mode+'.txt'
                
                if os.path.isfile(os.path.join(indir, name_txt)) and os.path.isfile(os.path.join(indir, name_mode_txt)):
                    with open(os.path.join(indir, name_txt), 'r') as f_txt, open(os.path.join(indir, name_mode_txt), 'r') as f_mode_txt:
                        t_in = f_mode_txt.read()
                        t_out = f_txt.read()
                        dist = text_distance(t_in, t_out, model, tokenizer)
                        # print("[",mode,"]", t_in)
                        # print("[raw]", t_out)
                        # print(f"[Sim] {dist:.2f}")
                        # print("*"*40)
                        text_dict = {
                            'input': t_in,
                            'output': t_out,
                            'similarity': dist,
                        }
                        content_list.append(text_dict)
    
    # Save the content_list to a JSON file
    with open(os.path.join(outdir, f't2t_{mode}_all.json'), 'w') as f:
        json.dump(content_list, f, indent=4)

    for ii in range(3,8):
        with open(os.path.join(outdir,f"t2t_{mode}_l{ii}.json"),"w") as fp:
            cts = [ll for ll in content_list if ll['similarity']>ii/10.]
            json.dump(cts, fp,indent=4)

    total = len(content_list)
    l3 = len([ll for ll in content_list if ll['similarity']>0.3])
    l4 = len([ll for ll in content_list if ll['similarity']>0.4])
    l5 = len([ll for ll in content_list if ll['similarity']>0.5])
    l6 = len([ll for ll in content_list if ll['similarity']>0.6])
    l7 = len([ll for ll in content_list if ll['similarity']>0.7])
    l8 = len([ll for ll in content_list if ll['similarity']>0.8])
    l9 = len([ll for ll in content_list if ll['similarity']>0.9])
    print(f"Total number: {total}")
    print(f"Larger than 0.3: {l3}")
    print(f"Larger than 0.4: {l4}")
    print(f"Larger than 0.5: {l5}")
    print(f"Larger than 0.6: {l6}")
    print(f"Larger than 0.7: {l7}")
    print(f"Larger than 0.8: {l8}")
    print(f"Larger than 0.9: {l9}")

def text_distance(tx1, tx2, model, tokenizer):
    tx = tokenizer([tx1, tx2])
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(tx.cuda())
    text_features /= text_features.norm(dim=-1, keepdim=True)
    dist = text_features[0] @ text_features[1]
    return dist.item()

def get_prompt(infile,outdir):
    with open(os.path.join(outdir,'prompts_ri.txt'),'w') as wf,open(infile) as f:
        data = json.load(f)
        for pair in data:
            print(pair['refined'],file=wf)

if __name__=="__main__":
    with open('./cot_caption_long.txt','r') as fp:
        data = fp.readlines()
    with open('./cot_compare/CoT_caption_step4.txt','w') as fp:
        for _ in data:
            try:
                _=_.split("prompt with more than 20 wordes and less 50 words.")[1]
                print(_.split("#### Step5")[0].strip(),file=fp)
            except:
                print("error",file=fp)

    
    # fire.Fire(main)