import os,json
import fire
from numpy import random
from tqdm import tqdm
from fastparquet import ParquetFile

def main(
        mode,
        indir, 
        outdir='./', 
        min_similarity=0.33,
        min_text_len=150,
        ):
    random.seed(0)

    if mode=="raw":
        raw(indir,outdir,min_similarity,min_text_len)
    elif mode=="word":
        word(indir,outdir,min_similarity,min_text_len)
    elif mode=="gaussian":
        gaussian(indir,outdir,min_similarity,min_text_len)
    elif mode=="count":
        count(indir,min_similarity,min_text_len)
    elif mode=="dist":
        dist(indir,min_similarity,min_text_len)
    elif mode=="get_len_cap":
        get_len_cap(indir,outdir,min_similarity)
    elif mode[:11]=="get_prompt_":
        std_p = mode[11:].split('.')
        if len(std_p)!=2:
            std_p=1
        elif not std_p[0].isdigit() or not std_p[1].isdigit():
            std_p=1
        else:
            std_p = float(mode[11:])
        get_prompt(indir,std_p)
    elif mode[:12]=="get_odd_img_":
        std_scale = mode[12:]
        if std_scale.isdigit():
            std_scale = int(std_scale)
        else:
            std_scale = 3
        get_odd_img(indir,outdir,min_similarity,std_scale)
    else:
        print("Wrong Mode")

def raw(indir,outdir,min_similarity,min_text_len):
    content_list = []
    pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
    for pp in tqdm(pp_list):
        path = os.path.join(indir, pp)
        pf = ParquetFile(path)
        dF = pf.to_pandas()
        dF = dF[dF.similarity<1.]
        dF = dF[dF.similarity>min_similarity]
        dF = dF[dF.TEXT.str.len()>min_text_len]
        for idx, row in dF.iterrows():
            text_dict = {
                        'input': row.TEXT[:len(row.TEXT)//2],
                        'output': row.TEXT,
                        }
            content_list.append(text_dict)

    with open(os.path.join(outdir,f"tcontinue_data_raw.json"),"w") as fp:
        json.dump(content_list, fp, indent=4)

def word(indir,outdir,min_similarity,min_text_len):
    content_list = []
    pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
    for pp in tqdm(pp_list):
        path = os.path.join(indir, pp)
        pf = ParquetFile(path)
        dF = pf.to_pandas()
        dF = dF[dF.similarity<1.]
        dF = dF[dF.similarity>min_similarity]
        dF = dF[dF.TEXT.str.len()>min_text_len]
        for idx, row in dF.iterrows():
            text = row.TEXT.strip().split(' ')
            text_dict = {
                        'input': text[:len(text)//2].join(' '),
                        'output': row.TEXT,
                        }
            content_list.append(text_dict)

    with open(os.path.join(outdir,f"tcontinue_data_word.json"),"w") as fp:
        json.dump(content_list, fp, indent=4)

def gaussian(indir,outdir,min_similarity,min_text_len):
    content_list = []
    pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
    for pp in tqdm(pp_list):
        path = os.path.join(indir, pp)
        pf = ParquetFile(path)
        dF = pf.to_pandas()
        dF = dF[dF.similarity<1.]
        dF = dF[dF.similarity>min_similarity]
        dF = dF[dF.TEXT.str.len()>min_text_len]
        for idx, row in dF.iterrows():
            text = row.TEXT.strip().split(' ')
            p = 0
            while p<0.3 or p>0.7:
                p = random.normal(0.5,1)
            text_dict = {
                        'input': ' '.join(text[:int(len(text)*p)]),
                        'output': row.TEXT,
                        }
            # if random.randint(20)>=4:
            #     continue
            content_list.append(text_dict)

    with open(os.path.join(outdir,f"tcontinue_data_gaussian.json"),"w") as fp:
        json.dump(content_list, fp, indent=4)

def count(indir,min_similarity,min_text_len):
    pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
    ori,sim,txt = 0,0,0
    for pp in tqdm(pp_list):
        path = os.path.join(indir, pp)
        pf = ParquetFile(path)
        dF = pf.to_pandas()
        ori += len(dF)
        dF = dF[dF.similarity<1.]
        dF = dF[dF.similarity>min_similarity]
        sim += len(dF)
        dF = dF[dF.TEXT.str.len()>min_text_len]
        txt += len(dF)
    print("origin caption:",ori)
    print("refined by similarity:",sim)
    print("refined by text len:",txt)

def dist(indir,min_similarity,min_text_len):
    print("start dist")
    dist_data=[]
    pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
    ori,sim,txt = 0,0,0
    print("dist_data", dist_data)
    for pp in tqdm(pp_list):
        path = os.path.join(indir, pp)
        pf = ParquetFile(path)
        dF = pf.to_pandas()
        for index, row in dF.iterrows():
            dist_data.append(len(row['TEXT']))
    print("dist_data", dist_data)
    import matplotlib.pyplot as plt
    plt.hist(data, color='lightgreen', ec='black', bins=15)
    plt.savefig('dist.png')

def get_prompt(infile,std_p):
    with open(f'./prompts_r_{std_p}.txt','w') as wfr:
        with open(f'./prompts_c_{std_p}.txt','w') as wcr:
            with open(infile) as f:
                data = json.load(f)
                for pair in data:
                    p = random.random()
                    if p<std_p:
                        print(pair['caption'],file=wcr)
                        print(pair['refined'],file=wfr)

def get_odd_img(indir,outdir,min_similarity,std_scale):
    with open(os.path.join(outdir,f"odd_img_{std_scale}.txt"),"w") as fp:
        pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
        for pp in tqdm(pp_list):
            path = os.path.join(indir, pp)
            pf = ParquetFile(path)
            dF = pf.to_pandas()
            dF = dF[dF.similarity<1.]
            dF = dF[dF.similarity>min_similarity]
            for idx, row in dF.iterrows():
                scale = row.HEIGHT/row.WIDTH
                if scale<std_scale and scale>1/std_scale:
                    continue
                print (row.TEXT,file=fp)

def get_len_cap(indir,outdir,min_similarity):
    with open(os.path.join(outdir,f"len_cap_long_21k.txt"),"w") as fpl:
        with open(os.path.join(outdir,f"len_cap_mid_21k.txt"),"w") as fpm:
            with open(os.path.join(outdir,f"len_cap_short_21k.txt"),"w") as fps:
                pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
                cap={'long':[],'mid':[],'short':[]}
                for pp in tqdm(pp_list):
                    path = os.path.join(indir, pp)
                    pf = ParquetFile(path)
                    dF = pf.to_pandas()
                    dF = dF[dF.similarity<1.]
                    dF = dF[dF.similarity>min_similarity]
                    
                    dF["sort_score"] = dF["similarity"]*5+dF["aesthetic_SCORE"]
                    dF = dF.sort_values(by=["sort_score"],ascending=[False])

                    dF_long = dF[dF.TEXT.str.len()>150]
                    dF_mid = dF[dF.TEXT.str.len()>90]
                    dF_mid = dF_mid[dF_mid.TEXT.str.len()<110]
                    dF_short = dF[dF.TEXT.str.len()<40]
                    cnt = 0
                    for idx, row in dF_long.iterrows():
                        cnt+=1
                        cap['long'].append(row.TEXT)
                        print(row.sort_score)
                        if cnt == 30000:
                            break
                    cnt = 0
                    for idx, row in dF_mid.iterrows():
                        cnt+=1
                        cap['mid'].append(row.TEXT)
                        if cnt == 30000:
                            break
                    cnt = 0
                    for idx, row in dF_short.iterrows():
                        cnt+=1
                        cap['short'].append(row.TEXT)
                        if cnt == 30000:
                            break
                # cap['long'] = random.choice(cap['long'],20000,replace=False)
                # cap['mid'] = random.choice(cap['mid'],20000,replace=False)
                # cap['short'] = random.choice(cap['short'],20000,replace=False)
                print(len(cap['long']),len(cap['mid']),len(cap['short']))
                for i in range(210000):
                    print(cap['long'][i],file=fpl)
                    print(cap['mid'][i],file=fpm)
                    print(cap['short'][i],file=fps)

if __name__=="__main__":
    
    pp_list = ['./test/2B-en-4.5_1.parquet']
    pp_list += ['../LAION/laion2B-en-aesthetic/'+pp for pp in os.listdir('../LAION/laion2B-en-aesthetic') if pp.endswith('.parquet')]

    dist_data=[]
    # pp_list = [pp for pp in os.listdir(indir) if pp.endswith('.parquet')]
    # ori,sim,txt = 0,0,0
    print("dist_data", dist_data)
    for pp in tqdm(pp_list):
        # path = os.path.join(indir, pp)
        pf = ParquetFile(pp)
        dF = pf.to_pandas()
        for index, row in dF.iterrows():
            # print("row", row)
            if row['TEXT']:
                dist_data.append(len(row['TEXT']))
    print("dist_data", dist_data)
    import json
    r2_object = json.dumps(dist_data, indent=4)
    f2=open("laion_distribution.json", 'w')
    f2.write(r2_object)
    import matplotlib.pyplot as plt
    plt.hist(data, color='lightgreen', ec='black', bins=15)
    plt.savefig('dist.png')


    # prompts = [[],[],[],[],[]]
    # for pp in pp_list:
    #     s = 0
    #     for p in prompts:
    #         s += len(p)
    #     if s==500:
    #         break
    #     pf = ParquetFile(pp)
    #     dF = pf.to_pandas()
    #     dF = dF[dF.similarity<1.]
    #     dF = dF[dF.similarity>0.33]
    #     dF = dF[dF.TEXT.str.len()<150]
    #     dF = dF[dF.TEXT.str.len()>50]

    #     if pp[:2]=='..':
    #         dF = dF[dF.aesthetic>4.0]
    #         dF_4 = dF[dF.aesthetic<5.0]
    #         dF = dF[dF.aesthetic>5.0]
    #         dF_5 = dF[dF.aesthetic<6.0]
    #         dF = dF[dF.aesthetic>6.0]
    #         dF_6 = dF[dF.aesthetic<7.0]
    #         dF = dF[dF.aesthetic>7.0]
    #         dF_7 = dF[dF.aesthetic<8.0]
    #         dF_8 = dF[dF.aesthetic>8.0]
    #     else:
    #         dF = dF[dF.AESTHETIC_SCORE>4.0]
    #         dF_4 = dF[dF.AESTHETIC_SCORE<5.0]
    #         dF = dF[dF.AESTHETIC_SCORE>5.0]
    #         dF_5 = dF[dF.AESTHETIC_SCORE<6.0]
    #         dF = dF[dF.AESTHETIC_SCORE>6.0]
    #         dF_6 = dF[dF.AESTHETIC_SCORE<7.0]
    #         dF = dF[dF.AESTHETIC_SCORE>7.0]
    #         dF_7 = dF[dF.AESTHETIC_SCORE<8.0]
    #         dF_8 = dF[dF.AESTHETIC_SCORE>8.0]

    #     dF_list = [dF_4,dF_5,dF_6,dF_7,dF_8]
    #     for idx,dF in enumerate(dF_list):
    #         print(len(dF),end=" ")
    #         for idy, row in dF.iterrows():
    #             if len(prompts[idx]) == 100:
    #                 break
    #             prompts[idx].append(row)
    
    # for idx,prompt in enumerate(prompts):
    #     with open(f'./test_{idx+4}.txt','w')as fp:
    #         for _ in prompt:
    #             print(_.URL.strip(),file=fp)

    # # for idx,prompt in enumerate(prompts):
    # #     with open(f'./test/{idx+4}.txt','w')as fp:
    # #         for _ in prompt:
    # #             print(_.TEXT.strip(),file=fp)

    # # fire.Fire(main)