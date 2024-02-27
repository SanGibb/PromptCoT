with open('./CoT.txt','r') as fp:
    data = fp.readlines()

head,pairs = 0,[]
for idx,_ in enumerate(data):
    if len(_) <10:
        continue
    
    if _[:10]=="#### Step1" and idx !=0:
        pairs.append((head,idx))
    elif _[:10]=="#### Step5":
        head = idx+1


print(pairs[:10],len(pairs))

prompts = []
for pair in pairs:
    lines = data[pair[0]:pair[1]]
    res = ''
    for line in lines:
        line = line.replace('\n',' ')
        res += line
    prompts.append(res)

with open('./CoT_prompts.txt','w') as fp:
    for p in prompts:
        print(p,file=fp)