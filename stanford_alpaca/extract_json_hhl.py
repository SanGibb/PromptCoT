import json
import random


input_file = '/data/liuyijiang/gpt-prompt/stanford_alpaca/coco_captions_val.json'
out_file = './coco_caption_random_1k.json'

# 读取原始JSON文件
with open(input_file, 'r') as f:
    data = json.load(f)

# 随机采样1000条数据
sampled_data = random.sample(data, 1000)

# 将采样的数据保存到新的JSON文件
with open(out_file, 'w') as f:
    json.dump(sampled_data, f, indent=4)

print("已保存随机采样的数据到新文件 ", out_file)