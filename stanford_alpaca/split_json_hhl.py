import json  
import os
import math

input_file = '/data/liuyijiang/gpt-prompt/stanford_alpaca/coco_captions_val.json'

output_path = './coco_data_hhl/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 读取原始JSON文件  
with open(input_file, 'r') as f:  
    data = json.load(f)  
  
# # 计算每个文件应该包含的数据数量  
# data_per_file = len(data) // 16  

# print('total len:', len(data))
# print("each split data len: ", data_per_file)

# 计算每个子文件的数据数量
num_files = 8
data_per_file = math.ceil(len(data) / num_files)

# 分割数据并保存为16个文件
for i in range(num_files):
    start_idx = i * data_per_file
    end_idx = min((i + 1) * data_per_file, len(data))
    sub_data = data[start_idx:end_idx]


    sub_file_name = os.path.join(output_path, f'coco_captions_val_{i + 1}_of_{num_files}.json')

    # 保存分割后的数据到子文件
    with open(sub_file_name, 'w') as sub_file:
        json.dump(sub_data, sub_file, indent=4)


print("finish!")