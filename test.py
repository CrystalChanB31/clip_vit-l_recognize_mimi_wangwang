from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
import os
#加载模型和处理器
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name,use_fast=True)
text_labels=["a photo of a cat", "a photo of a dog"]
#准备图片
image_directory = "./images/"
try:
    all_files = os.listdir(image_directory)
except:
    print(f"无法访问目录: {image_directory}")
    exit(1)
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
if not image_files:
    print(f"目录中没有找到图片文件: {image_directory}")
    exit(1)
print(f"在目录中找到 {len(image_files)} 张图片。")
print("="*40)
#遍历每张图片进行处理
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    try:
        # 加载图片
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"无法打开图片文件 {image_file}: {e}")
        print("---")
        continue
    print(f"正在处理图片: {image_file}")
    # 准备文本标签并进行推理
    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # 这是图片与文本的相似度分数
    probs = logits_per_image.softmax(dim=1) # 我们可以使用 Softmax 来获取标签的概率分布
    best_match_index = probs.argmax(dim=1).item()
    best_match_label = text_labels[best_match_index]
    best_match_prob = probs[0, best_match_index].item() # 获取该标签的概率值
    #打印结果
    print(f"图片文件: {image_file}")
    print(f"分析的标签: {text_labels}")
    print(f"结果: 这张图片最匹配的标签是 -> '{best_match_label}'")
    print(f"匹配概率: {best_match_prob * 100:.2f}%")
    print("---")
print("="*40)
print("---所有图片处理完成。---")
