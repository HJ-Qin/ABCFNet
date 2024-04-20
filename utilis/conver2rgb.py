import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm

def convert_to_rgb(tensor_image, device):
    mapping = {
        0: [255, 0, 0],
        1: [255, 255, 0],
        2: [0, 255, 0],
        3: [0, 255, 255],
        4: [0, 0, 255],
        5: [255, 255, 255]
    }

    height, width = tensor_image.shape[-2:]
    rgb_image = torch.zeros((3, height, width), dtype=torch.uint8, device=device)

    for i in range(height):
        for j in range(width):
            pixel_value = tensor_image[0, i, j].item()
            rgb_image[0, i, j] = torch.tensor(mapping[pixel_value][0], device=device)
            rgb_image[1, i, j] = torch.tensor(mapping[pixel_value][1], device=device)
            rgb_image[2, i, j] = torch.tensor(mapping[pixel_value][2], device=device)

    return rgb_image

input_folder = "./new"
output_folder = "./potsdamvalRGB"
os.makedirs(output_folder, exist_ok=True)

# 使用tqdm创建进度条
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for file_name in tqdm(os.listdir(input_folder)):
    if file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        input_file = os.path.join(input_folder, file_name)
        output_file = os.path.join(output_folder, file_name)

        image = Image.open(input_file).convert("L")
        tensor_image = torch.unsqueeze(torch.from_numpy(np.array(image)), 0).to(device)
        rgb_image = convert_to_rgb(tensor_image, device)
        rgb_image_np = rgb_image.permute(1, 2, 0).cpu().numpy()

        output_image = Image.fromarray(rgb_image_np)
        output_image.save(output_file)

print("Conversion completed.")
