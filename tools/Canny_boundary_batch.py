import os
import cv2
import numpy as np
import torch
 # 导入你的语义分割模型

from netss.MAasppVBLMANet_modify_modify import MANet

# 加载训练好的模型和权重


# 定义输入和输出文件夹路径
input_folder = "/Users/qintongxue/PycharmProjects/deep-learning-for-image-processing-master/pytorch_segmentation/deeplabv3-plus-pytorch-main/HRRSI_potsdam_image"
output_folder = "/Users/qintongxue/PycharmProjects/deep-learning-for-image-processing-master/pytorch_segmentation/deeplabv3-plus-pytorch-main/booundary/ours"

model = MANet(num_classes=7)  # 请替换为你的语义分割模型实例化代码
weights_dict = torch.load(
    r"/Users/qintongxue/PycharmProjects/deep-learning-for-image-processing-master/pytorch_segmentation/deeplabv3-plus-pytorch-main/logs/new_ours_Potsdam-epoch200-best_epoch_weights.pth11",
    map_location=torch.device('cpu'))
model.load_state_dict(weights_dict, strict=False)  # 请替换为你的模型权重文件路径

# 确保输出文件夹存在，如果不存在则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 遍历输入文件夹中的所有图像文件
for filename in os.listdir(input_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # 读取图像
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将图像数据转换为 tensor，并添加批次维度
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0)


        # 使用模型进行预测
        with torch.no_grad():
            predictions = model.forward(image_tensor)

        # 提取边界信息
        segmentation_map = np.argmax(predictions.cpu().numpy(), axis=1)
        edges = cv2.Canny(segmentation_map[0].astype(np.uint8), 0, 1)

        # 生成黑白色边界图
        boundary_image = np.zeros_like(image)
        boundary_image[edges != 0] = [255, 255, 255]

        # 构建输出图像路径
        output_path = os.path.join(output_folder, filename)

        # 保存边界图像
        cv2.imwrite(output_path, cv2.cvtColor(boundary_image, cv2.COLOR_RGB2BGR))
        print(f"Processed image saved: {output_path}")

print("All images processed and saved.")
