import cv2
import os
import numpy as np

# 定义两个文件夹的路径
folder1_path = 'model_output_RGB_vailingen/GT_output_Vaihingen_RGB'
folder2_path = 'model_output_RGB_vailingen/ours_modify_Vaihingen_result-RGB'
output_path = 'bad_different_RGB_NUM'

# 获取文件夹1中的所有文件名
folder1_files = os.listdir(folder1_path)

# 遍历文件夹1中的文件
for filename in folder1_files:
    # 拼接文件路径
    file1_path = os.path.join(folder1_path, filename)
    file2_path = os.path.join(folder2_path, filename)

    # 读取两个图像
    img1 = cv2.imread(file1_path)
    img2 = cv2.imread(file2_path)

    # 检查图像是否成功读取
    if img1 is not None and img2 is not None:
        # 计算两个图像的差异
        diff = cv2.absdiff(img1, img2)

        # 找到差异像素的位置
        diff_pixels = np.where(diff > 0)

        # 计算差异像素的数量
        num_diff_pixels = len(diff_pixels[0])

        # 将差异像素标记为紫色
        for x, y in zip(diff_pixels[0], diff_pixels[1]):
            diff[x, y] = [255, 0, 255]  # BGR颜色空间，紫色

        # 在图像的右上角添加文本
        text = f'Error Pixels: {num_diff_pixels}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = diff.shape[1] - text_size[0] - 10
        text_y = 30
        cv2.putText(diff, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 保存差异图像
        diff_filename = filename
        diff_path = os.path.join(output_path, diff_filename)
        cv2.imwrite(diff_path, diff)
    else:
        print(f"无法读取文件: {filename}")

print("差异图像生成完成")
