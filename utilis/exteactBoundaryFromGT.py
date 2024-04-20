# from PIL import Image
#
#
# def process_image(image_path):
#     # 打开图片
#     img = Image.open(image_path)
#
#     # 将图像转换为灰度图
#     img = img.convert("L")
#
#     # 获取图片的宽度和高度
#     width, height = img.size
#
#     # 获取图片的像素数据
#     pixel_data = img.load()
#
#     # 遍历每个像素
#     for y in range(height):
#         for x in range(width):
#             # 获取当前像素的灰度值
#             pixel_value = pixel_data[x, y]
#
#             # 如果当前像素是黑色（灰度值为0）
#             if pixel_value == 0:
#                 # 将黑色像素变为白色
#                 pixel_data[x, y] = 255
#             else:
#                 # 将非黑色像素变为黑色
#                 pixel_data[x, y] = 0
#
#     # 保存修改后的图片
#     img.save("processed_image.tif")
#
#
# # 调用函数，传入图片路径
# process_image("/Users/qintongxue/PycharmProjects/deep-learning-for-image-processing-master/pytorch_segmentation/deeplabv3-plus-pytorch-main/booundary/image2/top_potsdam_7_13_label_noBoundary.tif")
#

import os
from PIL import Image
from tqdm import tqdm

def process_images_in_folder(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中所有的图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    # 设置进度条
    progress_bar = tqdm(total=len(image_files), desc='Processing Images')

    # 遍历每张图像
    for image_file in image_files:
        # 构建输入图像的路径
        input_image_path = os.path.join(input_folder, image_file)

        # 打开图像
        img = Image.open(input_image_path)

        # 将图像转换为灰度图
        img = img.convert("L")

        # 获取图像的宽度和高度
        width, height = img.size

        # 获取图像的像素数据
        pixel_data = img.load()

        # 遍历每个像素
        for y in range(height):
            for x in range(width):
                # 获取当前像素的灰度值
                pixel_value = pixel_data[x, y]

                # 如果当前像素是黑色（灰度值为0）
                if pixel_value == 0:
                    # 将黑色像素变为白色
                    pixel_data[x, y] = 255
                else:
                    # 将非黑色像素变为黑色
                    pixel_data[x, y] = 0

        # 构建输出图像的路径
        output_image_path = os.path.join(output_folder, image_file)

        # 保存修改后的图像
        img.save(output_image_path)

        # 更新进度条
        progress_bar.update(1)

    # 关闭进度条
    progress_bar.close()

    print(f"所有图像处理完成，结果已保存到路径: {output_folder}")

# 调用函数，传入输入文件夹路径和输出文件夹路径
input_folder = "/Users/qintongxue/PycharmProjects/deep-learning-for-image-processing-master/pytorch_segmentation/deeplabv3-plus-pytorch-main/booundary/image2"
output_folder = "/Users/qintongxue/PycharmProjects/deep-learning-for-image-processing-master/pytorch_segmentation/deeplabv3-plus-pytorch-main/booundary/yes"
process_images_in_folder(input_folder, output_folder)


