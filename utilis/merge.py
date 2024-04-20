# from PIL import Image, ImageDraw
# import os
#
# # 指定五个文件夹路径
# folder_paths = [
#     'cam/image_960_RGB', # image
#     'cam/GT_car_shai',     #筛选后的car
#     'cam/baseline/car',   # baseline
#     'cam/baseline_boundary/car', # boundary
#     'cam/baseline_cpfa/car',    # cpfa
#     'cam/baseline_mfa/car',    #mfa
#     'cam/baseline_boundary_bes/car', # boundary + bes
#     'cam/baseline_boundary_cpfa/car', # boundary + cpfa
#     'cam/baseline_cpfa_mfa/car',       # cpfa + mfa
#     'cam/Aours/car',     # ours
# ]
#
#
#
# # folder_paths = [
# #     'model_output_RGB/image_960_RGB',
# #     'model_output_RGB/potsdamvalRGB',
# #     'model_output_RGB/NoBoundary_train_result_RGB',
# #     'model_output_RGB/300epoch-MAasppVBLMANet-train-RGB_results',
# # ]
#
# # folder_paths = [
# #     'contract_2_11/Image',
# #     'contract_2_11/GT2_11',
# #     'contract_2_11/UNet2_11',
# #     'contract_2_11/psp2_11',
# #     'contract_2_11/deeplabv3plus2_11',
# #     'contract_2_11/FPN2_11',
# #     'contract_2_11/PAN2_11',
# #     'contract_2_11/MANet2_11',
# #     'contract_2_11/unetplus2_11',
# #     'contract_2_11/A2FPN2_11',
# #     'contract_2_11/ours',
# # ]
#
#
#
# # 获取所有文件夹中的图片文件名集合
# file_names = []
# for folder_path in folder_paths:
#     file_names.append(set(os.listdir(folder_path)))
#
# # 找到六个文件夹中共同存在的图片文件名
# common_file_names = set.intersection(*file_names)
#
# # 设置间距大小和间距颜色（棕色）
# spacing = 10
# spacing_color = (127, 127, 127)  # 棕色的RGB颜色值
#
# # 图像大小
# image_width = len(folder_paths) * 512 + (len(folder_paths) - 1) * spacing
# image_height = 512
#
# # 遍历共同存在的文件名
# for filename in common_file_names:
#     # 创建一个新的图像，大小为六个图片加上间隔的宽度
#     new_image = Image.new('RGB', (image_width, image_height), spacing_color)
#
#     # 初始化 x 坐标为0
#     x_coordinate = 0
#
#     # 逐一将每个文件夹中的图片粘贴到新图像中，留出间隔
#     for folder_path in folder_paths:
#         image = Image.open(os.path.join(folder_path, filename))
#         new_image.paste(image, (x_coordinate, 0))
#         # 更新 x 坐标位置，加上当前图片的宽度和间距
#         x_coordinate += 512 + spacing
#
#     # 将拼接好的图像保存
#     new_image.save(f'car_cam_out/{filename}.jpg')
#
# print("图片拼接完成")
#
#
#



from PIL import Image, ImageDraw
import os

# 指定五个文件夹路径
folder_paths = [
    'cam/image_960_RGB', # image
    'boundary/GT_veg_Boundary',     #筛选后的car
    'boundary/Baseline/Baseline_veg',   # baseline
    'boundary/Baseline_boundary/Baseline_veg', # boundary
    'boundary/ours/ours_veg',    # cpfa
    # 'cam/baseline_mfa/veg',    #mfa
    # 'cam/baseline_boundary_bes/veg', # boundary + bes
    # 'cam/baseline_boundary_cpfa/veg', # boundary + cpfa
    # 'cam/baseline_cpfa_mfa/veg',       # cpfa + mfa
    # 'cam/Aours/veg',     # ours
]

# 获取所有文件夹中的图片文件名集合
file_names = []
for folder_path in folder_paths:
    # 忽略后缀的文件名集合
    file_names.append({os.path.splitext(filename)[0] for filename in os.listdir(folder_path)})

# 找到六个文件夹中共同存在的图片文件名
common_file_names = set.intersection(*file_names)

# 设置间距大小和间距颜色（棕色）
spacing = 10
spacing_color = (127, 127, 127)  # 棕色的RGB颜色值

# 图像大小
image_width = len(folder_paths) * 512 + (len(folder_paths) - 1) * spacing
image_height = 512

# 遍历共同存在的文件名
for filename in common_file_names:
    # 创建一个新的图像，大小为六个图片加上间隔的宽度
    new_image = Image.new('RGB', (image_width, image_height), spacing_color)

    # 初始化 x 坐标为0
    x_coordinate = 0

    # 逐一将每个文件夹中的图片粘贴到新图像中，留出间隔
    for folder_path in folder_paths:
        full_filename = os.path.join(folder_path, filename + ".jpg")  # 假设所有图片都是以jpg格式存储
        if os.path.exists(full_filename):
            image = Image.open(full_filename)
            new_image.paste(image, (x_coordinate, 0))
        # 更新 x 坐标位置，加上当前图片的宽度和间距
        x_coordinate += 512 + spacing

    # 将拼接好的图像保存
    new_image.save(f'boundary_veg/{filename}.jpg')

print("图片拼接完成")
