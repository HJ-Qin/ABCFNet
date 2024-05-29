from PIL import Image
import os
# 定义原始图像的尺寸和切割参数
original_width = 6000
original_height = 6000
stride = 300
tile_size = 512
num_tiles_per_row = original_width // stride
num_tiles_per_col = original_height // stride
output_folder = "extract"  # 拼接后的大图保存的文件夹名

# 创建一个空的列表，用于存储所有的小图像
tiles = []

# 加载小图像到列表中
for y in range(num_tiles_per_col):
    for x in range(num_tiles_per_row):
        # 计算小图的文件名
        filename = f'top_potsdam_2_10_RGB{y * num_tiles_per_row + x}.png'
        # 构建小图像的完整路径
        filepath = os.path.join("2_10", filename)  # 替换为小图像所在的文件夹路径
        # 打开小图
        tile = Image.open(filepath)
        # 将小图像添加到列表中
        tiles.append(tile)

# 创建一个新的大图像对象
result_image = Image.new('RGB', (original_width, original_height))

# 拼接小图到大图
for y in range(num_tiles_per_col):
    for x in range(num_tiles_per_row):
        # 计算小图在大图中的位置
        x_offset = x * stride
        y_offset = y * stride
        # 获取小图的实际大小
        tile_width, tile_height = tiles[y * num_tiles_per_row + x].size
        # 如果是最右侧和最下侧的小图，则调整偏移量和大小
        if x == num_tiles_per_row - 1:
            x_offset = original_width - tile_width
        if y == num_tiles_per_col - 1:
            y_offset = original_height - tile_height
        # 将小图粘贴到大图中的正确位置
        result_image.paste(tiles[y * num_tiles_per_row + x], (x_offset, y_offset))

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 保存拼接后的大图
result_image.save(os.path.join(output_folder, 'result_image_2_10_IMG.png'))
