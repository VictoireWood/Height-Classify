# from PIL import Image

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None

# # 读取四张图像
# image1 = Image.open("/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_01-01.tif")
# image2 = Image.open("/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_01-02.tif")
# image3 = Image.open("/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_02-01.tif")
# image4 = Image.open("/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_02-02.tif")

# # 创建一个新的图像，大小为四张图像拼接后的大小
# width = image1.width + image2.width
# height = image1.height + image3.height
# new_image = Image.new('RGB', (width, height))

# # 将四张图像拼接到新的图像上
# new_image.paste(image1, (0, 0))
# new_image.paste(image2, (image1.width, 0))
# new_image.paste(image3, (0, image1.height))
# new_image.paste(image4, (image1.width, image1.height))

# # 保存拼接后的图像
# new_image.save("/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09.tif")


from tifffile import imread, imsave
import numpy as np

# Load the four images
image1 = imread('/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_01-01.tif')
image2 = imread('/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_01-02.tif')
image3 = imread('/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_02-01.tif')
image4 = imread('/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09_02-02.tif')

# Get dimensions of the images
height1, width1, depth = image1.shape
height2, width2, depth = image4.shape

# Create a new image to store the final result
result = np.zeros((height1 + height2, width1 + width2, depth), dtype=image1.dtype)

# Place the four images in the correct positions
result[:height1, :width1] = image1
result[:height1, width1:] = image2
result[height1:, :width1] = image3
result[height1:, width1:] = image4

# Save the result as satellite09.tif
imsave('/root/workspace/ctf53sc7v38s73e0mksg/maps/UAV_VisLoc_dataset/09/satellite09.tif', result)