import PIL.Image
import torch
from PIL import Image, ImageFile
import numpy as np
def move_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def log_any(tensor: torch.Tensor, base: None|int|float) -> torch.Tensor:
    if base == None:
        return torch.log(tensor)
    elif base == 2 or base == 2.0:
        return torch.log2(tensor)
    elif base == 10 or base == 10.0:
        return torch.log10(tensor)
    elif type(base) == int or type(base) == float:
        base_tensor = torch.full_like(tensor, base)
        return torch.log(tensor) / torch.log(base_tensor)   # NOTE 换底公式
    else:
        raise NotImplementedError

def tensor_fft_2D(tensor: torch.Tensor, log_base: None|int|float):    # NOTE 输入的tensor是经过transform标准化的
    fft = torch.fft.fft2(tensor.float(), dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft)
    fft_abs = torch.abs(fft_shift)
    # fft_log = torch.log(fft_abs + 1)
    fft_log = log_any(fft_abs + 1, log_base)
    # fft_log = fft_log * 10
    return fft_log

def tensor_fft_3D(tensor: torch.Tensor, log_base: None|int|float):
    fft_3D = torch.zeros_like(tensor)
    # tensor_r = tensor[0,:,:]
    # tensor_g = tensor[1,:,:]
    # tensor_b = tensor[2,:,:]
    for i in range(0,3):
        fft_3D[i,:,:] = tensor_fft_2D(tensor[i,:,:], log_base)
    return fft_3D

def to_unit8(tensor_data: torch.Tensor) -> torch.Tensor:
    # # Find the minimum and maximum values in the tensor
    # min_val = torch.min(tensor_data)
    # max_val = torch.max(tensor_data)

    # # Scale the tensor to have values between 0 and 255
    # scaled_tensor = 255 * (tensor_data - min_val) / (max_val - min_val)

    # 不进行缩放，直接用原始数据
    scaled_tensor = tensor_data * 10

    # Convert the tensor back to integers
    scaled_tensor = scaled_tensor.byte()

    return scaled_tensor

def color_channel_fft(channel:PIL.Image.Image):
    # Convert PIL images to numpy arrays
    arr = np.array(channel)
    # Convert numpy arrays to PyTorch tensors
    tensor = torch.tensor(arr)
    # Perform Fourier transform on each channel
    fft = torch.fft.fft2(tensor.float(), dim=(-2, -1))
    fft_shift = torch.fft.fftshift(fft)
    fft_abs = torch.abs(fft_shift)
    fft_log = torch.log(fft_abs + 1)
    scaled = to_unit8(fft_log)
    channel_image = Image.fromarray(scaled.numpy().astype(np.uint8))
    return channel_image


def image_fft(rgb_image:PIL.Image.Image) -> PIL.Image.Image:
    r, g, b = rgb_image.split()

    r_image = color_channel_fft(r)
    g_image = color_channel_fft(g)
    b_image = color_channel_fft(b)

    result_image = Image.merge('RGB', (r_image, g_image, b_image))
    return result_image

def tensor_to_image(tensor:torch.Tensor) -> PIL.Image.Image:

    # 假设 tensor 是一个 PyTorch 张量，形状为 (C, H, W)，值在 [0, 1] 范围内
    # 步骤1: 确保张量值在 [0, 255] 范围内
    tensor = tensor * 255

    # 步骤2: 转换张量的数据类型为 uint8
    tensor = tensor.byte()

    # 步骤3: 将张量转换为 NumPy 数组
    numpy_array = tensor.permute(1, 2, 0).numpy()  # 从 (C, H, W) 转换为 (H, W, C)

    # 步骤4: 创建 PIL 图像
    image = Image.fromarray(numpy_array)

    # 现在 image 是一个 PIL.Image.Image 类型的变量
    return image
