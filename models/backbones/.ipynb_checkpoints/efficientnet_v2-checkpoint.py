import torch
import torch.nn as nn
import 

ENV2_ARCHS = {
    'efficientnet_v2_s': 1280,
    'efficientnet_v2_m': 1280,
    'efficientnet_v2_l': 1280,
}

class EfficientNet_V2(nn.Module):
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()  # NOTE - 调用nn.Module的初始化方法

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = get_pretrained_torchvision_model(
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token


    def forward(self, x: torch.Tensor):
        def forward(self, x):
        x = self.model.forward_features(x)
        return x


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')