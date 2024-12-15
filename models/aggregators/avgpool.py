from torch import nn

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),   #REVIEW: 这里是GeM吗？自适应池化
            Flatten(),
        )
    def forward(self, x):
        x = self.pooling(x)
        return x

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """
        #NOTE: 
        Forward pass of the Flatten module.
        forward: This is the forward pass method. It checks if the input tensor x has a shape where the second and third dimensions (height and width) are equal and equal to 1. If the assertion fails, it raises an error. Otherwise, it returns the flattened version of the tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, channels).

        Raises:
            AssertionError: If the height and width dimensions of the input tensor are not equal to 1.
        """
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:,:,0,0]