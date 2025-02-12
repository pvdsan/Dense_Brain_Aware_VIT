import torch
import torch.nn as nn

class RegionFeatureExtractor2(nn.Module):
    """
    Region-based CNN feature extractor #2.

    This network also processes 3D volumes (e.g., 41x41x41) for each region
    in a batch, then returns a feature vector of size `num_classes`.

    Args:
        num_classes (int): Dimension of the final feature vector per region.
        dropout_rate (float): Dropout probability for dropout layers (if used).
    """
    def __init__(self, num_classes=10, dropout_rate=0.3, normalization='none'):
        super(RegionFeatureExtractor2, self).__init__()
        
        norm_type = normalization.lower()
        if norm_type == 'layernorm':
            # For LayerNorm, the normalized shape must match the output shape from Conv3d.
            norm_layer1 = nn.LayerNorm([32, 39, 39, 39])
            norm_layer2 = nn.LayerNorm([64, 19, 19, 19])
        elif norm_type == 'batchnorm':
            norm_layer1 = nn.BatchNorm3d(32)
            norm_layer2 = nn.BatchNorm3d(64)
        elif norm_type == 'none':
            norm_layer1 = nn.Identity()
            norm_layer2 = nn.Identity()
        else:
            raise ValueError("Normalization type must be one of: 'layernorm', 'batchnorm', or 'none'.")
        # Input shape: (batch_size * num_regions, 1, 41, 41, 41)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False), # 41-> 39
            norm_layer1,
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2), # 39 -> 19
            nn.Dropout3d(p=dropout_rate),

            # Block 2
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False), # 19-> 17
            norm_layer2,
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3), # 17 -> 5

            nn.Flatten(),
            nn.Linear(64 * 5 * 5 * 5, num_classes)  # => 64*5*5*5 = 8000
        )

    def forward(self, x):
        """
        Forward pass:
         x shape: [batch_size, num_regions, 41, 41, 41]
        Returns:
         out shape: [batch_size, num_regions, num_classes]
        """
        batch_size = x.shape[0]
        num_regions = x.shape[1]

        # Merge batch & region dims for CNN
        # => (batch_size * num_regions, 1, 41, 41, 41)
        x = x.view(batch_size * num_regions, 1, 41, 41, 41)

        x = self.features(x)

        # Reshape back: (batch_size, num_regions, num_classes)
        x = x.view(batch_size, num_regions, -1)
        return x



