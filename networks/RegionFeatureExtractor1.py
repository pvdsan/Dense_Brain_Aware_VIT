import torch
import torch.nn as nn

class RegionFeatureExtractor1(nn.Module):
    """
    Region-based CNN feature extractor #1.
    
    Processes 3D volumes (e.g., 25x25x25) per region and returns a feature vector.
    
    Args:
        num_classes (int): Dimension of the final feature vector per region.
        dropout_rate (float): Dropout probability.
        use_layernorm (bool): If True, use LayerNorm instead of BatchNorm.
    """
    def __init__(self, num_classes=10, dropout_rate=0.3, normalization='none'):
        super(RegionFeatureExtractor1, self).__init__()
        

        norm_type = normalization.lower()
        if norm_type == 'layernorm':
            # For LayerNorm, the normalized shape must match the output shape from Conv3d.
            norm_layer1 = nn.LayerNorm([32, 23, 23, 23])
            norm_layer2 = nn.LayerNorm([64, 11, 11, 11])
            norm_layer3 = nn.LayerNorm([128, 2, 2, 2]) 
        elif norm_type == 'batchnorm':
            norm_layer1 = nn.BatchNorm3d(32)
            norm_layer2 = nn.BatchNorm3d(64)
            norm_layer3 = nn.BatchNorm3d(128)
        elif norm_type == 'none':
            norm_layer1 = nn.Identity()
            norm_layer2 = nn.Identity()
            norm_layer3 = nn.Identity()
        else:
            raise ValueError("Normalization type must be one of: 'layernorm', 'batchnorm', or 'none'.")
          
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False), # 25-> 23
            norm_layer1, 
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),  # Reduces dims: 23 -> 11
            nn.Dropout3d(p=dropout_rate),
            
            # Block 2
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),  # 11-> 9
            norm_layer2,
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),  # Reduces dims: 9 -> 4


            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, num_classes)
        )
    
    def forward(self, x):
        # x is of shape (batch_size, num_regions, 25, 25, 25)
        batch_size = x.shape[0]
        num_regions = x.shape[1]
        # Combine batch and region dims: (batch_size*num_regions, 1, 25, 25, 25)
        x = x.view(batch_size * num_regions, 1, 25, 25, 25)
        x = self.features(x)
        # Reshape back to (batch_size, num_regions, num_classes)
        x = x.view(batch_size, num_regions, -1)
        return x
