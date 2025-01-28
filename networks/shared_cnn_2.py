import torch.nn as nn

class RegionFeatureExtractor2(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(RegionFeatureExtractor2, self).__init__()
        
        # Input (1, 41, 41, 41)
        self.features = nn.Sequential(
            # First block
            nn.Conv3d(1, 32, kernel_size=3, padding=0, stride=1, bias=False),  # (32, 39, 39, 39)
#            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (32, 19, 19, 19)
#            nn.Dropout3d(p=dropout_rate),

            # Second block
            nn.Conv3d(32, 64, kernel_size=3, padding=0, stride=1, bias=False),  # (64, 17, 17, 17)
#            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (64, 8, 8, 8)
#            nn.Dropout3d(p=dropout_rate),

            # Third block
            nn.Conv3d(64, 128, kernel_size=3, padding=0, stride=1, bias=False),  # (128, 6, 6, 6)
#            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # (128, 3, 3, 3)
#            nn.Dropout3d(p=dropout_rate),

            # Fourth block
            nn.Conv3d(128, 256, kernel_size=3, padding=0, stride=1, bias=False),  # (256, 1, 1, 1)
#            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
#            nn.Dropout3d(p=dropout_rate),

            # Final output layer
            nn.Conv3d(256, num_classes, kernel_size=1, padding=0, bias=True)  # (num_classes, 1, 1, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_regions = x.shape[1]
        # Reshape to (batch_size * num_regions, 1, 41, 41, 41)
        x = x.view(batch_size * num_regions, 1, 41, 41, 41)
        x = self.features(x)
        # Reshape back to (batch_size, num_regions, num_classes)
        x = x.view(batch_size, num_regions, -1)
        return x
