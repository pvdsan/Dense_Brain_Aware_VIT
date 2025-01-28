import torch.nn as nn

class RegionFeatureExtractor1(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.3):
        super(RegionFeatureExtractor1, self).__init__()
        
        # Input (1, 25, 25, 25)
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=0, bias=False),  # Output: (32, 23, 23, 23)
#            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # Output: (32, 11, 11, 11)
#            nn.Dropout3d(p=dropout_rate),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=0, bias=False),  # Output: (64, 9, 9, 9)
#            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # Output: (64, 4, 4, 4)
#            nn.Dropout3d(p=dropout_rate),
            
            nn.Conv3d(64, 128, kernel_size=2, padding=0, bias=False),  # Output: (128, 3, 3, 3)
#            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),  # Output: (128, 1, 1, 1)
#            nn.Dropout3d(p=dropout_rate),
            
            nn.Conv3d(128, num_classes, kernel_size=1, padding=0, bias  = True)  # Output: (num_classes, 1, 1, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_regions = x.shape[1]
        x = x.view(batch_size * num_regions, 1, 25, 25, 25)
        x = self.features(x)
        x = x.view(batch_size, num_regions, -1)
        return x
