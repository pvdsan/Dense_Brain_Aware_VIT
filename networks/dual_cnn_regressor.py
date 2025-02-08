import torch
import torch.nn as nn
from networks.shared_cnn_1 import RegionFeatureExtractor1
from networks.shared_cnn_2 import RegionFeatureExtractor2
from networks.regressor import Regressor


class RegionFeatureExtractor2(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(RegionFeatureExtractor2, self).__init__()
        
        # Input (1, 41, 41, 41)
        self.features = nn.Sequential(
            # First block
            nn.Conv3d(1, 32, kernel_size=3, padding=0, stride=1, bias=False), 
#            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), 
#            nn.Dropout3d(p=dropout_rate),

            # Second block
            nn.Conv3d(32, 64, kernel_size=3, padding=0, stride=1, bias=False),  
#            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3),

            nn.Flatten(),
            nn.Linear(64 * 5 * 5 * 5, num_classes)  # Fully connected layer

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



class RegionFeatureExtractor2(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(RegionFeatureExtractor2, self).__init__()
        
        # Input (1, 41, 41, 41)
        self.features = nn.Sequential(
            # First block
            nn.Conv3d(1, 32, kernel_size=3, padding=0, stride=1, bias=False), 
#            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2), 
#            nn.Dropout3d(p=dropout_rate),

            # Second block
            nn.Conv3d(32, 64, kernel_size=3, padding=0, stride=1, bias=False),  
#            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3),

            nn.Flatten(),
            nn.Linear(64 * 5 * 5 * 5, num_classes)  # Fully connected layer

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




class Regressor(nn.Module):
    def __init__(self, input_size = 332*10, hidden_size1 = 256, hidden_size2 = 128, output_size = 1):
        super(Regressor, self).__init__()
        
        self.fullyConnectedLayer = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, x):
        return self.fullyConnectedLayer(x)


class DualCNNRegressor(nn.Module):
    def __init__(self):
        super(DualCNNRegressor, self).__init__()
        self.cnn1 = RegionFeatureExtractor1()
        self.cnn2 = RegionFeatureExtractor2()
        self.regressor = Regressor(input_size=332*10)


    def forward(self, x1, x2, e1, e2):
        
        batch_size = x1.size(0)
        out1 = self.cnn1(x1) # btc_size 305, num_features
        out2 = self.cnn2(x2) # btc_size, 27, num_features
        #out1 = torch.cat((out1, e1), dim=2)  # Shape: [batch_size, 305, num_features+3]
        #out2 = torch.cat((out2, e2), dim=2)  # Shape: [batch_size, 27, num_features+3]
        
        # Combine both outputs
        combined = torch.cat((out1, out2), dim=1)  # Shape: [batch_size, 332, num_features+3]
        
        # Flatten before feeding into the regressor
        combined_flat = combined.view(batch_size, -1)  # Shape: [batch_size, 332*num_features+3]
        output = self.regressor(combined_flat)
        return output










