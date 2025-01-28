import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.shared_cnn_1 import RegionFeatureExtractor1
from networks.shared_cnn_2 import RegionFeatureExtractor2
from networks.regressor import Regressor


class DualCNNRegressor(nn.Module):
    def __init__(self):
        super(DualCNNRegressor, self).__init__()
        self.cnn1 = RegionFeatureExtractor1()
        self.cnn2 = RegionFeatureExtractor2()
        print("This is dual cnn attention")
        
        # Define attention layers
        self.attention1 = nn.MultiheadAttention(embed_dim=13, num_heads=1, batch_first=True)
        
        # Custom attention layer with reduced V dimension (5)
        self.attention2 = nn.MultiheadAttention(embed_dim=13, num_heads=1, batch_first=True)
        
        # Initialize the regressor with the updated input size
        # Combined sequence length = 274 + 27 = 301
        
        self.regressor = Regressor(input_size=301*13)
        
        # Layer Normalizations
        self.layer_norm1 = nn.LayerNorm(13)
        self.layer_norm2 = nn.LayerNorm(13)

    def forward(self, x1, x2, e1, e2):
        """
        Forward pass for DualCNNRegressor.

        Args:
            x1 (Tensor): Input tensor for the first CNN. Shape: (batch_size, channels, height, width)
            x2 (Tensor): Input tensor for the second CNN. Shape: (batch_size, channels, height, width)
            e1 (Tensor): Additional embeddings for the first CNN output. Shape: (batch_size, 174, embed_dim_e1)
            e2 (Tensor): Additional embeddings for the second CNN output. Shape: (batch_size, 39, embed_dim_e2)

        Returns:
            Tensor: Regression output.
        """
        batch_size = x1.size(0)
        
        # Pass through CNNs
        out1 = self.cnn1(x1)  # Expected Shape: [batch_size, 274, feature_dim1]
        out2 = self.cnn2(x2)  # Expected Shape: [batch_size, 27, feature_dim2]
        
        # Concatenate with additional embeddings e1 and e2
        # Ensure that feature_dim1 + embed_dim_e1 = 13 and feature_dim2 + embed_dim_e2 = 13
        out1 = torch.cat((out1, e1), dim=2)  # Shape: [batch_size, 274, 13]
        out2 = torch.cat((out2, e2), dim=2)  # Shape: [batch_size, 27, 13]
        
        # Combine both outputs
        combined = torch.cat((out1, out2), dim=1)  # Shape: [batch_size, 301, 13]
        
        # Apply first attention layer (self-attention)
        attn_output1, attn_weights1 = self.attention1(combined, combined, combined)  # Shape: [batch_size, 301, 13]
        
        # Add residual connection
        attn_output1 = attn_output1 + combined  # Shape: [batch_size, 301, 13]
        
        # Apply Layer Normalization after residual connection
        attn_output1 = self.layer_norm1(attn_output1)  # Shape: [batch_size, 301, 13]
        
        # Apply second attention layer with V dimension reduced to 5
        attn_output2, attn_weights2 = self.attention2(attn_output1, attn_output1, attn_output1)  # Shape: [batch_size, 301, 5]
        
        # Apply Layer Normalization after second attention
        attn_output2 = self.layer_norm2(attn_output2)  # Shape: [batch_size, 301, 5]
        
        # Flatten before feeding into the regressor
        combined_flat = attn_output2.contiguous().view(batch_size, -1)  # Shape: [batch_size, 301 * 5] = [batch_size, 1065]
        
        # Pass through regressor
        output = self.regressor(combined_flat)  # Shape depends on regressor's output
        
        return output
