import torch
import torch.nn as nn
from networks.RegionFeatureExtractor1 import RegionFeatureExtractor1
from networks.RegionFeatureExtractor2 import RegionFeatureExtractor2
from networks.regressor import Regressor


class DualCNNRegressor(nn.Module):
    """
    A dual-CNN model that uses two region-specific feature extractors (cnn1, cnn2),
    and then combines their outputs to feed into a final regressor.
    
    Constructor Arguments:
    ----------------------
    dropout_rate     : float, dropout probability for the CNNs
    num_classes      : int,   output dimension for each CNN
    use_pos_encoding : bool,  if True, incorporate positional encodings e1 & e2
    use_attention    : bool,  if True, include attention-based logic
    """
    def __init__(
        self,
        dropout_rate=0.3,
        num_classes=10,
        use_pos_encoding=False,
        use_attention=False,
        normalization='none',
    ):
        super(DualCNNRegressor, self).__init__()
        
        # Instantiate the two CNN-based region feature extractors
        # Adjust "num_classes" or "dropout_rate" as needed
        self.cnn1 = RegionFeatureExtractor1(num_classes=num_classes, dropout_rate=dropout_rate, normalization=normalization)
        self.cnn2 = RegionFeatureExtractor2(num_classes=num_classes, dropout_rate=dropout_rate, normalization=normalization)
        
        # If each CNN outputs <num_regions, num_classes>,
        # total regions = 305 + 27 = 332, each of dimension `num_classes`.
        # => Combined flattened size = 332 * num_classes.
        if use_pos_encoding:
            # If using positional encodings, adjust the input size accordingly.
            # Assuming e1 and e2 have a fixed dimension of 3 (e.g., x, y, z coordinates).
            self.regressor = Regressor(input_size=332 * (num_classes + 3))
        else:
            self.regressor = Regressor(input_size=332 * num_classes)
        
        # Store flags
        self.use_pos_encoding = use_pos_encoding
        self.use_attention = use_attention

    def forward(self, x1, x2, e1=None, e2=None):
        """
        Forward pass where:
          x1 -> input tensor for the first CNN (shape: [batch_size, 305, 41, 41, 41])
          x2 -> input tensor for the second CNN (shape: [batch_size,  27, 41, 41, 41])
          e1 -> optional positional encodings for x1 (could be shape [batch_size, 305, 3], for instance)
          e2 -> optional positional encodings for x2 (could be shape [batch_size,  27, 3])
        
        Returns:
          output -> regression output (shape: [batch_size, 1])
        """
        batch_size = x1.size(0)
        
        # Extract region-based features
        out1 = self.cnn1(x1)  # shape: [batch_size, 305, num_classes]
        out2 = self.cnn2(x2)  # shape: [batch_size,  27, num_classes]

        # Optionally incorporate positional encodings
        if self.use_pos_encoding and e1 is not None and e2 is not None:
            # For example, just concatenate them along the feature dimension:
            out1 = torch.cat((out1, e1), dim=2)  # shape: [batch_size, 305, num_classes + e1_dim]
            out2 = torch.cat((out2, e2), dim=2)  # shape: [batch_size,  27, num_classes + e2_dim]


        # Todo: Implement attention-based logic if required
        if self.use_attention:
            # e.g., self_attention(out1), or cross_attention(out1, out2), etc.
            pass  # You'd implement attention-based transformations here.

        # Combine the two sets of features along the region dimension
        combined = torch.cat((out1, out2), dim=1)  # shape: [batch_size, 332, feature_dim]
        
        # Flatten to feed into the regressor
        combined_flat = combined.view(batch_size, -1)  # shape: [batch_size, 332 * feature_dim]
        
        # Final regression output
        output = self.regressor(combined_flat)
        return output









