import torch
from torch.utils.data import Dataset
import os

class MRIDataset(Dataset):
    def __init__(self, precomputed_dir):
        """
        Args:
            precomputed_dir (str): Path to the directory with precomputed tensors.
            device (str): Device to load tensors on ('cuda' or 'cpu').
        """
        self.precomputed_dir = precomputed_dir
        
        # Get all subject directories in the precomputed directory
        self.subject_dirs = [os.path.join(precomputed_dir, d) for d in os.listdir(precomputed_dir) 
                             if os.path.isdir(os.path.join(precomputed_dir, d)) and d.startswith('subject_')]

    def __len__(self):
        return len(self.subject_dirs)
    
    def __getitem__(self, idx):
        """
        Loads precomputed tensors for the given index.

        Args:
            idx (int): Index of the subject.
        
        Returns:
            cnn1_data (torch.Tensor): CNN1 input data.
            cnn2_data (torch.Tensor): CNN2 input data.
            working_memory_score (torch.Tensor): Target value (working memory score).
            encodings_cnn1 (torch.Tensor): Positional encodings for CNN1.
            encodings_cnn2 (torch.Tensor): Positional encodings for CNN2.
        """
        subject_path = self.subject_dirs[idx]
        
        # Load precomputed tensors
        cnn1_data = torch.load(os.path.join(subject_path, 'cnn1.pt'), weights_only=True)
        cnn2_data = torch.load(os.path.join(subject_path, 'cnn2.pt'), weights_only=True)
        working_memory_score = torch.load(os.path.join(subject_path, 'working_memory.pt'), weights_only=True)
        encodings_cnn1 = torch.load(os.path.join(subject_path, 'cnn1_encodings.pt'), weights_only=True)
        encodings_cnn2 = torch.load(os.path.join(subject_path, 'cnn2_encodings.pt'), weights_only=True)
        
        return cnn1_data, cnn2_data, working_memory_score, encodings_cnn1, encodings_cnn2
