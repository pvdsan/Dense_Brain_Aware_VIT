import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import multiprocessing
import logging


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIProcessingModel:
    """Class to handle processing of MRI data."""
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def process_subject(self, subject_id, cnn1_data, cnn2_data, cnn1_encodings, cnn2_encodings, working_memory):
        subject_dir = os.path.join(self.output_dir, f"subject_{subject_id}")
        os.makedirs(subject_dir, exist_ok=True)
        
        
        # Move tensors to CPU before saving
        cnn1_data = cnn1_data.cpu()
        cnn2_data = cnn2_data.cpu()
        cnn1_encodings = cnn1_encodings.cpu()
        cnn2_encodings = cnn2_encodings.cpu()
        working_memory = working_memory.cpu()

        # Save tensors
        torch.save(cnn1_data, os.path.join(subject_dir, "cnn1.pt"))
        torch.save(cnn2_data, os.path.join(subject_dir, "cnn2.pt"))
        torch.save(cnn1_encodings, os.path.join(subject_dir, "cnn1_encodings.pt"))
        torch.save(cnn2_encodings, os.path.join(subject_dir, "cnn2_encodings.pt"))
        torch.save(working_memory, os.path.join(subject_dir, "working_memory.pt"))

        logger.info(f"Processed subject {subject_id}")


class MRIDataset(Dataset):
    """Custom Dataset for loading MRI images and extracting regions."""
    def __init__(self, subjects_df, small_masks_df, large_masks_df, device):
        self.subjects = subjects_df
        self.small_masks = small_masks_df
        self.large_masks = large_masks_df
        self.device = device

    def __len__(self):
        return len(self.subjects)

    def extract_kernel(self, image, center, size):
        """Extract a 3D kernel from an image at a given center."""
        half_size = size // 2
        slices = []
        for dim, c in enumerate(center):
            start = max(0, c - half_size)
            end = min(image.shape[dim], c + half_size + 1)
            slices.append(slice(start, end))
        kernel = image[slices[0], slices[1], slices[2]]

        # Pad if kernel is smaller than required size
        padding = [(max(0, half_size - c), max(0, (c + half_size + 1) - image.shape[dim])) for dim, c in enumerate(center)]
        kernel = np.pad(kernel, padding, mode='constant', constant_values=0)
        return kernel

    def __getitem__(self, idx):
        subject_row = self.subjects.iloc[idx]
        subject_id = subject_row["src_subject_id"]
        image_path = subject_row["PathToFile"]
        working_memory = subject_row["tfmri_nb_all_beh_c2b_rate_norm"]

        # Load MRI image
        img = nib.load(image_path)
        image_data = torch.tensor(img.get_fdata(), dtype=torch.float32).to(self.device)

        # Process 25x25x25 regions
        cnn1_data = []
        cnn1_encodings = []
        for _, mask_row in self.small_masks.iterrows():
            center = mask_row["Mask Center"].astype(int)
            mni_scaled = torch.tensor(mask_row["Mni_Scaled"], dtype=torch.float32).to(self.device)
            kernel = torch.tensor(self.extract_kernel(image_data.cpu().numpy(), center, size=25), dtype=torch.float32).to(self.device)

            cnn1_data.append(kernel)
            cnn1_encodings.append(mni_scaled)

        # Process 41x41x41 regions
        cnn2_data = []
        cnn2_encodings = []
        for _, mask_row in self.large_masks.iterrows():
            center = mask_row["Mask Center"].astype(int)
            mni_scaled = torch.tensor(mask_row["Mni_Scaled"], dtype=torch.float32).to(self.device)
            kernel = torch.tensor(self.extract_kernel(image_data.cpu().numpy(), center, size=41), dtype=torch.float32).to(self.device)

            cnn2_data.append(kernel)
            cnn2_encodings.append(mni_scaled)

        cnn1_data = torch.stack(cnn1_data) if cnn1_data else torch.empty(0).to(self.device)
        cnn2_data = torch.stack(cnn2_data) if cnn2_data else torch.empty(0).to(self.device)
        cnn1_encodings = torch.stack(cnn1_encodings) if cnn1_encodings else torch.empty(0).to(self.device)
        cnn2_encodings = torch.stack(cnn2_encodings) if cnn2_encodings else torch.empty(0).to(self.device)
        working_memory = torch.tensor([working_memory], dtype=torch.float32).to(self.device)

        return subject_id, cnn1_data, cnn2_data, cnn1_encodings, cnn2_encodings, working_memory


def split_dataframe(df, num_splits):
    """Split a DataFrame into a specified number of smaller DataFrames."""
    return np.array_split(df, num_splits)


def worker(args):
    """Worker function to process a subset of the dataset."""
    subset, small_masks_df, large_masks_df, output_dir, gpu_id = args
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    dataset = MRIDataset(subset, small_masks_df, large_masks_df, device)
    model = MRIProcessingModel(output_dir)

    for subject_id, cnn1_data, cnn2_data, cnn1_encodings, cnn2_encodings, working_memory in dataset:
        model.process_subject(subject_id, cnn1_data, cnn2_data, cnn1_encodings, cnn2_encodings, working_memory)


if __name__ == "__main__":
    # Detect available GPUs
    num_gpus = torch.cuda.device_count()

    # Load data
    small_masks_df = pd.read_csv("/data/users4/sdeshpande8/Dense_Brain_Aware_VIT/Mask_Data/Small_Masks.csv")
    large_masks_df = pd.read_csv("/data/users4/sdeshpande8/Dense_Brain_Aware_VIT/Mask_Data/Large_Masks.csv")
    subject_df = pd.read_csv("/data/users4/sdeshpande8/Dense_Brain_Aware_VIT/Subject_Data/SMRI_Dataset_Earliest.csv")

    # Parse columns
    for df in [small_masks_df, large_masks_df]:
        df["Mask Center"] = df["Mask Center"].apply(lambda x: np.array(eval(x)[:3]))
        df["Mni_Scaled"] = df["Mni_Scaled"].apply(lambda x: np.array(eval(x)))

    # Output directory
    output_dir = "/data/users4/sdeshpande8/precomputed_tensors_working_memory"
    os.makedirs(output_dir, exist_ok=True)

    # Split the dataset
    subject_subsets = split_dataframe(subject_df, num_gpus)
    logger.info(f"Dataset split into {num_gpus} subsets for processing.")

    # Prepare arguments for multiprocessing
    args_list = [
        (subject_subsets[gpu_id], small_masks_df, large_masks_df, output_dir, gpu_id)
        for gpu_id in range(num_gpus)
    ]

    # Create multiprocessing pool
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_gpus) as pool:
        pool.map(worker, args_list)

    logger.info("Processing complete.")
