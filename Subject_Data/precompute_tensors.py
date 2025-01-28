import os
import torch
import pandas as pd
import nibabel as nib
import ast
import numpy as np
from tqdm import tqdm
import multiprocessing
import logging
import argparse
from typing import List, Tuple

def split_dataframe(df: pd.DataFrame, num_splits: int) -> List[pd.DataFrame]:
    """
    Split a DataFrame into `num_splits` approximately equal parts.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        num_splits (int): Number of subsets to create.

    Returns:
        List[pd.DataFrame]: List containing the split DataFrames.
    """
    return np.array_split(df, num_splits)

def preprocess_subset(subject_subset: pd.DataFrame, mask1_df: pd.DataFrame, mask2_df: pd.DataFrame, output_dir: str, gpu_id: int):
    """
    Preprocess a subset of subjects using a specific GPU.

    Args:
        subject_subset (pd.DataFrame): Subset of the subject DataFrame.
        mask1_df (pd.DataFrame): DataFrame for Small Masks.
        mask2_df (pd.DataFrame): DataFrame for Large Masks.
        output_dir (str): Directory to save precomputed tensors.
        gpu_id (int): GPU index to use for processing.
    """
    # Configure logging for this GPU
    logging.basicConfig(
        filename=f'preprocessing_gpu_{gpu_id}.log',
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )
    logger = logging.getLogger(f'GPU_{gpu_id}')
    logger.info(f"Process started for GPU {gpu_id}")

    # Set the specific GPU for this process
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Process using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU {gpu_id}: {torch.cuda.get_device_name(device)}")
    else:
        logger.warning(f"GPU {gpu_id} not available. Using CPU instead.")

    def load_masks(mask_df: pd.DataFrame) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Load masks and their encodings from a DataFrame.

        Args:
            mask_df (pd.DataFrame): DataFrame containing mask information.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: List of mask tensors and stacked encodings tensor.
        """
        masks = []
        encodings = []
        for _, row in mask_df.iterrows():
            try:
                mask_path = row['Mask File Path']
                mask_np = nib.load(mask_path).get_fdata()
                mask_tensor = torch.tensor(mask_np, dtype=torch.float32, device=device)
                masks.append(mask_tensor)

                encoding_str = row['Mni_Scaled']
                encoding_list = ast.literal_eval(encoding_str)
                encoding_tensor = torch.tensor(encoding_list, dtype=torch.float32, device=device)
                encodings.append(encoding_tensor)
            except FileNotFoundError:
                logger.error(f"Mask file not found: {mask_path}. Skipping this mask.")
            except (ValueError, SyntaxError) as e:
                logger.error(f"Error parsing 'Mni_Scaled' for mask at {mask_path}: {e}. Skipping this mask.")
            except Exception as e:
                logger.exception(f"Unexpected error loading mask or encoding for mask at {mask_path}: {e}. Skipping this mask.")

        if encodings:
            try:
                encodings = torch.stack(encodings)
            except Exception as e:
                logger.error(f"Error stacking encodings: {e}. Using empty tensor.")
                encodings = torch.empty(0, device=device)
        else:
            encodings = torch.empty(0, device=device)

        return masks, encodings

    def apply_masks_and_pad(image: torch.Tensor, masks: List[torch.Tensor], target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Apply masks to an image and pad/crop to the target shape.

        Args:
            image (torch.Tensor): The image tensor.
            masks (List[torch.Tensor]): List of mask tensors.
            target_shape (Tuple[int, int, int]): The desired shape after padding/cropping.

        Returns:
            torch.Tensor: Tensor containing masked and padded/cropped images.
        """
        masked_images = []
        for mask in masks:
            try:
                masked_image = image * mask  # Element-wise multiplication
                padded_image = truncate(masked_image, target_shape)
                if padded_image.numel() > 0:
                    masked_images.append(padded_image)
            except Exception as e:
                logger.error(f"Error applying mask and padding: {e}. Skipping this mask.")

        if masked_images:
            try:
                return torch.stack(masked_images).unsqueeze(1)  # Shape: (num_masks, 1, D, H, W)
            except Exception as e:
                logger.error(f"Error stacking masked images: {e}. Returning empty tensor.")
                return torch.empty(0, 1, *target_shape, device=device)
        else:
            return torch.empty(0, 1, *target_shape, device=device)

    def truncate(region: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Truncate (center crop) the image to the target shape.

        Args:
            region (torch.Tensor): The masked image tensor.
            target_shape (Tuple[int, int, int]): The desired shape.

        Returns:
            torch.Tensor: The truncated tensor.
        """
        current_shape = region.shape
        min_dim = [min(ts, cs) for ts, cs in zip(target_shape, current_shape)]
        start_indices = [(cs - ts) // 2 for ts, cs in zip(min_dim, current_shape)]
        try:
            cropped_region = region[
                start_indices[0]:start_indices[0] + min_dim[0],
                start_indices[1]:start_indices[1] + min_dim[1],
                start_indices[2]:start_indices[2] + min_dim[2]
            ]
            return cropped_region
        except Exception as e:
            logger.error(f"Error during truncation: {e}. Returning empty tensor.")
            return torch.empty(0, device=device)

    # Load masks
    masks_cnn1, encodings_cnn1 = load_masks(mask1_df)
    masks_cnn2, encodings_cnn2 = load_masks(mask2_df)

    processed_count = 0
    skipped_count = 0

    for idx in tqdm(range(len(subject_subset)), desc=f"GPU {gpu_id} Preprocessing"):
        row = subject_subset.iloc[idx]
        # Incorporate gpu_id into subject_id to ensure uniqueness
        subject_id = f"gpu{gpu_id}_idx{idx}"
        img_path = row.get('PathToFile', None)

        if not img_path:
            logger.error(f"No 'PathToFile' found for subject {subject_id}. Skipping.")
            skipped_count += 1
            continue

        # Load image
        try:
            image_np = nib.load(img_path).get_fdata()
            image = torch.tensor(image_np, dtype=torch.float32, device=device)
        except FileNotFoundError:
            logger.warning(f"Image file not found: {img_path}. Skipping subject {subject_id}.")
            skipped_count += 1
            continue
        except Exception as e:
            logger.error(f"Error loading image {img_path} for subject {subject_id}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Apply masks and pad/crop
        try:
            cnn1_data = apply_masks_and_pad(image, masks_cnn1, (25, 25, 25))
            cnn2_data = apply_masks_and_pad(image, masks_cnn2, (41, 41, 41))
        except Exception as e:
            logger.error(f"Error processing masks for subject {subject_id}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Get working memory score
        try:
            working_memory_score = torch.tensor(row['tfmri_nb_all_beh_c2b_rate_norm'], dtype=torch.float32, device=device)
        except KeyError:
            logger.error(f"'tfmri_nb_all_beh_c2b_rate_norm' not found for subject {subject_id}. Skipping.")
            skipped_count += 1
            continue
        except Exception as e:
            logger.error(f"Error processing working memory score for subject {subject_id}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Move tensors to CPU
        try:
            cnn1_data_cpu = cnn1_data.cpu()
            cnn2_data_cpu = cnn2_data.cpu()
            working_memory_score_cpu = working_memory_score.cpu()
            encodings_cnn1_cpu = encodings_cnn1.cpu()
            encodings_cnn2_cpu = encodings_cnn2.cpu()
        except Exception as e:
            logger.error(f"Error moving tensors to CPU for subject {subject_id}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Save tensors in GPU-specific subdirectory
        subject_output_dir = os.path.join(output_dir, f"gpu_{gpu_id}", f"subject_{subject_id}")
        os.makedirs(subject_output_dir, exist_ok=True)

        try:
            torch.save(cnn1_data_cpu, os.path.join(subject_output_dir, 'cnn1_data.pt'))
            torch.save(cnn2_data_cpu, os.path.join(subject_output_dir, 'cnn2_data.pt'))
            torch.save(working_memory_score_cpu, os.path.join(subject_output_dir, 'working_memory_score.pt'))
            torch.save(encodings_cnn1_cpu, os.path.join(subject_output_dir, 'encodings_cnn1.pt'))
            torch.save(encodings_cnn2_cpu, os.path.join(subject_output_dir, 'encodings_cnn2.pt'))
            logger.info(f"Successfully saved tensors for subject {subject_id}.")
            processed_count += 1
        except Exception as e:
            logger.error(f"Error saving tensors for subject {subject_id}: {e}.")
            skipped_count += 1
            continue

        # Free GPU memory
        del image, cnn1_data, cnn2_data, working_memory_score
        torch.cuda.empty_cache()

    logger.info(f"GPU {gpu_id} completed. Processed: {processed_count}, Skipped: {skipped_count}")

def worker(preprocess_subset_args):
    """
    Worker function to unpack arguments and call preprocess_subset.

    Args:
        preprocess_subset_args (tuple): Arguments for preprocess_subset.
    """
    preprocess_subset(*preprocess_subset_args)

def main():
    """
    Main function to orchestrate the preprocessing across multiple GPUs and CPUs.
    """
    parser = argparse.ArgumentParser(description='Preprocess neuroimaging data.')
    parser.add_argument('--subject_csv', type=str, default='Subject_Data/SMRI_Dataset_Earliest.csv', help='Path to subject CSV file.')
    parser.add_argument('--mask1_csv', type=str, default='3D_Mask_Data/Small_Masks.csv', help='Path to Small Masks CSV file.')
    parser.add_argument('--mask2_csv', type=str, default='3D_Mask_Data/Large_Masks.csv', help='Path to Large Masks CSV file.')
    parser.add_argument('--output_dir', type=str, default='precomputed_tensors', help='Directory to save precomputed tensors.')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level.')
    args = parser.parse_args()

    # Configure the root logger
    logging.basicConfig(
        filename='preprocessing_main.log',
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
    )
    logger = logging.getLogger('Main')
    logger.info("Starting preprocessing pipeline.")

    # Define the paths
    subject_csv_file = args.subject_csv
    mask1_csv_file = args.mask1_csv
    mask2_csv_file = args.mask2_csv
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV files with error handling
    try:
        subject_df = pd.read_csv(subject_csv_file)
        logger.info(f"Successfully loaded subject CSV file: {subject_csv_file}")
    except FileNotFoundError:
        logger.error(f"Subject CSV file not found: {subject_csv_file}")
        return
    except Exception as e:
        logger.error(f"Error loading subject CSV file: {e}")
        return

    try:
        mask1_df = pd.read_csv(mask1_csv_file)
        logger.info(f"Successfully loaded Mask1 CSV file: {mask1_csv_file}")
    except FileNotFoundError:
        logger.error(f"Mask1 CSV file not found: {mask1_csv_file}")
        return
    except Exception as e:
        logger.error(f"Error loading Mask1 CSV file: {e}")
        return

    try:
        mask2_df = pd.read_csv(mask2_csv_file)
        logger.info(f"Successfully loaded Mask2 CSV file: {mask2_csv_file}")
    except FileNotFoundError:
        logger.error(f"Mask2 CSV file not found: {mask2_csv_file}")
        return
    except Exception as e:
        logger.error(f"Error loading Mask2 CSV file: {e}")
        return

    # Validate required columns
    required_columns = ['src_subject_id', 'PathToFile', 'tfmri_nb_all_beh_c2b_rate_norm']
    missing_columns = [col for col in required_columns if col not in subject_df.columns]
    if missing_columns:
        logger.error(f"Missing columns in subject CSV: {missing_columns}")
        return

    # Split the dataset into subsets for GPUs
    num_gpus = args.num_gpus
    subject_subsets = split_dataframe(subject_df, num_gpus)
    logger.info(f"Dataset split into {num_gpus} subsets for processing.")

    # Prepare arguments for each process
    args_list = []
    for gpu_id in range(num_gpus):
        subset = subject_subsets[gpu_id]
        args_tuple = (subset, mask1_df, mask2_df, output_dir, gpu_id)
        args_list.append(args_tuple)

    # Create a multiprocessing Pool with `num_gpus` processes
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_gpus) as pool:
        pool.map(worker, args_list)

    logger.info("Preprocessing completed across all GPUs.")
    print("Preprocessing completed across all GPUs.")

if __name__ == "__main__":
    main()
