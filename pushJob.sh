#!/bin/bash
#SBATCH -p qTRDGPUH
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH -e error%A.err 
#SBATCH -o out%A.out
#SBATCH -A trends396s109
#SBATCH -J Working_Memory_Prediction
#SBATCH --oversubscribe
#SBATCH --mail-user=pvdsan@gmail.com


# a small delay at the start often helps
sleep 2s 

#activate the environment
source /home/users/sdeshpande8/anaconda3/bin/activate cogn

# CD into your directory
cd /data/users4/sdeshpande8/Brain_Aware_Transformer_3D/

# run the matlab batch script
python main.py  --model_name baseline_3_layer_cnn_2_layer_regressor_100_features_lr_1e-5_no_pos

# a delay at the end is also good practice
sleep 10s
