#!/bin/bash

#SBATCH --job-name=stream_speech_train_job
#SBATCH --account=intelligentsystems
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -o log/train/%x_%j.out
#SBATCH -e log/train/%x_%j.err

./download_dataset_and_train.sh
