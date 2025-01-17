#!/bin/bash

#SBATCH --job-name=stream_speech_eval_job
#SBATCH --account=cse
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -o log/eval/%x_%j.out
#SBATCH -e log/eval/%x_%j.err

./download_dataset_and_eval.sh
