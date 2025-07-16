#!/bin/bash

#SBATCH --job-name=llama3-8B-context_scaling
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=8 
#SBATCH --cpus-per-task=16        
#SBATCH --mem=600G              
#SBATCH --gres=gpu:8                       
#SBATCH --time=14-00:00:00   
#SBATCH --output=/mnt/petrelfs/tangzecheng/MyRLHF/sbatch_logs/%J.out       
#SBATCH --error=/mnt/petrelfs/tangzecheng/MyRLHF/sbatch_logs/%J.err         
#SBATCH --partition=belt_road  
#SBATCH --exclusive     

export http_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/ 
export https_proxy=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTP_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/
export HTTPS_PROXY=http://tangzecheng:Jn7iXe92XJUVYa5whNh07VJKZR6miGQ62it3goTiLBxRs8uZxkFD3gF0cQ3w@10.1.20.50:23128/

source /mnt/petrelfs/tangzecheng/anaconda3/etc/profile.d/conda.sh

conda activate zecheng

cd /mnt/petrelfs/tangzecheng/MyRLHF/openrlhf

bash scripts/llama3-8B-base/context_scaling_lora_baseline.sh