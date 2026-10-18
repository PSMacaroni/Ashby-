#!/bin/bash
# Pull nerve / vessel / other tiles out of a SPIDER dataset (CPU only, mostly network).
# Arguments go straight to get_spider.py. From the project folder on Delta:
#
#     mkdir -p data/logs
#     sbatch scripts/models/get_spider_cpu.sbatch --organ colorectal --out /projects/bfqi/$USER/datasets/spider_colorectal
#     sbatch scripts/models/get_spider_cpu.sbatch --organ skin --out /projects/bfqi/$USER/datasets/spider_skin \
#         --vessel-cap 3000 --other-per-class 250
#     tail -f data/logs/get_spider-<jobid>.out
#SBATCH --job-name=get_spider
#SBATCH --account=bfqi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --output=data/logs/%x-%j.out

source scripts/delta_env.sh
python scripts/models/get_spider.py --workers 14 "$@"
