# Set up Python on Delta. Run this in every new Delta terminal or job:
#
#     source scripts/delta_env.sh
#
# 1. Load Delta's PyTorch (GPU-ready torch, Python 3.12).
module load pytorch-conda/2.12
# 2. That module hides packages installed with `pip install --user` (it sets
#    PYTHONNOUSERSITE=1), which is where our extra packages live. Show them again.
unset PYTHONNOUSERSITE
