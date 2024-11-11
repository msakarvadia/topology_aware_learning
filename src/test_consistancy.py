import os

# flush any cached logs
command = "rm logs/ -r"
os.system(command)

for i in range(5):
    command = f"""
    python decentralized_main.py  \
        --prox_coeff 0 \
        --aggregation_strategy unweighted \
        --rounds 2 --out_dir logs/{i}/ \
        --no_train \
    """
    os.system(command)
