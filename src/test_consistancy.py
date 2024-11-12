import os

# flush any cached logs
# command = "rm logs/ -r"
# os.system(command)

i = 0
command = f"""
python decentralized_main.py  \
    --prox_coeff 0 \
    --aggregation_strategy unweighted \
    --rounds 2 --out_dir logs_no_train/{i}/ \
    --no_train \
"""
command = f"""
python decentralized_main.py  \
    --prox_coeff 0 \
    --aggregation_strategy unweighted \
    --rounds 2 --out_dir logs/no_agg/{i}/ \
    --aggregation_strategy test_agg \
"""
command = f"""
python decentralized_main.py  \
    --prox_coeff 0 \
    --aggregation_strategy unweighted \
    --rounds 2 --out_dir logs/w_agg/{i}/ \
    --aggregation_strategy unweighted \
"""
for i in range(50):
    command = f"""
    python decentralized_main.py  \
        --prox_coeff 0 \
        --aggregation_strategy unweighted \
        --rounds 2 --out_dir logs/scale_agg/{i}/ \
        --aggregation_strategy scale_agg \
    """
    os.system(command)
