#!/bin/bash
source /data/yangyingtao02/miniconda3/bin/activate CT-COEVO
cd /data/yangyingtao02/ct_coevo
export PYTHONPATH=/data/yangyingtao02:$PYTHONPATH

DATASET=$1
DATA_DIR="/data/yangyingtao02/RECDEVBENCH/recdevbench/evalset/${DATASET}/data/public"

echo "=============================================="
echo "CT-COEVO EVO MODE"
echo "Dataset: $DATASET"
echo "Data: $DATA_DIR"
echo "=============================================="

python -u -c "
from ct_coevo.agent import CTCoEvoAgent
agent = CTCoEvoAgent(
    dataset_name='$DATASET',
    data_dir='$DATA_DIR',
    api_key='tp-ctkhn0m3puyckkebcnz9xu87s09tw2bpwbl62nhk7q9xq09q',
    model='mimo-v2.5-pro',
    base_url='https://token-plan-cn.xiaomimimo.com/v1',
    timeout_sec=86400,
    evolve=True,
    state_dir='/data/yangyingtao02/ct_coevo/state/global',
)
result = agent.run()
print(f'DONE. Best score: {result[\"best_score\"]}')
"
