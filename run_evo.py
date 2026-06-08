import sys
sys.path.insert(0, '/data/yangyingtao02')

from ct_coevo import run_ct_coevo

result = run_ct_coevo(
    dataset_name="acquire_valued_shoppers_challenge",
    data_dir="/data/yangyingtao02/RECDEVBENCH/recdevbench/evalset/acquire_valued_shoppers_challenge/data/public",
    api_key="tp-ctkhn0m3puyckkebcnz9xu87s09tw2bpwbl62nhk7q9xq09q",
    model="mimo-v2.5-pro",
    base_url="https://token-plan-cn.xiaomimimo.com/v1",
    timeout_sec=86400,
    evolve=True,
)

print(f"DONE. Best score: {result['best_score']}")
