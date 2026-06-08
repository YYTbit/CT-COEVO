import sys
sys.path.insert(0, '/data/yangyingtao02')

from ct_coevo import run_ct_coevo

result = run_ct_coevo(
    dataset_name="recsys_2018_spotify",
    data_dir="/data/yangyingtao02/RECDEVBENCH/recdevbench/evalset/recsys_2018_spotify/data/public",
    api_key="sk-cvbft3yx5jjede999b4u370ty9orqklr28xl72ydzb0q0dca",
    model="mimo-v2.5",
    base_url="https://api.xiaomimimo.com/v1",
    timeout_sec=86400,
    evolve=True,
)

print(f"DONE. Best score: {result['best_score']}")
