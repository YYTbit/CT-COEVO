import sys
sys.path.insert(0, '/data/yangyingtao02')

from ct_coevo.agent import CTCoEvoAgent

agent = CTCoEvoAgent(
    dataset_name="recsys_2018_spotify",
    data_dir="/data/yangyingtao02/RECDEVBENCH/recdevbench/evalset/recsys_2018_spotify/data/public",
    api_key="tp-ctkhn0m3puyckkebcnz9xu87s09tw2bpwbl62nhk7q9xq09q",
    model="mimo-v2.5-pro",
    base_url="https://token-plan-cn.xiaomimimo.com/v1",
    timeout_sec=86400,
    evolve=True,
    workspace_dir="/data/yangyingtao02/ct_coevo/workspace/recsys_2018_spotify-20260606_154801",
)

result = agent.run()
print(f"DONE. Best score: {result['best_score']}")
