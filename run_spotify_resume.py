import sys
sys.path.insert(0, '/data/yangyingtao02')

from ct_coevo import CTCoEvoAgent

# Resume from the latest workspace
agent = CTCoEvoAgent(
    dataset_name="recsys_2018_spotify",
    data_dir="/data/yangyingtao02/RECDEVBENCH/recdevbench/evalset/recsys_2018_spotify/data/public",
    api_key="YOUR_API_KEY",
    model="mimo-v2.5",
    base_url="https://api.xiaomimimo.com/v1",
    timeout_sec=86400,
    evolve=True,
    workspace_dir="/data/yangyingtao02/ct_coevo/workspace/recsys_2018_spotify-20260607_221531",
)

result = agent.run()
print(f"DONE. Best score: {result['best_score']}")
