"""
runner.py — CLI entry point for CT-COEVO.

Usage:
    python runner.py --mode evo --dataset <evoset_dataset> --api-key sk-xxx
    python runner.py --mode eval --dataset <evalset_dataset> --api-key sk-xxx
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# EvalSet datasets (from paper Table 1)
EVALSET_DATASETS = {
    "acquire_valued_shoppers_challenge": {"category": "CTR", "metric": "auc"},
    "airbnb_recruiting_new_user_bookings": {"category": "MLC", "metric": "ndcg"},
    "avazu_ctr_prediction": {"category": "CTR", "metric": "auc"},
    "beeradvocate": {"category": "R&S", "metric": "rmse"},
    "criteo_display_advertising_challenge": {"category": "CTR", "metric": "auc"},
    "diginetica_session_recommendation": {"category": "Seq", "metric": "mrr"},
    "elo_merchant_category_recommendation": {"category": "R&S", "metric": "rmse"},
    "emotion_aware_music_recommendation": {"category": "Rank", "metric": "recall"},
    "event_recommendation_engine_challenge": {"category": "Rank", "metric": "ndcg"},
    "expedia_hotel_recommendations": {"category": "Rank", "metric": "ndcg"},
    "h_and_m_personalized_fashion_recommendations": {"category": "Rank", "metric": "recall"},
    "home_depot_product_search_relevance": {"category": "R&S", "metric": "rmse"},
    "ijcai_2018_alimama_cvr": {"category": "CTR", "metric": "auc"},
    "job_view_recommendation": {"category": "CTR", "metric": "auc"},
    "kgrec_music_sound_recommendation": {"category": "Rank", "metric": "ndcg"},
    "kkbox_music_recommendation_challenge": {"category": "CTR", "metric": "auc"},
    "kuairand_video_ctr": {"category": "CTR", "metric": "auc"},
    "lfm_1b_music_recommendation": {"category": "Rank", "metric": "recall"},
    "malanshan_video_recommendation": {"category": "CTR", "metric": "auc"},
    "merrec_c2c_session_recommendation": {"category": "Seq", "metric": "mrr"},
    "product_recommendation_2025": {"category": "Rank", "metric": "ndcg"},
    "ratebeer_multi_aspect_reviews": {"category": "R&S", "metric": "rmse"},
    "recsys_2018_spotify": {"category": "Rank", "metric": "recall"},
    "rental_product_recommendation": {"category": "CTR", "metric": "auc"},
    "retailrocket_recommendation": {"category": "CTR", "metric": "auc"},
    "santander_product_recommendation": {"category": "MLC", "metric": "f1"},
    "social_ecommerce": {"category": "CTR", "metric": "auc"},
    "taobao_recommendation": {"category": "CTR", "metric": "auc"},
    "yelp_recsys_2013": {"category": "Rank", "metric": "ndcg"},
}

# EvoSet datasets (from paper Table 1)
EVOSET_DATASETS = {
    "book_crossing": {"category": "R&S", "metric": "rmse"},
    "douban_movie": {"category": "R&S", "metric": "rmse"},
    "eachmovie": {"category": "R&S", "metric": "rmse"},
    "epinions": {"category": "R&S", "metric": "rmse"},
    "flixster": {"category": "R&S", "metric": "rmse"},
    "jester": {"category": "R&S", "metric": "rmse"},
    "kdd2003": {"category": "R&S", "metric": "rmse"},
    "lastfm_360k": {"category": "R&S", "metric": "rmse"},
    "ml_1m": {"category": "R&S", "metric": "rmse"},
    "netflix": {"category": "R&S", "metric": "rmse"},
    "steam_data": {"category": "R&S", "metric": "rmse"},
    "yahoo_music_r1": {"category": "R&S", "metric": "rmse"},
    "digg2009": {"category": "CTR", "metric": "auc"},
    "epinions_binary": {"category": "CTR", "metric": "auc"},
    "ml_1m_binary": {"category": "CTR", "metric": "auc"},
    "msnbc_binary": {"category": "CTR", "metric": "auc"},
    "netflix_binary": {"category": "CTR", "metric": "auc"},
    "synthetic_ctr": {"category": "CTR", "metric": "auc"},
    "yahoo_binary": {"category": "CTR", "metric": "auc"},
    "amazon_grocery": {"category": "MLC", "metric": "f1"},
    "amazon_video": {"category": "MLC", "metric": "f1"},
    "movielens_binary": {"category": "MLC", "metric": "f1"},
    "amazon_beauty": {"category": "Rank", "metric": "ndcg"},
    "amazon_electronics": {"category": "Rank", "metric": "ndcg"},
    "amazon_musical": {"category": "Rank", "metric": "ndcg"},
    "amazon_office": {"category": "Rank", "metric": "ndcg"},
    "gowalla": {"category": "Rank", "metric": "recall"},
    "lastfm_1k": {"category": "Rank", "metric": "recall"},
    "ml_10m": {"category": "Rank", "metric": "recall"},
    "ml_20m": {"category": "Rank", "metric": "recall"},
    "ml_25m": {"category": "Rank", "metric": "recall"},
    "amazon_cds": {"category": "Seq", "metric": "mrr"},
    "amazon_kindle": {"category": "Seq", "metric": "mrr"},
    "diginetica": {"category": "Seq", "metric": "mrr"},
}


def resolve_data_dir(dataset_name: str) -> Optional[Path]:
    """Resolve the data directory for a dataset."""
    roots = [
        Path("./RECDEVBENCH/recdevbench/evalset"),
        Path("./RECDEVBENCH/recdevbench/evoset"),
    ]
    for root in roots:
        candidate = root / dataset_name
        if candidate.exists():
            return candidate
        candidate = root / dataset_name / "data" / "public"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="CT-COEVO Runner")
    parser.add_argument("--mode", choices=["evo", "eval"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--api-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=86400)
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.dataset)
    if data_dir is None:
        print(f"Error: Dataset '{args.dataset}' not found")
        sys.exit(1)

    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Data: {data_dir}")
    print(f"Model: {args.model}")
    print(f"Steps: {args.max_steps}")
    print(f"Timeout: {args.timeout}s")

    evolve = args.mode == "evo"

    from ct_coevo.ct_coevo_agent import CTCoEvoAgent
    agent = CTCoEvoAgent(
        dataset_name=args.dataset,
        data_dir=str(data_dir),
        api_key=args.api_key,
        model=args.model,
        base_url=args.api_url,
        
        timeout_sec=args.timeout,
        evolve=evolve,
    )
    result = agent.run()

    print(f"\n{'='*60}")
    print(f"Final: {args.dataset}")
    print(f"Best Score: {result['best_score']}")
    print(f"Steps: {result['steps_completed']}")
    print(f"Time: {result['elapsed_seconds']:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
