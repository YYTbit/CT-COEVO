"""
runner.py - CLI entry point for CT-COEVO.

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

# EvalSet (49 datasets, 2012-2025), recent recommendation datasets for transfer evaluation
EVALSET_DATASETS = {
    "yelp_recsys_2013": {"category": "R&S", "metric": "rmse", "year": "2013"},
    "zomato_restaurant_recommendation": {"category": "R&S", "metric": "rmse", "year": "2019"},
    "lfm_1b_music_recommendation": {"category": "R&S", "metric": "rmse", "year": "2016"},
    "elo_merchant_category_recommendation": {"category": "R&S", "metric": "rmse", "year": "2019"},
    "beeradvocate": {"category": "R&S", "metric": "rmse", "year": "2017"},
    "ratebeer": {"category": "R&S", "metric": "rmse", "year": "2017"},
    "avazu_ctr_prediction": {"category": "CTR", "metric": "logloss", "year": "2014"},
    "wechat_recommendation": {"category": "CTR", "metric": "uauc", "year": "2021"},
    "criteo_display_advertising_challenge": {"category": "CTR", "metric": "logloss", "year": "2014"},
    "kdd_cup_2012_track2": {"category": "CTR", "metric": "logloss", "year": "2012"},
    "ijcai_2018_alimama_cvr": {"category": "CTR", "metric": "logloss", "year": "2018"},
    "recsys_2020_twitter": {"category": "CTR", "metric": "logloss", "year": "2020"},
    "recsys2024_challenge": {"category": "CTR", "metric": "auc", "year": "2024"},
    "stumbleupon_evergreen_classification": {"category": "MLC", "metric": "auc", "year": "2013"},
    "wsdm_2023_fraud_detection": {"category": "MLC", "metric": "auc", "year": "2023"},
    "kkbox_music_recommendation_challenge": {"category": "MLC", "metric": "auc", "year": "2017"},
    "acquire_valued_shoppers_challenge": {"category": "MLC", "metric": "auc", "year": "2014"},
    "tmall_purchase_prediction": {"category": "MLC", "metric": "auc", "year": "2015"},
    "antai_recommendation": {"category": "MLC", "metric": "f1", "year": "2019"},
    "social_ecommerce": {"category": "MLC", "metric": "f1", "year": "2020"},
    "kuairec": {"category": "Rank", "metric": "recall@k", "year": "2022"},
    "event_recommendation_engine_challenge": {"category": "Rank", "metric": "map@200", "year": "2013"},
    "outbrain_click_prediction": {"category": "Rank", "metric": "map@12", "year": "2017"},
    "kdd_cup_2020_debiasing": {"category": "Rank", "metric": "ndcg@50", "year": "2020"},
    "product_recommendation_2025": {"category": "Rank", "metric": "ndcg@10", "year": "2025"},
    "airbnb_recruiting_new_user_bookings": {"category": "Rank", "metric": "ndcg@5", "year": "2016"},
    "expedia_hotel_recommendations": {"category": "Rank", "metric": "map@5", "year": "2016"},
    "emotion_aware_music_recommendation": {"category": "Rank", "metric": "recall@10", "year": "2022"},
    "matchverse_matrimonial_challenge": {"category": "Rank", "metric": "mrr@100", "year": "2023"},
    "kgrec_music_sound_recommendation": {"category": "Rank", "metric": "ndcg@10", "year": "2024"},
    "twitch_stream_recommendation": {"category": "Rank", "metric": "hit@10", "year": "2024"},
    "recsys_challenge_2017": {"category": "Rank", "metric": "precision@k", "year": "2017"},
    "santander_product_recommendation": {"category": "Rank", "metric": "map@7", "year": "2016"},
    "merrec": {"category": "Seq", "metric": "mrr", "year": "2024"},
    "kuairand": {"category": "Seq", "metric": "recall@k", "year": "2022"},
    "yoochoose_recsys2015": {"category": "Seq", "metric": "recall@20", "year": "2015"},
    "retailrocket_recommendation": {"category": "Seq", "metric": "recall@20", "year": "2015"},
    "diginetica_session_recommendation": {"category": "Seq", "metric": "recall@20", "year": "2016"},
    "taobao_recommendation": {"category": "Seq", "metric": "mrr", "year": "2018"},
    "h_and_m_personalized_fashion_recommendations": {"category": "Seq", "metric": "map@12", "year": "2022"},
    "otto_recommender_system": {"category": "Seq", "metric": "recall@20", "year": "2023"},
    "recsys_2018_spotify": {"category": "Seq", "metric": "r-precision", "year": "2018"},
    "wsdm_2021_booking": {"category": "Seq", "metric": "precision@4", "year": "2021"},
    "kdd_cup_2023_amazon": {"category": "Seq", "metric": "recall@10", "year": "2023"},
    "recsys_challenge_2019_trivago": {"category": "Seq", "metric": "mrr", "year": "2019"},
    "recsys_challenge_2022_dressipi": {"category": "Seq", "metric": "mrr", "year": "2022"},
    "malanshan_video_recommendation": {"category": "Seq", "metric": "mrr", "year": "2020"},
    "tianchi_news_recommendation": {"category": "Seq", "metric": "mrr@5", "year": "2020"},
    "rental_product_recommendation": {"category": "Seq", "metric": "mrr@10", "year": "2024"},
}

# EvoSet (34 datasets, 1997-2010), classical recommendation datasets for evolution
EVOSET_DATASETS = {
    "book_crossing": {"category": "R&S", "metric": "rmse", "year": "2004"},
    "douban_movie": {"category": "R&S", "metric": "rmse", "year": "2010"},
    "eachmovie": {"category": "R&S", "metric": "rmse", "year": "1997"},
    "epinions": {"category": "R&S", "metric": "rmse", "year": "2003"},
    "flixster": {"category": "R&S", "metric": "rmse", "year": "2007"},
    "jester": {"category": "R&S", "metric": "rmse", "year": "1999"},
    "kdd2003": {"category": "R&S", "metric": "rmse", "year": "2003"},
    "lastfm_360k": {"category": "R&S", "metric": "rmse", "year": "2008"},
    "ml_1m": {"category": "R&S", "metric": "rmse", "year": "2003"},
    "netflix": {"category": "R&S", "metric": "rmse", "year": "2006"},
    "steam_data": {"category": "R&S", "metric": "rmse", "year": None},
    "yahoo_music_r1": {"category": "R&S", "metric": "rmse", "year": "2005"},
    "digg2009": {"category": "CTR", "metric": "auc", "year": "2009"},
    "epinions_trust": {"category": "CTR", "metric": "auc", "year": "2003"},
    "yahoo_r6a": {"category": "CTR", "metric": "auc", "year": "2009"},
    "bibsonomy": {"category": "MLC", "metric": "map@k", "year": "2007"},
    "brightkite_loc": {"category": "MLC", "metric": "f1", "year": "2009"},
    "citeulike": {"category": "MLC", "metric": "f1", "year": "2010"},
    "delicious": {"category": "MLC", "metric": "map@k", "year": "2007"},
    "slashdot_zoo": {"category": "MLC", "metric": "f1", "year": "2009"},
    "wiki_vote": {"category": "MLC", "metric": "f1", "year": "2008"},
    "amazon_copurch": {"category": "Rank", "metric": "ndcg@10", "year": "2003"},
    "gowalla": {"category": "Rank", "metric": "hr@10", "year": "2009"},
    "lastfm_ranking": {"category": "Rank", "metric": "hr@10", "year": "2005"},
    "libimseti": {"category": "Rank", "metric": "ndcg@10", "year": "2006"},
    "ml_10m": {"category": "Rank", "metric": "ndcg@10", "year": "2009"},
    "amazon_seq": {"category": "Seq", "metric": "mrr", "year": "2003"},
    "gazelle": {"category": "Seq", "metric": "mrr", "year": "2000"},
    "kdd2001": {"category": "Seq", "metric": "mrr", "year": "2001"},
    "kosarak": {"category": "Seq", "metric": "accuracy", "year": "2001"},
    "msnbc": {"category": "Seq", "metric": "accuracy", "year": "1999"},
    "rsc2009": {"category": "Seq", "metric": "mrr", "year": "2009"},
    "tafeng": {"category": "Seq", "metric": "mrr", "year": "2000"},
    "wikispeedia": {"category": "Seq", "metric": "mrr", "year": "2009"},
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
    parser.add_argument("--api-url", default="https://api.example.com/v1")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3.2")
    parser.add_argument("--harrier-model",
                        default=None,
                        help="Harrier-OSS (0.6B) embedding model id or local directory "
                             "(default: microsoft/harrier-oss-v1-0.6b or $HARRIER_MODEL_DIR)")
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
    if args.harrier_model:
        print(f"Harrier model: {args.harrier_model}")
    print(f"Timeout: {args.timeout}s")

    evolve = args.mode == "evo"

    from ct_coevo.agent import CTCoEvoAgent
    agent = CTCoEvoAgent(
        dataset_name=args.dataset,
        data_dir=str(data_dir),
        api_key=args.api_key,
        model=args.model,
        base_url=args.api_url,

        timeout_sec=args.timeout,
        evolve=evolve,
        harrier_model=args.harrier_model,
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
