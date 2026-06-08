#!/bin/bash
cd /data/yangyingtao02/ct_coevo
source /data/yangyingtao02/miniconda3/bin/activate CT-COEVO
python3 run_spotify_resume.py 2>&1 | tee log_resume_$(date +%Y%m%d_%H%M%S).log
