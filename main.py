import argparse
import os

from preprocess.lv1 import preprocess_1k_pbm
from models.lv1 import models_1k_pbm

def main():
    parser = argparse.ArgumentParser(description="Captcha processing pipeline")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["0", "1", "2", "3", "4", "5", "6", "7"],
        help="Choose a stage to run",
        default="0"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["lv1_1k_pbm", "lv1_ctt_sis", "lv2_1k_5digits", "lv2_1k_pbm_noise"],
        help="Choose a dataset to run"
    )
    
    args = parser.parse_args()
    
    if args.stage == "0":
        if args.dataset == "lv1_1k_pbm":
            # Preprocessing
            preprocess_1k_pbm()
        
            # Model selection
            models_1k_pbm()
        
        
    # Preprocessing
    elif args.stage == "1":
        if args.dataset == "lv1_1k_pbm":
            preprocess_1k_pbm()
        
    elif args.stage == "2":
        if args.dataset == "lv1_1k_pbm":
            models_1k_pbm()
    


if __name__ == "__main__":
    main()
