import argparse
import os

from preprocess.lv0 import preprocess_emnist
from preprocess.lv1 import preprocess_1k_pbm
from models.model_selection import model_selection_pipeline

def main():
    parser = argparse.ArgumentParser(description="Captcha processing pipeline")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["0", "1", "2"],
        help="Choose a stage to run",
        default="0"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["lv0_emnist", "lv1_1k_pbm", "lv1_ctt_sis", "lv2_1k_5digits", "lv2_1k_pbm_noise"],
        help="Choose a dataset to run"
    )
    
    args = parser.parse_args()
    
    if args.stage == "0":
        if args.dataset == "lv0_emnist":
            # Preprocessing
            preprocess_emnist()
        
            # Model selection
            model_selection_pipeline(
                data_dir="output/dataset/lv0_emnist/build", 
                output_dir="output/models/lv0_emnist/"
            )
        
        elif args.dataset == "lv1_1k_pbm":
            preprocess_1k_pbm()
            model_selection_pipeline(
                data_dir="output/dataset/lv1_1k_pbm/build", 
                output_dir="output/models/lv1_1k_pbm/"
            )
        
    # 1. Preprocessing
    elif args.stage == "1":
        if args.dataset == "lv0_emnist":
            preprocess_emnist()
        if args.dataset == "lv1_1k_pbm":
            preprocess_1k_pbm()
        
    # 2. Model selection
    elif args.stage == "2":
        model_selection_pipeline(
            data_dir=f"output/dataset/{args.dataset}/build", 
            output_dir=f"output/models/{args.dataset}/"
        )
    


if __name__ == "__main__":
    main()
