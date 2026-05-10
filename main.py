import argparse
import os

from datalayer.pbm_1k_process import process_1k_pbm_dataset
from preprocess.general_preprocess import run_preprocessing_pipeline
from segment.general_segmentation import run_segmentation_pipeline
from datalayer.build_dataset import run_dataset_pipeline
from model.train import run_training_pipeline
from model.inference import run_inference_demo
from model.evaluate import evaluate_system

def main():
    parser = argparse.ArgumentParser(description="Captcha processing pipeline")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["0", "1", "2", "3", "4", "5", "6", "7"],
        help="Choose a stage to run"
    )

    args = parser.parse_args()

    if args.stage == "0":
        # 1. Data layer
        process_1k_pbm_dataset()
        
        # 2. Preprocessing
        run_preprocessing_pipeline()
        
        # 3. Segmentation
        run_segmentation_pipeline()

        # 4. Build dataset
        run_dataset_pipeline()

        # 5. Train model
        run_training_pipeline()
        
        # 6. Inference Demo
        run_inference_demo()
        
        # 7. Evaluation
        evaluate_system()
        
    elif args.stage == "1":
        # Change depends on dataset
        process_1k_pbm_dataset()
    elif args.stage == "2":
        run_preprocessing_pipeline()
    elif args.stage == "3":
        run_segmentation_pipeline()
    elif args.stage == "4":
        run_dataset_pipeline()
    elif args.stage == "5":
        run_training_pipeline()
    elif args.stage == "6":
        run_inference_demo()
    elif args.stage == "7":
        evaluate_system()


if __name__ == "__main__":
    main()
