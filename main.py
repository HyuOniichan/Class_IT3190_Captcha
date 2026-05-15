import argparse
import os

from datalayer.pbm_1k_process import process_1k_pbm_dataset
from datalayer.prepare_dataset_v2 import prepare_dataset_v2
from preprocess.general_preprocess import run_preprocessing_pipeline # For backward compatibility (Dataset 1)
from preprocess.general_preprocess_v2 import run_preprocessing_pipeline_v2 # For new strategies
from preprocess.strategies import Dataset1PreprocessingStrategy, Dataset2PreprocessingStrategy
from segment.general_segmentation import run_segmentation_pipeline # For backward compatibility (Dataset 1)
from segment.general_segmentation_v2 import run_segmentation_pipeline_v2 # For new strategies
from segment.strategies import Dataset1SegmentationStrategy, Dataset2SegmentationStrategy

from datalayer.build_dataset import run_dataset_pipeline
from model.train import run_training_pipeline
from model.inference import run_inference_demo, load_inference_model, predict_captcha
from model.evaluate import evaluate_system

def main():
    parser = argparse.ArgumentParser(description="Captcha processing pipeline")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=["0", "1", "2", "3", "4", "5", "6", "7"],
        help="Choose a stage to run"
    )
    parser.add_argument(
        "--dataset", type=str, default="1",
        choices=["1", "2"],
        help="Choose which dataset to process (1, 2)"
    )

    args = parser.parse_args()
    
    # Test input
    image_path ="evaluation/input_3.png"
    my_model, my_device = load_inference_model()
    
    # Determine input and output directories based on dataset choice
    if args.dataset == "1":
        raw_input_dir = "dataset/raw/1k_pbm"
        processed_output_dir = "dataset/processed/1k_pbm"
        segmented_output_dir = "dataset/segmented/1k_pbm"
        preprocessing_strategy = Dataset1PreprocessingStrategy()
        segmentation_strategy = Dataset1SegmentationStrategy(
            min_area=50, 
            char_size=(28, 28)
        )
    elif args.dataset == "2":
        raw_input_dir = "dataset_v2/raw"
        processed_output_dir = "dataset_v2/processed"
        segmented_output_dir = "dataset_v2/segmented"
        metadata_dir = "dataset_v2/meta"
        ready_output_dir = "dataset_v2/ready"
        preprocessing_strategy = Dataset2PreprocessingStrategy(
            denoise_method="median",
            threshold_method="adaptive",
            line_removal_kernel_size=3,
            denoising_kernel_size=3
        )
        segmentation_strategy = Dataset2SegmentationStrategy(
            min_area=30,  # Lower threshold for noisy images
            char_size=(28, 28),
            min_aspect_ratio=0.1,  # More lenient aspect ratio
            max_aspect_ratio=3.0,
            min_height=5  # Lower minimum height
        )


    if args.stage == "0":
        # 1. Data layer
        if args.dataset == "1":
            process_1k_pbm_dataset()
        elif args.dataset == "2":
            prepare_dataset_v2()

        # 2. Preprocessing
        if args.dataset == "1":
            run_preprocessing_pipeline(input_dir=raw_input_dir, output_dir=processed_output_dir)
        elif args.dataset == "2":
            run_preprocessing_pipeline_v2(input_dir=raw_input_dir, output_dir=processed_output_dir, strategy=preprocessing_strategy)
        
        # 3. Segmentation
        if args.dataset == "1":
            run_segmentation_pipeline(input_dir=processed_output_dir, output_dir=segmented_output_dir, strategy=segmentation_strategy)
        elif args.dataset == "2":
            run_segmentation_pipeline_v2(input_dir=processed_output_dir, output_dir=segmented_output_dir, strategy=segmentation_strategy)

        # 4. Build dataset
        if args.dataset == "1":
            run_dataset_pipeline()
        elif args.dataset == "2":
            run_dataset_pipeline(
                metadata_path=metadata_dir,
                segmented_dir=segmented_output_dir,
                output_dir=ready_output_dir,
            )

        # 5. Train model
        run_training_pipeline()
        
        # 6. Inference Demo
        run_inference_demo()
        
        # 7. Evaluation
        evaluate_system()
        
    elif args.stage == "1":
        # Data layer for each dataset
        if args.dataset == "1":
            process_1k_pbm_dataset()
        elif args.dataset == "2":
            prepare_dataset_v2()
    elif args.stage == "2":
        if args.dataset == "1":
            run_preprocessing_pipeline(input_dir=raw_input_dir, output_dir=processed_output_dir)
        elif args.dataset == "2":
            run_preprocessing_pipeline_v2(input_dir=raw_input_dir, output_dir=processed_output_dir, strategy=preprocessing_strategy)
    elif args.stage == "3":
        if args.dataset == "1":
            run_segmentation_pipeline(input_dir=processed_output_dir, output_dir=segmented_output_dir)
        elif args.dataset == "2":
            run_segmentation_pipeline_v2(input_dir=processed_output_dir, output_dir=segmented_output_dir, strategy=segmentation_strategy)
    elif args.stage == "4":
        if args.dataset == "1":
            run_dataset_pipeline()
        elif args.dataset == "2":
            run_dataset_pipeline(
                metadata_path=metadata_dir,
                segmented_dir=segmented_output_dir,
                output_dir=ready_output_dir,
            )
    elif args.stage == "5":
        run_training_pipeline()
    elif args.stage == "6":
        pred_text, chars = predict_captcha(img_path=image_path, model=my_model, device=my_device)
        print(f"Predicted CAPTCHA text: {pred_text}")
    elif args.stage == "7":
        evaluate_system()


if __name__ == "__main__":
    main()