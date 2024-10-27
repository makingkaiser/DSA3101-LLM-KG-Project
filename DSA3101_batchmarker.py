import os  
import json  
import glob  
from typing import Dict  
import numpy as np  
from collections import defaultdict  
from DSA3101_marker import evaluate_model_output  # Assuming evaluate_model_output is as provided above  
  
def evaluate_folders(ground_truth_folder: str, prediction_folder: str) -> Dict:  
    """  
    Evaluate all pairs of ground truth and prediction files across two folders, averaging metrics across all pairs.  
      
    Args:  
        ground_truth_folder (str): Path to the folder containing the ground truth JSON files  
        prediction_folder (str): Path to the folder containing the prediction JSON files  
          
    Returns:  
        Dict: Averaged metrics  
    """  
    # Get all ground truth files  
    ground_truth_files = glob.glob(os.path.join(ground_truth_folder, "response_[0-9]*.json"))  
      
    # Initialize accumulators for metrics  
    metrics_accumulator = defaultdict(list)  
    num_evaluated_pairs = 0  
    failed_evaluations = []  
  
    # Process each pair of files  
    for gt_file in sorted(ground_truth_files):  
        try:  
            # Extract file number and construct prediction file name  
            file_number = gt_file.split("response_")[-1].split(".json")[0]  
            pred_file = os.path.join(prediction_folder, f"response_{file_number}_evaluated.json")  
  
            # Check if prediction file exists  
            if not os.path.exists(pred_file):  
                print(f"Prediction file not found for {gt_file}")  
                failed_evaluations.append(gt_file)  
                continue  
  
            # Load ground truth and prediction data  
            with open(gt_file, 'r') as f:  
                ground_truth = json.load(f)  
            with open(pred_file, 'r') as f:  
                prediction = json.load(f)  
  
            # Evaluate the pair  
            result = evaluate_model_output(ground_truth, prediction)  
  
            # Accumulate metrics  
            for metric, value in result["overall_metrics"].items():  
                metrics_accumulator[metric].append(value)  
  
            num_evaluated_pairs += 1  
                  
        except Exception as e:  
            print(f"Failed to evaluate {gt_file} and {pred_file}: {e}")  
            failed_evaluations.append(gt_file)  
  
    # Calculate average metrics across all pairs  
    averaged_metrics = {metric: np.mean(values) for metric, values in metrics_accumulator.items()}  
  
    return {  
        "averaged_metrics": averaged_metrics,  
        "num_evaluated_pairs": num_evaluated_pairs,  
        "failed_evaluations": failed_evaluations  
    }  
  
# Example usage  
if __name__ == "__main__":  
    ground_truth_folder = "gt"  
    prediction_folder = "prediction"  
    results = evaluate_folders(ground_truth_folder, prediction_folder)  
    print(results)  
