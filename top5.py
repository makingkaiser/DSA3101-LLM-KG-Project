import os
import json
import glob
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from DSA3101_marker import evaluate_model_output

def find_top_evaluations(ground_truth_folder: str, prediction_folder: str, top_n: int = 5) -> List[Dict]:
    """
    Evaluate all pairs of files and identify the top N evaluations based on individual metrics.
    
    Args:
        ground_truth_folder (str): Path to the folder containing the ground truth JSON files
        prediction_folder (str): Path to the folder containing the prediction JSON files
        top_n (int): Number of top evaluations to return (default: 5)
        
    Returns:
        List[Dict]: List of dictionaries containing the top N evaluations with their metrics and file information
    """
    # Get all ground truth files
    ground_truth_files = glob.glob(os.path.join(ground_truth_folder, "response_[0-9]*.json"))
    
    # List to store all evaluation results
    all_evaluations = []

    # Process each pair of files
    for gt_file in sorted(ground_truth_files):
        try:
            # Extract file number and construct prediction file name
            file_number = gt_file.split("response_")[-1].split(".json")[0]
            pred_file = os.path.join(prediction_folder, f"response_{file_number}_evaluated.json")

            # Check if prediction file exists
            if not os.path.exists(pred_file):
                print(f"Prediction file not found for {gt_file}")
                continue

            # Load ground truth and prediction data
            with open(gt_file, 'r') as f:
                ground_truth = json.load(f)
            with open(pred_file, 'r') as f:
                prediction = json.load(f)

            # Evaluate the pair
            result = evaluate_model_output(ground_truth, prediction)
            
            # Calculate individual scores for each category
            category_scores = {
                'entities': {
                    entity_type: metrics['f1_score']
                    for entity_type, metrics in result['entity_metrics'].items()
                },
                'relationships': result['relationship_metrics']['f1_score'],
                'overall': result['overall_metrics']['f1_score']
            }
            
            # Store evaluation result with file information
            evaluation_entry = {
                'file_number': file_number,
                'ground_truth_file': gt_file,
                'prediction_file': pred_file,
                'scores': category_scores,
                'detailed_metrics': result
            }
            
            all_evaluations.append(evaluation_entry)
                
        except Exception as e:
            print(f"Failed to evaluate {gt_file}: {e}")

    # Sort evaluations by different metrics and get top N for each category
    top_evaluations = {
        'overall': sorted(all_evaluations, 
                         key=lambda x: x['scores']['overall'], 
                         reverse=True)[:top_n],
        'relationships': sorted(all_evaluations, 
                              key=lambda x: x['scores']['relationships'], 
                              reverse=True)[:top_n],
        'entities': {
            entity_type: sorted(all_evaluations, 
                              key=lambda x: x['scores']['entities'][entity_type], 
                              reverse=True)[:top_n]
            for entity_type in ['persons', 'organizations', 'roles', 'products_services']
        }
    }

    return {
        'top_evaluations': top_evaluations,
        'total_evaluated': len(all_evaluations),
        'summary': {
            'overall_mean_f1': np.mean([eval['scores']['overall'] for eval in all_evaluations]),
            'relationship_mean_f1': np.mean([eval['scores']['relationships'] for eval in all_evaluations]),
            'entity_mean_f1': {
                entity_type: np.mean([eval['scores']['entities'][entity_type] for eval in all_evaluations])
                for entity_type in ['persons', 'organizations', 'roles', 'products_services']
            }
        }
    }

# Example usage
if __name__ == "__main__":
    ground_truth_folder = "minutes_ground_truth"
    prediction_folder = "minutes_ground_truth_prediction"
    results = find_top_evaluations(ground_truth_folder, prediction_folder)
    
    # Print top overall performers
    print("\nTop Overall Performers:")
    for idx, eval_result in enumerate(results['top_evaluations']['overall'], 1):
        print(f"\n{idx}. File Number: {eval_result['file_number']}")
        print(f"   Overall F1 Score: {eval_result['scores']['overall']:.3f}")
        print(f"   Relationship F1 Score: {eval_result['scores']['relationships']:.3f}")
        print("   Entity F1 Scores:")
        for entity_type, score in eval_result['scores']['entities'].items():
            print(f"      - {entity_type}: {score:.3f}")