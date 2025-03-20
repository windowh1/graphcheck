import argparse
from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Optional, TextIO


def parse_args(
) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
        help="Input JSON file path"
    )
    parser.add_argument("--output_path", type=str, default=None,
        help="Output txt file path"
    )
    args = parser.parse_args()

    return args


def print_evaluation(
    prediction_list: List[str], 
    label_list: List[str], 
    file: Optional[TextIO] = None
) -> None:
    
    label_map = {"NOT_SUPPORTED": 0, "SUPPORTED": 1}

    labels = [label_map[e] for e in label_list]
    prediction_list = [label_map[e] for e in prediction_list]
    
    unique_labels = sorted(set(labels + prediction_list))
    target_names = [key for key, val in label_map.items() if val in unique_labels]
    target_names.sort(key=lambda x: label_map[x])

    print(file=file)
    print(
        classification_report(
            labels, 
            prediction_list, 
            target_names=target_names, 
            digits=4, 
            zero_division=0, 
            labels=unique_labels
        ), 
        file=file
    )
    print(file=file)
    
    matrix = confusion_matrix(labels, prediction_list, labels=unique_labels)
    row_sum = np.sum(matrix, axis=1).tolist()
    column_sum = np.sum(matrix, axis=0).tolist()
    column_sum.append(sum(column_sum))
    
    header = ["", *target_names, "SUB_TOTAL"]
    formatted_matrix = [
        [target_names[i], *matrix[i], row_sum[i]] for i in range(len(target_names))
    ]
    formatted_matrix.append(["SUB_TOTAL", *column_sum])

    max_widths = [
        max(len(str(row[i])) for row in [header] + formatted_matrix)
        for i in range(len(header))
    ]

    header_line = "\t".join(str(val).rjust(max_widths[i]) for i, val in enumerate(header))
    print(header_line, file=file)

    for row in formatted_matrix:
        formatted_line = "\t".join(str(val).rjust(max_widths[i]) for i, val in enumerate(row))
        print(formatted_line, file=file)


def print_evaluation_by_hop(
    prediction_list: List[str], 
    label_list: List[str], 
    num_hops_list: List[int], 
    file: Optional[TextIO] = None
) -> None:

    prediction_dict = defaultdict(list)
    label_dict = defaultdict(list)
    
    for pred, gt_label, num_hops in zip(prediction_list, label_list, num_hops_list):
        key = f"{num_hops}_hop"
        
        prediction_dict[key].append(pred.strip())
        label_dict[key].append(gt_label.strip()) 
        
        prediction_dict["total"].append(pred.strip())
        label_dict["total"].append(gt_label.strip())
    
    for key, _ in sorted(prediction_dict.items()):
        print(f"\n<{key}>", file=file)
        print_evaluation(prediction_dict[key], label_dict[key], file=file)
        print("\n", "-" * 52, file=file)


def make_evaluation_file(
    input_path: str, 
    output_path: str
) -> None:
    
    with open(input_path, "r") as f:
        results = json.load(f)
    
    prediction_list, label_list, num_hops_list = [], [], []
    for sample in results:
        prediction_list.append(sample["prediction"])
        label_list.append(sample["label"])
        num_hops_list.append(sample["num_hops"])
    
    if not output_path:
        output_path = input_path.replace(".json", ".txt")
    with open(output_path, "w") as f:
        print_evaluation_by_hop(prediction_list, label_list, num_hops_list, f)


if __name__ == "__main__":
    args = parse_args()
    make_evaluation_file(args.input_path, args.output_path)
