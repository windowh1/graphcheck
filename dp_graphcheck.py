import argparse
import json
import logging
import os
from typing import Any, Dict, List

from graphcheck import GraphCheck

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)


def parse_args(
) -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--input_filename", type=str, 
        help="Input JSON filename"
    )    
    parser.add_argument("--direct_filename", type=str, default=None, 
        help="Existing direct verification JSON filename"
    )
    parser.add_argument("--graph_filename", type=str, default=None, 
        help="Existing graph JSON filename"
    )
    parser.add_argument("--graphcheck_filename", type=str, default=None, 
        help="Existing graphcheck verification JSON filename"
    )
    parser.add_argument("--output_filename", type=str, default=None, 
        help="Output JSON filename"
    )
    parser.add_argument("--force_new_construction", action="store_true",
        help="Force new graph construction even if the graph file exists"
    )
    parser.add_argument("--openai_api_key", type=str, default=None,
        help="OpenAI api key"
    )
    parser.add_argument("--gpt_model_name", type=str, default="gpt-4o-2024-08-06", 
        help="GPT model version used for graph construction"
    )
    parser.add_argument("--use_openai_batch_api", action="store_true", 
        help="Use OpenAI Batch API for cost-efficient processing (50% lower cost but slower generation)."
    )
    parser.add_argument("--base_model_name", type=str, default="google/flan-t5-xl", 
        help="Hugging Face model identifier"
    )
    parser.add_argument("--base_model_cache_dir", type=str, default="./hf_cache", 
        help="Directory to cache the Hugging Face model"
    )
    parser.add_argument("--setting", type=str, default="open-book",
        choices=["open-book", "open-book+gold", "close-book"], 
        help="Retrieval setting mode - options: [open-book, open-book+gold, close-book]"
    )
    parser.add_argument("--top_k", type=int, default=10, 
        help="Number of retrieved documents"
    )
    parser.add_argument("--path_limit", type=int, default=5, 
        help="Maximum number of identification paths"
    )
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size for verification"
    )
    
    return parser.parse_args()


class DPGraphCheck(GraphCheck):
    def __init__(
        self, 
        args: argparse.Namespace
    ):
        
        super().__init__(args)
        
        if args.direct_filename:
            self.direct_path = os.path.join(
                "results", self.dataset, "verifications", "direct", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.direct_top_k}", args.direct_filename
            )
            
        if args.graphcheck_filename:
            self.graphcheck_path = os.path.join(
                "results", self.dataset, "verifications", "graphcheck", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.top_k}", args.graphcheck_filename
            )
                        
        if args.output_filename:
            self.dp_graphcheck_path = os.path.join(
                "results", self.dataset, "verifications", "dp-graphcheck", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.top_k}", args.output_filename
            )
        else:
            self.dp_graphcheck_path = os.path.join(
                "results", self.dataset, "verifications", "dp-graphcheck", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.top_k}", args.input_filename
            )
    
    
    def run_dp_graphcheck(
        self,
    ) -> List[Dict[str, Any]]:
        
        if self.has_complete_results(self.direct_path):
            logger.info("Direct verification results for all samples are already available. Skipping verification.")
            with open(self.direct_path, "r") as f:
                direct_result_list = json.load(f)
            
        else:
            direct_result_list = self.run_direct()
        
        stage_2_input_indices = {sample["index"] for sample in direct_result_list if sample["prediction"] == "NOT_SUPPORTED"}
        logger.info(f"{len(stage_2_input_indices)}/{len(direct_result_list)} samples proceed stage 2 verification.")
        
        if self.has_complete_results(self.graphcheck_path, stage_2_input_indices):
            logger.info("GraphCheck verification results for all samples are already available. Skipping verification.")
            with open(self.graphcheck_path, "r") as f:
                graphcheck_result_list = json.load(f)
            graphcheck_result_list = [sample for sample in graphcheck_result_list if sample["index"] in stage_2_input_indices]
            
        else:
            graphcheck_result_list = self.run_graphcheck(stage_2_input_indices)
            
        result_dict = {}
        for sample in direct_result_list:
            result = {key : value for key, value in sample.items() if key != "retrieval_info"}
            result["definition_triples"] = None
            result["triples"] = None
            result["verification_process"] = [{
                "path": "direct",
                "path_prediction": sample["prediction"],
                "infilled_definition_triples": None,
                "infilled_triples": None,
                "path_verification": [{
                    "subclaim": sample["claim"],
                    "subclaim_prediction": sample["prediction"],
                    "retrieval_info": sample["retrieval_info"]
                }]
            }]
            result_dict[sample["index"]] = result
        
        for sample in graphcheck_result_list:
            result_dict[sample["index"]]["definition_triples"] = sample["definition_triples"]
            result_dict[sample["index"]]["triples"] = sample["triples"]
            result_dict[sample["index"]]["verification_process"].extend(sample["verification_process"])
            result_dict[sample["index"]]["prediction"] = sample["prediction"]
            result_dict[sample["index"]]["is_correct"] = sample["is_correct"]
            
        result_list = [result_dict[index] for index in sorted(result_dict.keys())]            

        return result_list


if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        logger.info(f"- {key}: {value}")
    
    verifier = DPGraphCheck(args)
    result_list = verifier.run_dp_graphcheck()
    verifier.save_result(result_list, verifier.dp_graphcheck_path)
    
    print(f"\n* DP-GraphCheck verification results for {len(result_list)} samples *\n")
    verifier.print_result(result_list)
