import argparse
import json
import logging
import os
from typing import *

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
        help="Output JSON filename for Direct verification results"
    )
    parser.add_argument("--graph_filename", type=str, default=None, 
        help="Output JSON filename for graph construction results"
    )    
    parser.add_argument("--graphcheck_filename", type=str, default=None, 
        help="Output JSON filename for GraphCheck verification results"
    )
    parser.add_argument("--dp_graphcheck_filename", type=str, default=None, 
        help="Output JSON filename for DP-GraphCheck verification results"
    )
    parser.add_argument("--force_new_construction", action="store_true",
        help="Force new graph construction even if the graph file exists"
    )
    parser.add_argument("--construct_model_name", type=str, default="gpt-4o-2024-08-06", 
        help=(
            "Model version used for graph construction.\n"
            "Recommended options: [gpt-4o-2024-08-06, claude-3-7-sonnet-20250219, Qwen/Qwen2.5-72B-Instruct]"
        )
    )
    parser.add_argument("--api_key", type=str, default=None,
        help="API key required for using OpenAI (GPT) or Anthropic (Claude) models during graph construction."
    )
    parser.add_argument("--construct_batch_size", type=int, default=1,
        help="Batch size for graph construction. Set greater than 1 to use OpenAI Batch APIs for cost-efficient processing (up to 50% cheaper but slower generation)."
    )
    parser.add_argument("--base_model_name", type=str, default="google/flan-t5-xl", 
        help="Model used for Direct, text infilling, triple verification and strategy selection."
    )
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", 
        help="Directory to cache the Hugging Face model"
    )
    parser.add_argument("--setting", type=str, default="open-book",
        choices=["open-book", "open-book+gold"], 
        help="Retrieval setting mode - options: [open-book, open-book+gold]"
    )
    parser.add_argument("--document_level", type=str, default="concat+each",
        choices=["concat", "each", "concat+each"], 
        help="Document-level setting mode - options: [concat, each, concat+each]"
    )
    parser.add_argument("--top_k", type=int, default=10, 
        help="Number of retrieved documents"
    )
    parser.add_argument("--path_limit", type=int, default=5, 
        help="Maximum number of identification paths"
    )
    
    return parser.parse_args()


class DPGraphCheck(GraphCheck):
    def __init__(
        self, 
        args: argparse.Namespace
    ):
        
        super().__init__(args)
        
        if args.dp_graphcheck_filename:
            dp_graphcheck_filename = args.dp_graphcheck_filename
        else:
            dp_graphcheck_filename = args.input_filename
        self.dp_graphcheck_path = os.path.join(
            "results", self.dataset, "verifications", "dp-graphcheck", self.base_model_name.split("/")[-1], 
            self.setting, dp_graphcheck_filename
        )
    
    
    def run_dp_graphcheck(
        self,
    ) -> List[Dict[str, Any]]:
        
        do_direct_w_strategy_selection = not self.has_complete_results(self.direct_path)
        result_list, direct_indices, graphcheck_indices = self.run_strategy_selector(do_direct_w_strategy_selection)
        
        if do_direct_w_strategy_selection:
            direct_result_list = [sample for sample in result_list if sample["index"] in direct_indices]            
        
        else:
            with open(self.direct_path, "r") as f:
                direct_result_list = json.load(f)            
            direct_result_list = [sample for sample in direct_result_list if sample["index"] in direct_indices]
                    
        if self.has_complete_results(self.graphcheck_path, graphcheck_indices):
            with open(self.graphcheck_path, "r") as f:
                graphcheck_result_list = json.load(f)
            graphcheck_result_list = [sample for sample in graphcheck_result_list if sample["index"] in graphcheck_indices]
            
        else:
            graphcheck_result_list = self.run_graphcheck(graphcheck_indices)
            
        result_dict = {sample["index"]: sample for sample in result_list}
        
        for sample in direct_result_list:
            result_dict[sample["index"]]["prediction"] = sample["prediction"]
            result_dict[sample["index"]]["is_correct"] = sample["is_correct"]
            result_dict[sample["index"]]["definition_triples"] = None
            result_dict[sample["index"]]["triples"] = None
            result_dict[sample["index"]]["verification_process"] = [{
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
            result_dict[sample["index"]].pop("retrieval_info", None)
            result_dict[sample["index"]].pop("supported_doc_id", None)
        
        for sample in graphcheck_result_list:
            result_dict[sample["index"]]["prediction"] = sample["prediction"]
            result_dict[sample["index"]]["is_correct"] = sample["is_correct"]
            result_dict[sample["index"]]["definition_triples"] = sample["definition_triples"]
            result_dict[sample["index"]]["triples"] = sample["triples"]
            result_dict[sample["index"]]["verification_process"] = sample["verification_process"]

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
