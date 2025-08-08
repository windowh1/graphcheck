import argparse
import json
import logging
import os
import random
import re
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import *

from direct import Direct
from model_library.base_model import BaseModel
from model_library.construct_model import ConstructModel
from utils.graph import Graph

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True
)
logger = logging.getLogger(__name__)
random.seed(26)


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
    parser.add_argument("--force_new_construction", action="store_true",
        help="Force new graph construction even if the graph file is already available"
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


class GraphCheck(Direct):
    def __init__(
        self, 
        args: argparse.Namespace,
    ):
        super().__init__(args)
        
        self.force_new_construction = args.force_new_construction
        self.construct_model_name = args.construct_model_name
        self.api_key = args.api_key
        self.construct_batch_size = args.construct_batch_size
        
        if args.graph_filename:
            graph_filename = args.graph_filename
        else:
            graph_filename = args.input_filename
        self.graph_path = os.path.join(
            "results", self.dataset, "graphs", self.construct_model_name, graph_filename
        )
        
        self.path_limit = args.path_limit
        self.document_level = args.document_level
        
        if args.graphcheck_filename:
            graphcheck_filename = args.graphcheck_filename
        else:
            graphcheck_filename = args.input_filename
        self.graphcheck_path = os.path.join(
            "results", self.dataset, "verifications", "graphcheck", self.base_model_name.split("/")[-1], 
            self.setting, graphcheck_filename
        )


    def get_infilling_retrieval_query(
        self, 
        graph: Graph, 
        target_la_ent: str
    ) -> str:
        """
        Construct a retrieval query for latent entity infilling.
        
        The retrieval query is formed by concatenating all triples (except the definition triple) 
        that include the target latent entity and exclude any unidentified latent entities. 
        Each latent entity in the query is replaced with its corresponding reference mapped in the definition triples, 
        forming a nearly complete sentence.
        
        ----------
        Example:
        
        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] a musician
        (ENT2) [SEP] is [SEP] a band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington
        
        [Infilling retrieval query for (ENT1)]
        a musician is part of Tall Birds. 
        """

        sub_graph = [f"{triple.sentence}." for triple in graph.la_ent_2_sub_triples[target_la_ent]]
        query = " ".join(
            [triple_sent for triple_sent in sub_graph if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {target_la_ent}]
        )
        
        # Handle edge case where no relevant triples exist
        if query == "":
            query = f"{graph.la_ent_2_def_triple[target_la_ent].sentence}."
            
        while re.search(r"\(ENT\d+\)", query):
            for la_ent, definition in graph.la_ent_2_def.items():
                query = query.replace(la_ent, definition)
            if graph.has_la_ent_w_no_def == 1: # Edge case
                break
        
        return query
    
    
    def get_infilling_query(
        self, 
        graph: Graph, 
        target_la_ent: str
    ) -> str:
        """
        Construct an infilling query for latent entity infilling.
        
        The infilling query is formed by concatenating all triples that include the target latent entity exclude any other unidentified latent entities. 
        The target latent entity is replaced with the special token to indicate that it should be infilled.
        
        ----------
        Example:
        
        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] a musician
        (ENT2) [SEP] is [SEP] a band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington
        
        [Infilling query for (ENT1)]
        <extra_id_0> is part of Tall Birds. <extra_id_0> is a musician.        
        """
        
        sub_graph = [f"{triple.sentence}." for triple in graph.la_ent_2_sub_triples[target_la_ent]]
        sub_graph.append(f"{graph.la_ent_2_def_triple[target_la_ent].sentence}.")

        query = " ".join(
            [triple_sent for triple_sent in sub_graph if set(re.findall(r"\(ENT\d+\)", triple_sent)) == {target_la_ent}]
        )

        # Handle edge case where no relevant triples exist
        if query == "":
            query = f"{graph.la_ent_2_def_triple[target_la_ent].sentence}."
        
        variable_name = "<extra_id_0>"
        query = query.strip().replace(target_la_ent, variable_name)
        
        while re.search(r"\(ENT\d+\)", query):
            for la_ent, definition in graph.la_ent_2_def.items():
                query = query.replace(la_ent, definition)
            if graph.has_la_ent_w_no_def == 1: # Edge case
                break                   
        
        return query
    
    
    def infill_graph(
        self, 
        graph: Graph, 
        path: List[str], 
        gold_id_list: List[str], 
        gold_evidence: str,
        claim: str
    ) -> Graph:
        
        infilled_def_triple_sents = [def_triple.triple_sent for def_triple in graph.def_triples]
        infilled_triple_sents = [triple.triple_sent for triple in graph.triples]
        infilling_log = []
        for i, target_la_ent in enumerate(path):
            retrieval_query = self.get_infilling_retrieval_query(graph, target_la_ent)
            evidence_list, retrieval_info = self.retrieve_evidence(retrieval_query, gold_id_list, gold_evidence)
            
            evidence = self.truncate('\n'.join(evidence_list))    
            
            infilling_query = self.get_infilling_query(graph, target_la_ent)
            answer = self.base_model.infill(infilling_query, evidence)
            
            if not answer.strip():
                logger.error(f"No answer generated for {target_la_ent} in path {path}.")
                answer = graph.la_ent_2_def[target_la_ent]
            else:
                answer = answer.split("\n")[0].strip()
            
            infilled_def_triple_sents = [
                sent.replace(target_la_ent, answer) for sent in infilled_def_triple_sents
            ]
            infilled_triple_sents = [
                sent.replace(target_la_ent, answer) for sent in infilled_triple_sents
            ]
            remained_def_triple_sents = [
                sent for sent in infilled_def_triple_sents if re.search(r"\(ENT\d+\)", sent.split()[0])
            ]
            if remained_def_triple_sents:
                graph = Graph(remained_def_triple_sents, infilled_triple_sents)
            infilling_log.append({
                "infilling_index": i, 
                "target_latent_entity": target_la_ent, 
                "infilling_query": infilling_query, 
                "infilling_answer": answer, 
                "infilling_retrieval_info": {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None
            })
        
        return Graph(infilled_def_triple_sents, infilled_triple_sents), infilling_log


    def verify_graph(
        self, 
        graph: Graph, 
        gold_id_list: List[str], 
        gold_evidence: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        
        path_prediction = "SUPPORTED"
        path_verification = []
        
        for triple in graph.total_triples:
            subclaim_prediction, retrieval_info, supported_doc_id = self.verify_claim(
                triple.sentence, gold_id_list, gold_evidence, self.top_k, self.document_level
            )
            
            path_verification.append({
                "subclaim": triple.sentence,
                "subclaim_prediction": subclaim_prediction,
                "retrieval_info": {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None,
                "supported_doc_id": supported_doc_id
            })
            
            if subclaim_prediction == "NOT_SUPPORTED":
                path_prediction = "NOT_SUPPORTED"
                break
        
        return path_prediction, path_verification

    
    def process_sample_graphcheck(
        self, 
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:        
        
        try:
            prediction = "NOT_SUPPORTED"
            verification_process = []
            
            graph = Graph(sample["definition_triples"], sample["triples"])
            
            if graph.num_la_ent > 0:
                path_list = graph.get_valid_paths(self.path_limit)
            else:
                path_list = [[]]
            
            for path_idx, path in enumerate(path_list):
                infilled_graph, infilling_log = self.infill_graph(graph, path, sample["gold_id_list"], sample["gold_evidence"], sample["claim"])
                path_prediction, path_verification = self.verify_graph(infilled_graph, sample["gold_id_list"], sample["gold_evidence"])
                
                path_info = f"{' - '.join(path)}" if path else None
                    
                verification_process.append({
                    "path_index": path_idx,
                    "path": path_info,
                    "path_prediction": path_prediction,
                    "infilled_definition_triples": infilled_graph.def_triple_sents,
                    "infilled_triples": infilled_graph.triple_sents,
                    "infilling_log": infilling_log,
                    "path_verification": path_verification
                })
                
                if path_prediction == "SUPPORTED":
                    prediction = "SUPPORTED"
                    break
            
            sample.update({
                "prediction": prediction,
                "is_correct": (prediction == sample["label"]),
                "verification_process": verification_process
            })
        
        except Exception as e:
            logger.warning(f"Failed to verify sample. Skipping this sample. Error: {e}")
            
            prediction = random.choice(["SUPPORTED", "NOT_SUPPORTED"])
            sample.update({
                "prediction": prediction,
                "is_correct": (prediction == sample["label"]),
                "verification_process": None
            })
        
        return sample
    
    
    def run_graphcheck(
        self,
        input_indices: Set[int] = None
    ) -> List[Dict[str, Any]]:

        if self.has_complete_results(self.graph_path, input_indices) and not self.force_new_construction:
            logger.info(f"Graphs for all samples are already available. Skipping construction.")

            with open(self.graph_path, "r") as f:
                graph_list = json.load(f)
            if input_indices:
                graph_list = [sample for sample in graph_list if sample["index"] in input_indices]
        
        else:
            self.construct_model = ConstructModel(
                self.construct_model_name,
                self.api_key,
                self.cache_dir,
                self.construct_batch_size
            )
            with open(self.input_path, "r") as f:
                input_list = json.load(f)
            if input_indices:
                input_list = [sample for sample in input_list if sample["index"] in input_indices]
            
            graph_list = self.construct_model.construct_graph(input_list, self.graph_path, self.force_new_construction)
        

        logger.info("Starting GraphCheck verification...")

        if self.base_model is None:
            tokenizer = T5Tokenizer.from_pretrained(self.base_model_name, cache_dir=self.cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, cache_dir=self.cache_dir, device_map="auto")
            self.base_model = BaseModel(model, tokenizer)
            logger.info(f"Base model '{self.base_model_name}' initialized successfully.")        
            
        result_list = []
        for i, sample in enumerate(graph_list):
            graph_list[i] = {
                "index": sample["index"],
                "uid": sample["uid"],
                "num_hops": sample["num_hops"],
                "gold_id_list": sample["gold_id_list"],
                "gold_evidence": sample["gold_evidence"],
                "claim": sample["claim"],
                "label": sample["label"],
                "prediction": None,
                "is_correct": None,
                "definition_triples": sample["definition_triples"],
                "triples": sample["triples"],
                "verification_process": None
            }
        
        for i, sample in enumerate(tqdm(graph_list)):
            result = self.process_sample_graphcheck(sample)
            result_list.append(result)
                
        logger.info("GraphCheck verification completed!")
        
        return result_list


if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        logger.info(f"- {key}: {value}")
    
    verifier = GraphCheck(args)
    result_list = verifier.run_graphcheck()
    verifier.save_result(result_list, verifier.graphcheck_path)
    
    print(f"\n* GraphCheck verification results for {len(result_list)} samples *\n")
    verifier.print_result(result_list)
