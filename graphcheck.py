import argparse
import json
import logging
from openai import OpenAI
import os
import random
import re
from tqdm import tqdm
from typing import Any, Dict, List, Set, Tuple

from direct import Direct
from model_library.construct_model import ConstructModel
from utils.graph import Graph

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
    parser.add_argument("--graph_filename", type=str, default=None, 
        help="Existing graph JSON filename"
    )
    parser.add_argument("--output_filename", type=str, default=None, 
        help="Output JSON filename"
    )
    parser.add_argument("--force_new_construction", action="store_true",
        help="Force new graph construction even if the graph file is already available"
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


class GraphCheck(Direct):
    def __init__(
        self, 
        args: argparse.Namespace,
    ):
        super().__init__(args)
        
        self.force_new_construction = args.force_new_construction
        self.openai_api_key = args.openai_api_key
        self.gpt_model_name = args.gpt_model_name
        self.use_openai_batch_api = args.use_openai_batch_api
        
        if args.graph_filename:
            self.graph_path = os.path.join(
                "results", self.dataset, "graphs", self.gpt_model_name, args.graph_filename
            )
        else:
            self.graph_path = os.path.join(
                "results", self.dataset, "graphs", self.gpt_model_name, args.input_filename
            )
        
        if self.setting.startswith("open-book"):
            self.top_k = args.top_k
        else:
            self.top_k = 0
        
        self.path_limit = args.path_limit
        
        if args.output_filename:
            self.graphcheck_path = os.path.join(
                "results", self.dataset, "verifications", "graphcheck", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.top_k}", args.output_filename
            )
        else:
            self.graphcheck_path = os.path.join(
                "results", self.dataset, "verifications", "graphcheck", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.top_k}", args.input_filename
            )


    def has_complete_results(
        self,
        file_path: str,
        input_indices: Set[int] = None
    ) -> bool:
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                result_list = json.load(f)
            result_indices = {sample["index"] for sample in result_list}

            if not input_indices:
                with open(self.input_path, "r") as f:
                    input_list = json.load(f)
                input_indices = {sample["index"] for sample in input_list}
            
            missing_indices = input_indices - result_indices

            if not missing_indices:
                return True
        
        return False
    
    
    def get_infilling_retrieval_query(
        self, 
        graph: Graph, 
        target_la_ent: str
    ) -> str:
        """
        Construct a retrieval query for latent entity infilling.
        
        The retrieval query is formed by concatenating all triples (except the definition triple) 
        that include the target latent entity to be infilled. 
        Each latent entity in the query is replaced with its corresponding reference mapped in the definition triples, 
        forming a nearly complete sentence.
        
        ----------
        Example:
        
        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] the musician
        (ENT2) [SEP] is [SEP] the band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington
        
        [Infilling retrieval query for (ENT1)]
        the musician is part of Tall Birds. the musician is a percussionist for the band.
        
        """
        
        sub_graph = graph.la_ent_2_sub_triples[target_la_ent]
        query = " ".join([f"{triple.sentence}." for triple in sub_graph])
        
        # Handle edge case where no relevant triples exist
        if query == "":
            query = graph.la_ent_2_def_triple[target_la_ent].sentence
            
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
        
        The infilling query is formed by concatenating all triples that include the target latent entity to be infilled. 
        The target latent entity is replaced with the special token to indicate that it should be infilled,
        while all other latent entities in the query are replaced with their corresponding references mapped in the definition triples.
        
        ----------
        Example:
        
        [Graph]
        # Latent Entities:
        (ENT1) [SEP] is [SEP] the musician
        (ENT2) [SEP] is [SEP] the band
        # Triples:
        (ENT1) [SEP] is part of [SEP] Tall Birds
        (ENT1) [SEP] is a percussionist for [SEP] (ENT2)
        (ENT2) [SEP] formed in [SEP] Issaquah, Washington
        
        [Infilling query for (ENT1)]
        <extra_id_0> is part of Tall Birds. <extra_id_0> is a percussionist for the band. <extra_id_0> is the musician.
        
        """
        
        sub_graph = graph.la_ent_2_sub_triples[target_la_ent]
        query = " ".join([f"{triple.sentence}." for triple in sub_graph])
        
        variable_name = "<extra_id_0>"
        query += f" {graph.la_ent_2_def_triple[target_la_ent].sentence}."
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
        gold_evidence: str
    ) -> Graph:
        
        infilled_def_triple_sents = [def_triple.triple_sent for def_triple in graph.def_triples]
        infilled_triple_sents = [triple.triple_sent for triple in graph.triples]
        
        for target_la_ent in path:
            retrieval_query = self.get_infilling_retrieval_query(graph, target_la_ent)
            if self.setting.startswith("open-book"):
                evidence, _ = self.retrieve_evidence(retrieval_query, gold_id_list, gold_evidence, self.top_k)
            else:
                evidence, _ = None, None
            
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
        
        return Graph(infilled_def_triple_sents, infilled_triple_sents)


    def batch_infill_graph(
        self, 
        batch_graphs: List[Graph], 
        batch_paths: List[List[str]], 
        batch_gold_id_lists: List[List[str]], 
        batch_gold_evidences: List[str]
    ) -> List[Graph]:
    
        batch_infilled_def_triple_sents = [
            [def_triple.triple_sent for def_triple in graph.def_triples] for graph in batch_graphs
        ]
        batch_infilled_triple_sents = [
            [triple.triple_sent for triple in graph.triples] for graph in batch_graphs
        ]
        
        max_path_len = max(len(path) for path in batch_paths)
        for la_ent_idx in range(max_path_len):
            batch_indices, batch_la_ents = [], []
            batch_retrieval_queries = []
            
            for batch_idx, (graph, path) in enumerate(zip(batch_graphs, batch_paths)):
                if la_ent_idx >= len(path):  # Skip if this graph has no more target entities
                    continue
                batch_indices.append(batch_idx)
                target_la_ent = path[la_ent_idx]
                batch_la_ents.append(target_la_ent)
                batch_retrieval_queries.append(self.get_infilling_retrieval_query(graph, target_la_ent))
            
            if batch_retrieval_queries and self.setting.startswith("open-book"):
                batch_evidences, _ = self.batch_retrieve_evidence(
                    batch_retrieval_queries,
                    [batch_gold_id_lists[batch_idx] for batch_idx in batch_indices],
                    [batch_gold_evidences[batch_idx] for batch_idx in batch_indices],
                    self.top_k
                )
            else:
                batch_evidences = None
            
            batch_infilling_queries = []
            for batch_idx, target_la_ent in zip(batch_indices, batch_la_ents):
                graph = batch_graphs[batch_idx]
                batch_infilling_queries.append(self.get_infilling_query(graph, target_la_ent))
            
            batch_answers = self.base_model.batch_infill(batch_infilling_queries, batch_evidences)
            
            for batch_idx, target_la_ent, answer in zip(batch_indices, batch_la_ents, batch_answers):
                graph = batch_graphs[batch_idx]
                
                if not answer.strip():
                    logger.error(f"No answer generated for {target_la_ent} in path {path}.")
                    answer = graph.la_ent_2_def[target_la_ent]
                else:
                    answer = answer.split("\n")[0].strip()
                
                batch_infilled_def_triple_sents[batch_idx] = [
                    sent.replace(target_la_ent, answer) for sent in batch_infilled_def_triple_sents[batch_idx]
                ]
                batch_infilled_triple_sents[batch_idx] = [
                    sent.replace(target_la_ent, answer) for sent in batch_infilled_triple_sents[batch_idx]
                ]
                remained_def_triple_sents = [
                    sent for sent in batch_infilled_def_triple_sents[batch_idx] if re.search(r"\(ENT\d+\)", sent.split()[0])
                ]
                if remained_def_triple_sents:
                    batch_graphs[batch_idx] = Graph(remained_def_triple_sents, batch_infilled_triple_sents[batch_idx])
        
        batch_infilled_graph = [
            Graph(infilled_def_triple_sents, infilled_triple_sents) 
            for infilled_def_triple_sents, infilled_triple_sents in zip(batch_infilled_def_triple_sents, batch_infilled_triple_sents)
        ]
        return batch_infilled_graph
    
    
    def verify_graph(
        self, 
        graph: Graph, 
        gold_id_list: List[str], 
        gold_evidence: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        
        path_prediction = "SUPPORTED"
        path_verification = []
        
        for triple in graph.total_triples:
            subclaim_prediction, retrieval_info = self.verify_claim(
                triple.sentence, gold_id_list, gold_evidence, self.top_k
            )
            
            path_verification.append({
                "subclaim": triple.sentence,
                "subclaim_prediction": subclaim_prediction,
                "retrieval_info": retrieval_info
            })
            
            if subclaim_prediction == "NOT_SUPPORTED":
                path_prediction = "NOT_SUPPORTED"
                break
        
        return path_prediction, path_verification
    
    
    def batch_verify_graph(
        self, 
        batch_graphs: List[Graph], 
        batch_gold_id_lists: List[List[str]], 
        batch_gold_evidences: List[str]
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        
        batch_path_predictions = ["SUPPORTED"] * len(batch_graphs)
        batch_path_verifications = [[] for _ in range(len(batch_graphs))]
        
        max_triple_count = max(len(graph.total_triples) for graph in batch_graphs)
        
        for triple_idx in range(max_triple_count):
            batch_indices = []
            batch_subclaims = []
            
            for batch_idx, graph in enumerate(batch_graphs):
                if triple_idx >= len(graph.total_triples):
                    continue
                if batch_path_predictions[batch_idx] == "NOT_SUPPORTED":
                    continue
                batch_indices.append(batch_idx)
                batch_subclaims.append(graph.total_triples[triple_idx].sentence)
            
            if not batch_subclaims:
                    break
            
            batch_predictions, batch_retrieval_infos = self.batch_verify_claim(
                batch_subclaims,
                [batch_gold_id_lists[batch_idx] for batch_idx in batch_indices],
                [batch_gold_evidences[batch_idx] for batch_idx in batch_indices],
                self.top_k
            )
            if not batch_retrieval_infos:
                batch_retrieval_infos = [None] * len(batch_subclaims)
            
            for batch_idx, subclaim, subclaim_prediction, retrieval_info in zip(
                batch_indices, batch_subclaims, batch_predictions, batch_retrieval_infos
            ):
                batch_path_verifications[batch_idx].append({
                    "subclaim": subclaim,
                    "subclaim_prediction": subclaim_prediction,
                    "retrieval_info": retrieval_info
                })
                if subclaim_prediction == "NOT_SUPPORTED":
                    batch_path_predictions[batch_idx] = "NOT_SUPPORTED"
            
            if all(path_prediction == "NOT_SUPPORTED" for path_prediction in batch_path_predictions):
                break
        
        return batch_path_predictions, batch_path_verifications
    
    
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
                infilled_graph = self.infill_graph(graph, path, sample["gold_id_list"], sample["gold_evidence"])
                path_prediction, path_verification = self.verify_graph(infilled_graph, sample["gold_id_list"], sample["gold_evidence"])
                
                path_info = f"path {path_idx+1}: {' - '.join(path)}" if path else None
                    
                verification_process.append({
                    "path": path_info,
                    "path_prediction": path_prediction,
                    "infilled_definition_triples": infilled_graph.def_triple_sents,
                    "infilled_triples": infilled_graph.triple_sents,
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
    
    
    def process_batch_graphcheck(
        self, 
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        try:
            batch_predictions = ["NOT_SUPPORTED"] * len(batch)
            batch_verification_processes = [[] for _ in range(len(batch))]
            
            batch_graphs = [Graph(sample["definition_triples"], sample["triples"]) for sample in batch]
            
            batch_path_lists = []
            for graph in batch_graphs:
                if graph.num_la_ent > 0:
                    path_list = graph.get_valid_paths(self.path_limit)
                else:
                    path_list = [[]]
                batch_path_lists.append(path_list)
            
            max_path_count = max(len(path_list) for path_list in batch_path_lists)
            
            for path_idx in range(max_path_count):
                sub_batch_indices, sub_batch_paths = [], []
                for batch_idx, path_list in enumerate(batch_path_lists):
                    if path_idx >= len(path_list):
                        continue
                    if batch_predictions[batch_idx] == "SUPPORTED":
                        continue
                    sub_batch_indices.append(batch_idx)
                    sub_batch_paths.append(path_list[path_idx])
                
                if not sub_batch_indices:
                    break
                
                sub_batch_infilled_graph = self.batch_infill_graph(
                    [batch_graphs[batch_idx] for batch_idx in sub_batch_indices],
                    sub_batch_paths, 
                    [batch[batch_idx]["gold_id_list"] for batch_idx in sub_batch_indices],
                    [batch[batch_idx]["gold_evidence"] for batch_idx in sub_batch_indices]
                )
                
                sub_batch_path_predictions, sub_batch_path_verifications = self.batch_verify_graph(
                    sub_batch_infilled_graph, 
                    [batch[batch_idx]["gold_id_list"] for batch_idx in sub_batch_indices],
                    [batch[batch_idx]["gold_evidence"] for batch_idx in sub_batch_indices]
                )
                
                for batch_idx, path, infilled_graph, path_prediction, path_verification in zip(
                    sub_batch_indices, sub_batch_paths, sub_batch_infilled_graph, 
                    sub_batch_path_predictions, sub_batch_path_verifications
                ):
                    path_info = f"path {path_idx+1}: {' - '.join(path)}" if path else None
                    
                    batch_verification_processes[batch_idx].append({
                        "path": path_info,
                        "path_prediction": path_prediction,
                        "infilled_definition_triples": infilled_graph.def_triple_sents,
                        "infilled_triples": infilled_graph.triple_sents,
                        "path_verification": path_verification
                    })
                    
                    if path_prediction == "SUPPORTED":
                        batch_predictions[batch_idx] = "SUPPORTED"
                
                if all(prediction == "SUPPORTED" for prediction in batch_predictions):
                    break
            
            for sample, prediction, verification_process in zip(
                batch, batch_predictions, batch_verification_processes
            ):
                sample.update({
                    "prediction": prediction,
                    "is_correct": (prediction == sample["label"]),
                    "verification_process": verification_process
                })
        
        except Exception as e:
            logger.warning(f"Failed to verify batch. Skipping this batch. Error: {e}")
            
            batch_predictions = [random.choice(["SUPPORTED", "NOT_SUPPORTED"]) for _ in range(len(batch))]
            for sample, prediction in zip(batch, batch_predictions):
                sample.update({
                    "prediction": prediction,
                    "is_correct": (prediction == sample["label"]),
                    "verification_process": None
                })
        
        return batch
    
    
    def run_graphcheck(
        self,
        input_indices: Set[int] = None
    ) -> List[Dict[str, Any]]:

        self.graphs_exist = self.has_complete_results(self.graph_path, input_indices)
        if self.graphs_exist and not self.force_new_construction:
            logger.info(f"Graphs for all samples are already available. Skipping construction.")

            with open(self.graph_path, "r") as f:
                graph_list = json.load(f)
            if input_indices:
                graph_list = [sample for sample in graph_list if sample["index"] in input_indices]
        
        else:
            client = OpenAI(api_key=self.openai_api_key)
            self.construct_model = ConstructModel(client, self.gpt_model_name, self.use_openai_batch_api)

            with open(self.input_path, "r") as f:
                input_list = json.load(f)
            if input_indices:
                input_list = [sample for sample in input_list if sample["index"] in input_indices]
            
            graph_list = self.construct_model.construct_graph(input_list, self.graph_path, self.force_new_construction)
        

        logger.info("Starting GraphCheck verification...")

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
        
        if self.batch_size > 1:
            graph_list = sorted(graph_list, key=lambda x: len(x["definition_triples"]))

            for i in tqdm(range(0, len(graph_list), self.batch_size)):
                batch = graph_list[i : i + self.batch_size]
                batch_results = self.process_batch_graphcheck(batch)
                result_list.extend(batch_results)
                    
        else:
            for sample in tqdm(graph_list):
                result = self.process_sample_graphcheck(sample)
                result_list.append(result)
                
        logger.info("GraphCheck verification completed!")
        
        if self.batch_size > 1:
            result_list = sorted(result_list, key=lambda x: x["index"])
            
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
