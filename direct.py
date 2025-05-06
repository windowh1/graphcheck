import argparse
import json
import logging
import os
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from model_library.base_model import BaseModel
from utils.retriever import PyseriniRetriever
from utils.evaluator import print_evaluation_by_hop

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
    parser.add_argument("--output_filename", type=str, default=None, 
        help="Output JSON filename"
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
    parser.add_argument("--batch_size", type=int, default=1,
        help="Batch size for verification"
    )
    
    return parser.parse_args()


class Direct:
    def __init__(
        self, 
        args: argparse.Namespace,
    ):
        self.dataset = args.dataset
        self.input_path = os.path.join("datasets", self.dataset, "claims", args.input_filename)
                
        self.base_model_name = args.base_model_name
        tokenizer = T5Tokenizer.from_pretrained(self.base_model_name, cache_dir=args.base_model_cache_dir)
        model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, cache_dir=args.base_model_cache_dir, device_map="auto")
        self.base_model = BaseModel(model, tokenizer)
        logger.info(f"Base model '{self.base_model_name}' initialized successfully.")
        
        self.setting = args.setting
        if self.setting.startswith("open-book"):
            retriever_corpus_dir = f"./datasets/{args.dataset}/corpus/index"
            self.retriever = PyseriniRetriever(retriever_corpus_dir, use_bm25=True, k1=0.9, b=0.4)
            self.direct_top_k = 5
        else:
            self.retriever = None
            self.direct_top_k = 0
        self.evidence_max_len = 40000
        
        self.batch_size = args.batch_size
    
        if args.output_filename:
            self.direct_path = os.path.join(
                "results", self.dataset, "verifications", "direct", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.direct_top_k}", args.output_filename
            )
        else:
            self.direct_path = os.path.join(
                "results", self.dataset, "verifications", "direct", self.base_model_name.split("/")[-1], 
                self.setting, f"top_{self.direct_top_k}", args.input_filename
            )
    
    
    def check_retrieval(
        self, 
        doc_id: str, 
        gold_id_list: List[str]
    ) -> int:
        
        return 1 if doc_id in gold_id_list else 0
    
    
    def retrieve_evidence(
        self, 
        query: str, 
        gold_id_list: List[str], 
        gold_evidence: str,
        top_k: int = 5
    ) -> Tuple[str, Dict[str, Any]]:
        
        doc_id_list, score_list, is_gold_list, evidence = [], [], [], []
        hit_list = self.retriever.retrieve(query, top_k)
        
        if self.setting == "open-book+gold":
            doc_id_list.extend(gold_id_list)
            score_list.extend([0] * len(gold_id_list))
            is_gold_list.extend([1] * len(gold_id_list))
            if gold_evidence:  
                evidence.append(gold_evidence.strip())
        
        for hit in hit_list:
            if len(doc_id_list) >= top_k:
                break  
            
            if hit["doc_id"] not in doc_id_list:
                doc_id_list.append(hit["doc_id"])
                score_list.append(hit["score"])
                is_gold_list.append(self.check_retrieval(hit["doc_id"], gold_id_list))
                if hit["text"]:
                    evidence.append(hit["text"].strip())
        
        retrieval_info = {
            "query": query,
            "doc_id_list": doc_id_list, 
            "score_list": score_list, 
            "is_gold_list": is_gold_list 
            }
        
        evidence = "\n".join(evidence)
        if len(evidence) > self.evidence_max_len:            
            evidence = evidence[:self.evidence_max_len]
            logger.warning(f"Evidence length exceeds {self.evidence_max_len} characters and has been truncated to prevent GPU memory overflow.")
        
        return evidence, retrieval_info
    
    
    def batch_retrieve_evidence(
        self, 
        batch_queries: List[str], 
        batch_gold_id_lists: List[List[str]], 
        batch_gold_evidences: List[str],
        top_k: int = 5
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
    
        batch_evidences, batch_retrieval_infos = [], []
        batch_qids = [str(i) for i in range(len(batch_queries))]  # Unique query IDs
        batch_hit_lists = self.retriever.batch_retrieve(batch_queries, batch_qids, top_k)
        
        for i, query in enumerate(batch_queries):
            qid = batch_qids[i]
            gold_id_list = batch_gold_id_lists[i]
            gold_evidence = batch_gold_evidences[i]
            
            doc_id_list, score_list, is_gold_list, evidence = [], [], [], []
            hit_list = batch_hit_lists.get(qid, [])
            
            if self.setting == "open-book+gold":
                doc_id_list.extend(gold_id_list)
                score_list.extend([0] * len(gold_id_list))
                is_gold_list.extend([1] * len(gold_id_list))
                if gold_evidence:
                    evidence.append(gold_evidence.strip())
            
            for hit in hit_list:
                if len(doc_id_list) >= top_k:
                    break  
                
                if hit["doc_id"] not in doc_id_list:
                    doc_id_list.append(hit["doc_id"])
                    score_list.append(hit["score"])
                    is_gold_list.append(self.check_retrieval(hit["doc_id"], gold_id_list))
                    if hit["text"]:
                        evidence.append(hit["text"].strip())
            
            retrieval_info = {
                "query": query,
                "doc_id_list": doc_id_list,
                "score_list": score_list,
                "is_gold_list": is_gold_list
            }
            
            evidence = "\n".join(evidence)
            if len(evidence) > self.evidence_max_len:
                evidence = evidence[:self.evidence_max_len]
                logger.warning(f"Evidence length exceeds {self.evidence_max_len} characters and has been truncated to prevent GPU memory overflow.")
            
            batch_evidences.append(evidence)
            batch_retrieval_infos.append(retrieval_info)
        
        return batch_evidences, batch_retrieval_infos
    
    
    def verify_claim(
        self, 
        claim: str, 
        gold_id_list: List[str], 
        gold_evidence: str,
        top_k: int = 5
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        
        if self.setting.startswith("open-book"):
            evidence, retrieval_info = self.retrieve_evidence(
                claim, gold_id_list, gold_evidence, top_k
            )
        else:
            evidence, retrieval_info = None, None
        
        answer = self.base_model.verify(claim, evidence)
        prediction = "SUPPORTED" if answer else "NOT_SUPPORTED"
        
        return prediction, retrieval_info
    
    
    def batch_verify_claim(
        self, 
        batch_claims: List[str], 
        batch_gold_id_lists: List[List[str]], 
        batch_gold_evidences: List[str],
        top_k: int = 5
    ) -> Tuple[List[str], Optional[List[Dict[str, Any]]]]:
        
        if self.setting.startswith("open-book"):
            batch_evidences, batch_retrieval_infos = self.batch_retrieve_evidence(
                batch_claims, batch_gold_id_lists, batch_gold_evidences, top_k
            )
        else:
            batch_evidences, batch_retrieval_infos = None, None
        
        batch_answers = self.base_model.batch_verify(batch_claims, batch_evidences)
        batch_predictions = ["SUPPORTED" if answer else "NOT_SUPPORTED" for answer in batch_answers]
        
        return batch_predictions, batch_retrieval_infos
    
    
    def process_sample_direct(
        self, 
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:        
        
        try:
            prediction, retrieval_info = self.verify_claim(
                sample["claim"], sample["gold_id_list"], sample["gold_evidence"], self.direct_top_k
            )
            sample.update({
                "prediction": prediction,
                "is_correct": (prediction == sample["label"]),
                "retrieval_info": retrieval_info
            })
        
        except Exception as e:
            logger.warning(f"Failed to verify sample. Skipping this sample. Error: {e}")
            
            prediction = random.choice(["SUPPORTED", "NOT_SUPPORTED"])
            sample.update({
                "prediction": prediction,
                "is_correct": (prediction == sample["label"]),
                "retrieval_info": None
            })
        
        return sample
    
    
    def process_batch_direct(
        self, 
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        
        try:
            batch_claims = [sample["claim"] for sample in batch]
            batch_gold_id_lists = [sample["gold_id_list"] for sample in batch]
            batch_gold_evidences = [sample["gold_evidence"] for sample in batch]
            batch_predictions, batch_retrieval_infos = self.batch_verify_claim(
                batch_claims, batch_gold_id_lists, batch_gold_evidences, self.direct_top_k
            )        
            
            if not batch_retrieval_infos:
                batch_retrieval_infos = [None] * len(batch)
            
            for sample, prediction, retrieval_info in zip(batch, batch_predictions, batch_retrieval_infos):
                sample.update({
                    "prediction": prediction,
                    "is_correct": (prediction == sample["label"]),
                    "retrieval_info": retrieval_info
                })
        
        except Exception as e:
            logger.warning(f"Failed to verify batch. Skipping this batch. Error: {e}")
            
            batch_predictions = [random.choice(["SUPPORTED", "NOT_SUPPORTED"]) for _ in range(len(batch))]
            for sample, prediction in zip(batch, batch_predictions):
                sample.update({
                    "prediction": prediction,
                    "is_correct": (prediction == sample["label"]),
                    "retrieval_info": None
                })
        
        return batch
    
    
    def print_result(
        self,
        result_list: List[Dict[str, Any]]
    ) -> None:
        
        prediction_list = [sample["prediction"] for sample in result_list]
        label_list = [sample["label"] for sample in result_list]
        num_hops_list = [sample["num_hops"] for sample in result_list]
        
        print_evaluation_by_hop(prediction_list, label_list, num_hops_list)
    
    
    def save_result(
        self,
        result_list: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(result_list, f, indent=4)
        logger.info(f"Results saved at '{output_path}'.")
    
    
    def run_direct(
        self,
    ) -> List[Dict[str, Any]]:
        
        with open(self.input_path, "r") as f:
            input_list = json.load(f)

        logger.info("Starting Direct verification...")

        result_list = []
        if self.batch_size > 1:
            for i in tqdm(range(0, len(input_list), self.batch_size)):
                batch = input_list[i : i + self.batch_size]
                batch_results = self.process_batch_direct(batch)
                result_list.extend(batch_results)
                
        else:
            for sample in tqdm(input_list):
                result = self.process_sample_direct(sample)
                result_list.append(result)
                
        logger.info("Direct verification completed!")
        
        return result_list


if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        logger.info(f"- {key}: {value}")
    
    verifier = Direct(args)
    result_list = verifier.run_direct()
    verifier.save_result(result_list, verifier.direct_path)
    
    print(f"\n* Direct verification results for {len(result_list)} samples *\n")
    verifier.print_result(result_list)
        
