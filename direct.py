import argparse
import json
import logging
import os
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from typing import *

from model_library.base_model import BaseModel
from utils.retriever import PyseriniRetriever
from utils.evaluator import print_evaluation_by_hop

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
    parser.add_argument("--top_k", type=int, default=10
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
        self.cache_dir = args.cache_dir
        self.base_model = None        
        
        retriever_corpus_dir = f"./datasets/{args.dataset}/corpus/index"
        self.retriever = PyseriniRetriever(retriever_corpus_dir, use_bm25=True, k1=0.9, b=0.4)
        self.evidence_max_len = 40000

        self.setting = args.setting
        self.top_k = args.top_k

        if args.direct_filename:
            direct_filename = args.direct_filename
        else:
            direct_filename = args.input_filename
        self.direct_path = os.path.join(
            "results", self.dataset, "verifications", "direct", self.base_model_name.split("/")[-1], 
            self.setting, direct_filename
        )
    
    
    def check_retrieval(
        self, 
        doc_id: str, 
        gold_id_list: List[str]
    ) -> int:
        
        return 1 if doc_id in gold_id_list else 0
    
    
    def truncate(
        self, 
        text: str
    ) -> str:
        
        if len(text) > self.evidence_max_len:            
            text = text[:self.evidence_max_len]
            logger.warning(f"Evidence length exceeds {self.evidence_max_len} characters and has been truncated to prevent GPU memory overflow.")
        
        return text

    
    def retrieve_evidence(
        self, 
        query: str, 
        gold_id_list: List[str], 
        gold_evidence: str,
        top_k: Optional[int] = 10
    ) -> Tuple[List[str], Dict[str, Any]]:
        
        doc_id_list, score_list, is_gold_list, evidence_list = [], [], [], []
        hit_list = self.retriever.retrieve(query, top_k)
        
        if self.setting == "open-book+gold":
            doc_id_list.extend(gold_id_list)
            score_list.extend([0] * len(gold_id_list))
            is_gold_list.extend([1] * len(gold_id_list))
            if gold_evidence:  
                evidence_list.append(gold_evidence.strip())
        
        for hit in hit_list:
            if len(doc_id_list) >= top_k:
                break  
            
            if hit["doc_id"] not in doc_id_list:
                doc_id_list.append(hit["doc_id"])
                score_list.append(hit["score"])
                is_gold_list.append(self.check_retrieval(hit["doc_id"], gold_id_list))
                if hit["text"]:
                    evidence_list.append(hit["text"].strip())
        
        retrieval_info = {
            "query": query,
            "doc_id_list": doc_id_list, 
            "is_gold_list": is_gold_list 
        }
        
        return evidence_list, retrieval_info


    def verify_claim(
        self, 
        claim: str, 
        gold_id_list: List[str], 
        gold_evidence: str,
        top_k: Optional[int] = 10,
        document_level: Optional[str] = "concat",
        retrieval_result: Optional[Tuple[List[str], Dict[str, Any]]] = None
    ) -> Tuple[str, Dict[str, Any], str]:
        
        if retrieval_result is None:
            evidence_list, retrieval_info = self.retrieve_evidence(
                claim, gold_id_list, gold_evidence, top_k
            )
        else:
            evidence_list, retrieval_info = retrieval_result
            
        answer, supported_doc_id = False, None
        
        if document_level == "concat":
            evidence_concat = self.truncate('\n'.join(evidence_list))
            answer = self.base_model.verify(claim, evidence_concat)
            if answer:
                supported_doc_id = "[concatenated_documents]"
            
        elif document_level == "each":
            for idx, evidence in enumerate(evidence_list):
                evidence = self.truncate(evidence)
                answer = self.base_model.verify(claim, evidence)
                if answer:
                    supported_doc_id = retrieval_info["doc_id_list"][idx]
                    break

        elif document_level == "concat+each":
            evidence_concat = self.truncate('\n'.join(evidence_list))
            answer = self.base_model.verify(claim, evidence_concat)
            if answer:
                supported_doc_id = "[concatenated_documents]"
            else:
                for idx, evidence in enumerate(evidence_list):
                    evidence = self.truncate(evidence)
                    answer = self.base_model.verify(claim, evidence)
                    if answer:
                        supported_doc_id = retrieval_info["doc_id_list"][idx]
                        break
        
        prediction = "SUPPORTED" if answer else "NOT_SUPPORTED"
        
        return prediction, retrieval_info, supported_doc_id
    
    
    def process_sample_direct(
        self, 
        sample: Dict[str, Any],
        retrieval_result: Optional[Tuple[List[str], Dict[str, Any]]] = None
    ) -> Dict[str, Any]:        
        
        try:
            prediction, retrieval_info, supported_doc_id = self.verify_claim(
                sample["claim"], sample["gold_id_list"], sample["gold_evidence"], self.top_k, "concat", retrieval_result
            )
            sample.update({
                "prediction": prediction,
                "is_correct": (prediction == sample["label"]),
                "retrieval_info": {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None,
                "supported_doc_id": supported_doc_id
            })
        
        except Exception as e:
            logger.warning(f"Failed to verify the sample. Skipping this sample. Error: {e}")
            
            prediction = random.choice(["SUPPORTED", "NOT_SUPPORTED"])
            sample.update({
                "prediction": prediction,
                "is_correct": (prediction == sample["label"]),
                "retrieval_info": None,
                "supported_doc_id": None
            })
        
        return sample
    
    
    def run_direct(
        self,
        input_indices: Set[int] = None
    ) -> List[Dict[str, Any]]:
        
        with open(self.input_path, "r") as f:
            input_list = json.load(f)

        if input_indices:
            input_list = [sample for sample in input_list if sample["index"] in input_indices]

        logger.info("Starting Direct verification...")

        if self.base_model is None:
            tokenizer = T5Tokenizer.from_pretrained(self.base_model_name, cache_dir=self.cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, cache_dir=self.cache_dir, device_map="auto")
            self.base_model = BaseModel(model, tokenizer)
            logger.info(f"Base model '{self.base_model_name}' initialized successfully.")        

        result_list = []
        for sample in tqdm(input_list):
            result = self.process_sample_direct(sample)
            result_list.append(result)
                
        logger.info("Direct verification completed!")
        
        return result_list


    def select_strategy(
        self, 
        claim: str, 
        gold_id_list: List[str], 
        gold_evidence: str,
        top_k: int = 10
    ) -> Tuple[str, Tuple[List[str], Dict[str, Any]]]:
        
        evidence_list, retrieval_info = self.retrieve_evidence(
            claim, gold_id_list, gold_evidence, top_k
        )
        
        evidence_concat = self.truncate('\n'.join(evidence_list))
        answer = self.base_model.strategy_select(claim, evidence_concat)
        
        strategy = "direct" if answer else "graphcheck"
        
        return strategy, (evidence_list, retrieval_info)
    
    
    def process_sample_ss(
        self, 
        sample: Dict[str, Any],
        do_direct: Optional[bool] = False
    ) -> Dict[str, Any]:        
        
        try:
            strategy, (evidence_list, retrieval_info) = self.select_strategy(
                sample["claim"], sample["gold_id_list"], sample["gold_evidence"], self.top_k
            )
            sample.update({
                "strategy": strategy,
                "ss_retrieval_info": {k: str(v) for k, v in retrieval_info.items()} if retrieval_info else None,
            })
            if do_direct and strategy == "direct":
                sample = self.process_sample_direct(sample, (evidence_list, retrieval_info))
        
        except Exception as e:
            logger.warning(f"Failed to select a strategy for the sample. Skipping this sample. Error: {e}")
            
            strategy = random.choice(["direct", "graphcheck"])
            sample.update({
                "strategy": strategy,
                "ss_retrieval_info": None,
            })
            if do_direct and strategy == "direct":
                sample = self.process_sample_direct(sample)
        
        return sample
    
    
    def run_strategy_selector(
        self,
        do_direct: Optional[bool] = False
    ) -> List[Dict[str, Any]]:
        
        with open(self.input_path, "r") as f:
            input_list = json.load(f)

        if do_direct:
            logger.info("Starting strategy selection with Direct verification...")
        else:
            logger.info("Starting strategy selection...")

        if self.base_model is None:
            tokenizer = T5Tokenizer.from_pretrained(self.base_model_name, cache_dir=self.cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(self.base_model_name, cache_dir=self.cache_dir, device_map="auto")
            self.base_model = BaseModel(model, tokenizer)
            logger.info(f"Base model '{self.base_model_name}' initialized successfully.")        

        result_list, direct_indices, graphcheck_indices = [], set(), set()
        for sample in tqdm(input_list):
            result = self.process_sample_ss(sample, do_direct)
            result_list.append(result)
            if result["strategy"] == "direct":
                direct_indices.add(result["index"])
            else:
                graphcheck_indices.add(result["index"])
                
        logger.info("Strategy selection completed!")
        logger.info(f"- # of samples verified with Direct: {len(direct_indices)}/{len(input_list)}")
        logger.info(f"- # of samples verified with GraphCheck: {len(graphcheck_indices)}/{len(input_list)}")
        
        if do_direct:
            logger.info("Direct verification completed!")

        
        
        return result_list, direct_indices, graphcheck_indices


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
    
    
    def has_complete_results(
        self,
        file_path: str,
        input_indices: Optional[Set[int]] = None
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
    

if __name__ == "__main__":
    args = parse_args()
    for key, value in vars(args).items():
        logger.info(f"- {key}: {value}")
    
    verifier = Direct(args)
    result_list = verifier.run_direct()
    verifier.save_result(result_list, verifier.direct_path)
    
    print(f"\n* Direct verification results for {len(result_list)} samples *\n")
    verifier.print_result(result_list)
        
