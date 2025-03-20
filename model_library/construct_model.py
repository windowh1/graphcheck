import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from model_library.prompt import construction_prompt

logger = logging.getLogger(__name__)


class GPT:
    def __init__(
        self, 
        client: Any, 
        gpt_model_name: str
    ):
        self.client = client
        self.gpt_model_name = gpt_model_name
    
    
    def generate(
        self, 
        user_message: str, 
        system_message: Optional[str] = None, 
        max_tokens: int = 1024, 
        temperature: float = 0.0, 
        top_p: float = 1.0
    ) -> str:
        
        if not system_message:
            messages=[
                {"role": "user", "content": user_message}
            ]
        else:
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        response = self.client.chat.completions.create(
            model=self.gpt_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return response.choices[0].message.content.strip()
    
    
    def wait_for_batch_completion(
        self, 
        batch_id: str
    ) -> str:
        
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            status = batch_status.status
            
            if status == "completed":
                return batch_status.output_file_id
            elif status in ["failed", "expired", "cancelled"]:
                self.check_batch_errors(batch_id)
                raise ValueError(f"Batch {batch_id} failed or was cancelled.")
            else:
                logger.info(f"Batch {batch_id} is still processing (status: {status}). Retrying in 30 seconds...")
                time.sleep(30)
    
    
    def check_batch_errors(
        self, 
        batch_id: str
    ) -> None:
        
        batch_status = self.client.batches.retrieve(batch_id)
        if batch_status.errors:
            logger.error(f"Batch {batch_id} errors: {batch_status.errors}")
        else:
            logger.error(f"No specific error details for batch {batch_id}.")
    
    
    def batch_generate(
        self, 
        user_message_list: List[str], 
        system_message_list: Optional[List[str]] = None, 
        max_tokens: int = 1024,
        temperature: float = 0.0, 
        top_p: float = 1.0
    ) -> List[str]:
        
        if not system_message_list:
            open_ai_messages_list = [
                {
                    "custom_id": f"request-{idx + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.gpt_model_name,
                        "messages": [
                            {"role": "user", "content": user_message}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                } for idx, user_message in enumerate(user_message_list)
            ]
        else:
            open_ai_messages_list = [
                {
                    "custom_id": f"request-{idx + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.gpt_model_name,
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                } for idx, (user_message, system_message) in enumerate(zip(user_message_list, system_message_list))
            ]
        
        # Write to a temporary file (in jsonl format)
        jsonl_file = "make_jsonl.jsonl"
        with open(jsonl_file, encoding="utf-8", mode="w") as file:
            for i in open_ai_messages_list:
                file.write(json.dumps(i) + "\n")
        
        # Create the batch
        batch_input_file = self.client.files.create(file=open(jsonl_file, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        response = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        if response is None:
            raise ValueError("Received no response from batches.create()")
        else:
            logger.info(f"Batch created successfully: {response}")
        
        # Wait for the batch to complete
        batch_id = response.id
        output_file_id = self.wait_for_batch_completion(batch_id)
        
        # Parse the output file content
        if output_file_id:
            file_response = self.client.files.content(output_file_id)
            batch_responses = file_response.text.strip().split("\n")
            content_list = []
            for response_str in batch_responses:
                try:
                    response_json = json.loads(response_str)
                    content = response_json["response"]["body"]["choices"][0]["message"]["content"]
                    content_list.append(content)
                except (KeyError, json.JSONDecodeError) as e:
                    logger.error(f"Error parsing response: {e}")
            
            return content_list
        
        else:
            raise ValueError(f"No output file generated for batch {batch_id}")


class ConstructModel(GPT):
    def __init__(
        self, 
        client: Any, 
        gpt_model_name: str,
        use_openai_batch_api: bool
    ):  
        super().__init__(client, gpt_model_name)
        
        self.gpt_model_name = gpt_model_name
        self.use_openai_batch_api = use_openai_batch_api
        self.openai_batch_size = 10
        self.construction_prompt = construction_prompt
    
    
    def parse_graph(
        self, 
        generated_text: str
    ) -> Tuple[List[str], List[str]]:
        
        first_section, second_section = [], []
        flag = 0
        
        lines = [line.strip() for line in generated_text.split("\n")]
        for line in lines:
            if not line:
                continue
            if "no latent entities identified" in line.lower():
                continue
            if line.startswith("# Latent Entities"):
                continue
            if line.startswith("# Triples"):
                flag = 1
                continue
            if not line.startswith("(ENT"):
                flag = 1
            
            if flag == 0:
                first_section.append(line)
            elif flag == 1:
                second_section.append(line)
        
        def_triples = []
        for idx, line in enumerate(first_section.copy()):
            expected_prefix = f"(ENT{idx+1}) [SEP] is [SEP]"
            if line.startswith(expected_prefix):
                def_triples.append(line)
                first_section.remove(line)
        
        triples = first_section + second_section
        
        return def_triples, triples
    
    
    def process_sample(
        self,
        sample: Dict[str, Any]
        ) -> Dict[str, Any]:
        
        prompt = self.construction_prompt.replace("<<target_claim>>", sample["claim"])
        answer = self.generate(prompt)
        
        def_triples, triples = self.parse_graph(answer)
        
        sample.update({
            "definition_triples": def_triples,
            "triples": triples
        })
        
        return sample
    
    
    def process_batch(
        self,
        batch: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
        
        batch_prompt = [
            self.construction_prompt.replace("<<target_claim>>", sample["claim"]) for sample in batch
        ]
        batch_answer = self.batch_generate(batch_prompt)
        
        for sample, answer in zip(batch, batch_answer):
            def_triples, triples = self.parse_graph(answer)
            sample.update({
                "definition_triples": def_triples,
                "triples": triples
            })
        
        return batch
    
    
    def construct_graph(
        self,
        input_list: List[Dict[str, Any]],
        graph_path: str,
        force_new_construction: bool = False
    ) -> List[Dict[str, Any]]:
        
        graph_list = []
        if os.path.exists(graph_path) and not force_new_construction:
            with open(graph_path, "r") as f:
                graph_list = json.load(f)
                
            input_indices = {sample["index"] for sample in input_list}
            
        existing_count = len([sample for sample in graph_list if sample["index"] in input_indices])
        total_count = len(input_list)
        
        if existing_count > 0:
            remaining_count = total_count - existing_count
            logger.info(f"{existing_count}/{total_count} samples already have graphs. Constructing graphs for remaining {remaining_count} samples...")
            
            existing_indices = {sample["index"] for sample in graph_list}
            input_list = [sample for sample in input_list if sample["index"] not in existing_indices]
        
        else:
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            logger.info(f"Constructing graphs for {len(input_list)} samples...")
        
        new_count = 0
        if self.use_openai_batch_api:
            for i in tqdm(range(0, len(input_list), self.openai_batch_size)):
                try:
                    batch = input_list[i : i + self.openai_batch_size]
                    batch_graph = self.process_batch(batch)
                    graph_list.extend(batch_graph)
                    
                    new_count += len(batch_graph)
                    if new_count % (self.openai_batch_size * 5) == 0 and new_count < remaining_count:
                        graph_list = sorted(graph_list, key=lambda x: x["index"])
                        with open(graph_path, "w") as f:
                            json.dump(graph_list, f, indent=4)
                            logger.info(f"Checkpoint: {new_count} new graphs saved at '{graph_path}'")
                    
                except Exception:
                    logger.warning("Failed to construct graphs. Skipping this batch.")
        else:
            for sample in tqdm(input_list):
                try:
                    graph = self.process_sample(sample)
                    graph_list.append(graph)
                    
                    new_count += 1
                    if new_count % 50 == 0 and new_count < remaining_count:
                        graph_list = sorted(graph_list, key=lambda x: x["index"])
                        with open(graph_path, "w") as f:
                            json.dump(graph_list, f, indent=4)
                            logger.info(f"Checkpoint: {new_count} new graphs saved at '{graph_path}'")
                
                except Exception:
                    logger.warning("Failed to construct a graph. Skipping this sample.")
        
        logger.info(f"Graph construction completed!")
        
        graph_list = sorted(graph_list, key=lambda x: x["index"])
        with open(graph_path, "w") as f:
            json.dump(graph_list, f, indent=4)
            logger.info(f"Graphs saved at '{graph_path}'")
        
        graph_list = [sample for sample in graph_list if sample["index"] in input_indices]
        
        return graph_list