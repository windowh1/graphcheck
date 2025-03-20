import logging
import random
import torch
from typing import List, Optional

logger = logging.getLogger(__name__)


class FlanT5:
    def __init__(
        self, 
        model, 
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
    
    
    def generate(
        self, 
        input: str, 
        **generator_args
    ) -> str:
        
        input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids, do_sample=False, **generator_args)
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        return answer
    
    
    def batch_generate(
        self, 
        batch_input: List[str], 
        **generator_args
    ) -> List[str]:
        tokenized = self.tokenizer(batch_input, return_tensors="pt", padding=True, truncation=False)
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids, attention_mask=attention_mask, do_sample=False, **generator_args)
        batch_answer = [
            text.strip() for text in self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        ]
        return batch_answer
    
    
class BaseModel(FlanT5):
    def __init__(
        self, 
        model, 
        tokenizer
    ):
        super().__init__(model, tokenizer)
        
        
    def get_infilling_prompt(
        self, 
        query: str, 
        evidence: Optional[str] = None
    ) -> str:
        
        if not evidence:
            prompt = f"Fill in the blank with the correct entity: {query}\nAnswer:"
        else:
            prompt = f"{evidence}\nBased on the above information, fill in the blank with the correct entity: {query}\nAnswer:"
        
        return prompt
    
    
    def infill(
        self, 
        query: str, 
        evidence: Optional[str]
    ) -> str:
        
        prompt = self.get_infilling_prompt(query, evidence)
        
        answer = self.generate(prompt, max_new_tokens=32)
        if answer.lower().startswith("blank is "):
            answer = answer[len("blank is "):]
        
        return answer
    
    
    def batch_infill(
        self, 
        batch_query: List[str], 
        batch_evidence: Optional[List[str]]
    ) -> List[str]:
        
        if not batch_evidence:
            batch_prompt = [
                self.get_infilling_prompt(query) for query in batch_query
            ]
        else:
            batch_prompt = [
                self.get_infilling_prompt(query, evidence) for query, evidence in zip(batch_query, batch_evidence)
            ]
            
        batch_answer = self.batch_generate(batch_prompt, max_new_tokens=32)
        batch_answer = [
            answer[len("blank is "):] if answer.lower().startswith("blank is ") else answer
            for answer in batch_answer
        ]
        
        return batch_answer
    
    
    def parse_boolean_answer(
        self, 
        answer: str
    ) -> bool:
        
        answer = answer.split("\n")[0].lower().strip(" .")
        boolean_mapping = {
            "true": True, "false": False, "yes": True, "no": False,
            "it is impossible to say": False, "it's impossible to say": False,
            "it is impossible to tell": False, "it's impossible to tell": False,
            "it is not possible to say": False, "it's not possible to say": False,
            "it is not possible to tell": False, "it's not possible to tell": False
        }
        
        if answer in boolean_mapping:
            return boolean_mapping[answer]
        
        for sample_text, boolean_value in boolean_mapping.items():
            if answer.startswith(sample_text):
                return boolean_value
        
        logger.error(f"Unmapped answer detected: '{answer}'")
        return random.choice([True, False])
    
    
    def get_verification_prompt(
        self, 
        claim: str, 
        evidence: Optional[str] = None
    ) -> str:
        
        claim = claim.strip().rstrip(".")
        if not evidence:
            prompt = f"Is it true that '{claim}'? True or false? If it's impossible to say, answer as false.\nAnswer:"
        else:
            prompt = f"{evidence}\nBased on the above information, is it true that '{claim}'? True or false? If it's impossible to say, answer as false.\nAnswer:"
        
        return prompt
    
    
    def verify(
        self, 
        claim: str, 
        evidence: Optional[str]
    ) -> bool:
        
        prompt = self.get_verification_prompt(claim, evidence)
        
        answer = self.generate(prompt, max_new_tokens=8)
        answer = self.parse_boolean_answer(answer)
        
        return answer
    
    
    def batch_verify(
        self, 
        batch_claim: List[str], 
        batch_evidence: Optional[List[str]]
    ) -> List[bool]:
        
        if not batch_evidence:
            batch_prompt = [
                self.get_verification_prompt(claim) for claim in batch_claim
            ]
        else:            
            batch_prompt = [
                self.get_verification_prompt(claim, evidence) for claim, evidence in zip(batch_claim, batch_evidence)
            ]
        
        batch_answer = self.batch_generate(batch_prompt, max_new_tokens=8)
        batch_answer = [self.parse_boolean_answer(answer) for answer in batch_answer]
        
        return batch_answer
