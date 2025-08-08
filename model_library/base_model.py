import logging
import random
import torch
from typing import *

logger = logging.getLogger(__name__)
random.seed(26)


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
        evidence: Optional[str] = None
    ) -> str:
        
        prompt = self.get_infilling_prompt(query, evidence)
        
        answer = self.generate(prompt, max_new_tokens=32)
        if answer.lower().startswith("blank is "):
            answer = answer[len("blank is "):]
        
        return answer
    
    
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
        
        if not evidence:
            prompt = f"Claim: {claim}\nIs the claim true or false?\nAnswer:"
        else:
            prompt = f"Evidence: {evidence}\nClaim: {claim}\nIs the claim true or false?\nAnswer:"
        
        return prompt
    
    
    def verify(
        self, 
        claim: str, 
        evidence: Optional[str] = None
    ) -> bool:
        
        prompt = self.get_verification_prompt(claim, evidence)
        
        answer = self.generate(prompt, max_new_tokens=8)
        answer = self.parse_boolean_answer(answer)
        
        return answer

    
    def get_selector_prompt(
        self, 
        claim: str, 
        evidence: str
    ) -> str:
        
        prompt = f"Evidence: {evidence}\nClaim: {claim}\nDoes the evidence contain sufficient information to support or refute the claim? Yes or no?\nAnswer:"
        
        return prompt


    def strategy_select(
        self, 
        claim: str, 
        evidence: str
    ) -> bool:
        
        prompt = self.get_selector_prompt(claim, evidence)

        answer = self.generate(prompt, max_new_tokens=8)
        answer = self.parse_boolean_answer(answer)

        return answer
