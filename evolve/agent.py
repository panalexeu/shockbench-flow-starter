from abc import ABC, abstractmethod

from openai import OpenAI
from pydantic import BaseModel 

class Output(BaseModel): 
    text: str
    changes: list[str] | None

class Agent(ABC):
    pass 

class AgentOpenAI(Agent): 
    def __init__(self): 
        self.client = OpenAI()
        self.model = 'gpt-6-luna'  # gpt-6-luna $0.1 input, $0.01 cached input, $0.125 cache writes, $0.5 output  
        self.reasoning = 'none' 
        self.cache_mode = 'explicit'
        self.history = self.init_history()
        self.input_tokens, self.output_tokens, self.cached_tokens,  self.cache_write_tokens = 0, 0, 0, 0
    
    def next_changes(self) -> Output:
        res = self.client.responses.parse(
            model=self.model, 
            reasoning={'effort': self.reasoning},
            text_format=Output,
            input=self.history, 
            prompt_cache_options={'mode': self.cache_mode}
        ) 
        self.upd_history(res.output)
        self.upd_usage(res.usage)
        return res.output_parsed, self.get_usage()

    def init_history(self) -> list: 
        hist =[
            {'role': 'developer', 'content': [{'type': 'input_text', 'text': self.get_prefix_context(), 'prompt_cache_breakpoint': { "mode": "explicit" }}]}
        ] 
        return hist

    def upd_history(self, out): 
        self.history += out

    def inj_history(self, msg: str): 
        self.history.append({'role': 'user', 'content': msg}) 

    def get_prefix_context(self) -> str: 
        return 'You are a helpful AI assistant. Your task is to remember user name.'

    def get_usage(self) -> dict: 
        return {
            'input_tokens': self.input_tokens, 
            'output_tokens': self.output_tokens, 
            'cached_tokens': self.cached_tokens, 
            'cache_write_tokens': self.cache_write_tokens
        } 

    def upd_usage(self, usage): 
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens 
        self.cached_tokens += usage.input_tokens_details.cached_tokens 
        self.cache_write_tokens += usage.input_tokens_details.cache_write_tokens 
