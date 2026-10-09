from typing import override
from abc import abstractmethod

from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel

class Changes(BaseModel): 
    reasoning: str
    changes: list[str]

class BaseAgent():
    def __init__(
        self, 
        model: str, 
        reasoning: str, 
        postfix: str | None 
    ):
        self.model = model 
        self.reasoning = reasoning 
        self.postfix = postfix

    @abstractmethod
    def next_change(self) -> Changes: 
        pass

    @abstractmethod
    def inj_history(self, msg: str): 
        pass 

    @abstractmethod
    def get_usage() -> dict: 
        pass 

    def _get_init_policy(self) -> str :
        with open(self.ctx_path + 'policy.py', 'r') as f: 
            return f.read() 

    def _get_root_ctx(self) -> str: 
        with open(self.ctx_path + 'ROOT_ext.md', 'r') as f: 
            return f.read() 

    def get_prefix_context(self) -> str: 
        prompt = F'''
SUPPLY-SHOCK CHAIN ENVIRONMENT CONTEXT: 

{self._get_root_ctx()}

INSTRUCTIONS: 

You are an AI assistant. Your task is to iteratively improve the RL policy for the supply-shock chain environment.

You are provided with a set of policies along with their scores. The first line of each policy is its score in the `# {{score}}` format (higher is better). Your task is to propose new policies that score higher than the provided ones.

The set may include failing policies (score `# -999`, then a `# error: {{message}}` line): avoid their mistakes. It may be empty if no policy exists yet.

The provided policies are only examples. The goal is the highest possible score, not staying close to them. You are free to propose a completely different policy: redesign the decision logic, use any information available in `config` and `observation`, keep internal state between steps, and so on.

Rules:
1. Every policy you propose must adhere strictly to the interface described in the environment context.
2. You may propose multiple changes in a single iteration, for example to compare several alternative policy changes or to sweep over policy parameters.
3. Structure your response in this order: first your reasoning, then your proposed changes.
4. Every proposed change is the full content of a new policy file (only Python source, no Markdown code fences). Changes do not accumulate between iterations.
5. Never hard-code shapes (numbers of nodes, routes, goods, weeks): a policy must work on every task variant (tiny, small, full), so read them from `config` and `observation`.
'''.strip()
        if self.postfix: 
            prompt += '\n' + self.postfix
        return prompt 

class BaseAgentLP(BaseAgent): 
    @override
    def get_prefix_context(self) -> str: 
        prompt = F'''
SUPPLY-SHOCK CHAIN ENVIRONMENT CONTEXT: 

{self._get_root_ctx()}

INSTRUCTIONS: 

You are an AI assistant. Your task is to iteratively improve the RL policy for the supply-shock chain environment.

You are provided with a set of policies along with their scores. The first line of each policy is its score in the `# {{score}}` format (higher is better). Your task is to propose new policies that score higher than the provided ones.

The set may include failing policies (score `# -999`, then a `# error: {{message}}` line): avoid their mistakes. It may be empty if no policy exists yet.

The provided policies are only examples. The goal is the highest possible score, not staying close to them. You are free to propose a completely different policy: redesign the decision logic, use any information available in `config` and `observation`, keep internal state between steps, and so on.

Every policy must decide its actions by solving a linear program.

Rules:
1. Every policy you propose must adhere strictly to the interface described in the environment context.
2. You may propose multiple changes in a single iteration, for example to compare several alternative policy changes or to sweep over policy parameters.
3. Structure your response in this order: first your reasoning, then your proposed changes.
4. Every proposed change is the full content of a new policy file (only Python source, no Markdown code fences). Changes do not accumulate between iterations.
5. Never hard-code shapes (numbers of nodes, routes, goods, weeks): a policy must work on every task variant (tiny, small, full), so read them from `config` and `observation`.
6. SciPy is available: use it to solve the linear program. 
'''.strip()
        if self.postfix:
            prompt += '\n' + self.postfix
        return prompt

class AgentOpenAI(BaseAgent):
    def __init__(self, model: str, reasoning: str, postfix: str | None): 
        super().__init__(model, reasoning, postfix)
        self.client = OpenAI()
        self.costs = {
            # per million tokens: 
            'gpt-6-luna': {'input_tokens': 0.1, 'output_tokens': 0.5, 'cached_tokens': 0.01, 'cache_write_tokens': 0.125},
            'gpt-6.1-sol': {'input_tokens': 2.0, 'output_tokens': 10.0, 'cached_tokens': 0.1, 'cache_write_tokens': 2.5},
        } 
        self.cache_mode = 'explicit'
        self.ctx_path = './evolve/context/'
        self.history = self.init_history()
        self.input_tokens, self.output_tokens, self.cached_tokens,  self.cache_write_tokens = 0, 0, 0, 0

    def next_change(self) -> Changes:
        res = self.client.responses.parse(
            model=self.model, 
            reasoning={'effort': self.reasoning},
            text_format=Changes,
            input=self.history, 
            prompt_cache_options={'mode': self.cache_mode}, 
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

    def get_usage(self) -> dict: 
        c = self.costs[self.model]
        # input_tokens includes cached and cache-write tokens: charge only the rest at the full input price
        uncached = self.input_tokens - self.cached_tokens - self.cache_write_tokens
        cost = (uncached * c['input_tokens']
                + self.cached_tokens * c['cached_tokens']
                + self.cache_write_tokens * c['cache_write_tokens']
                + self.output_tokens * c['output_tokens']) / 1_000_000
        return {
            'input_tokens': self.input_tokens, 
            'output_tokens': self.output_tokens, 
            'cached_tokens': self.cached_tokens, 
            'cache_write_tokens': self.cache_write_tokens, 
            'cost': cost
        } 

    def upd_usage(self, usage): 
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens 
        self.cached_tokens += usage.input_tokens_details.cached_tokens 
        self.cache_write_tokens += usage.input_tokens_details.cache_write_tokens 

class AgentOpenAILP(AgentOpenAI, BaseAgentLP): 
    pass 

class AgentAnthropic(BaseAgent):
    def __init__(self, model: str, reasoning: str | None, postfix: str | None):
        super().__init__(model, reasoning, postfix)
        self.client = Anthropic()
        self.costs = {
            # per million tokens; cache writes are the 5-minute ttl (1.25x input)
            'claude-haiku-4-5': {'input_tokens': 1.0, 'output_tokens': 5.0, 'cached_tokens': 0.1, 'cache_write_tokens': 1.25},
            'claude-sonnet-5-5': {'input_tokens': 2.0, 'output_tokens': 10.0, 'cached_tokens': 0.2, 'cache_write_tokens': 2.5},
            'claude-opus-5-5': {'input_tokens': 4.0, 'output_tokens': 20.0, 'cached_tokens': 0.2, 'cache_write_tokens': 5.0},
        }
        self.max_tokens = 16000          # required by the api; above ~21k the sdk requires streaming
        self.ctx_path = './evolve/context/'
        self.system = self.init_system()
        self.history = []
        self.input_tokens, self.output_tokens, self.cached_tokens,  self.cache_write_tokens = 0, 0, 0, 0

    def next_change(self) -> Changes:
        kwargs = {}
        if self.reasoning not in (None, 'none'):
            # effort: low | medium | high | xhigh | max; haiku 4.5 rejects it (use reasoning='none' there)
            kwargs['output_config'] = {'effort': self.reasoning}
        res = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.history,
            output_format=Changes,
            **kwargs,
        )
        if res.stop_reason in ('refusal', 'max_tokens'):
            raise RuntimeError(f'claude stopped with {res.stop_reason}: {res.stop_details}')
        self.upd_history(res.content)
        self.upd_usage(res.usage)
        return res.parsed_output, self.get_usage()

    def init_system(self) -> list:
        # explicit: the fixed prefix (environment, base policy, rules) is cached on its own
        return [{'type': 'text', 'text': self.get_prefix_context(), 'cache_control': {'type': 'ephemeral'}}]

    def upd_history(self, content):
        # blocks go back unchanged (thinking signatures included), minus the SDK-only parsed_output
        blocks = [b.to_dict() for b in content]
        for b in blocks: b.pop('parsed_output', None)
        self.history.append({'role': 'assistant', 'content': blocks})

    def inj_history(self, msg: str):
        self.history.append({'role': 'user', 'content': msg})

    def get_usage(self) -> dict:
        c = self.costs[self.model]
        # unlike openai, anthropic's input_tokens excludes cache reads and writes: the four counts add up
        cost = (self.input_tokens * c['input_tokens']
                + self.cached_tokens * c['cached_tokens']
                + self.cache_write_tokens * c['cache_write_tokens']
                + self.output_tokens * c['output_tokens']) / 1_000_000
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cached_tokens': self.cached_tokens,
            'cache_write_tokens': self.cache_write_tokens,
            'cost': cost
        }

    def upd_usage(self, usage):
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cached_tokens += usage.cache_read_input_tokens or 0
        self.cache_write_tokens += usage.cache_creation_input_tokens or 0

class AgentAnthropicLP(AgentAnthropic, BaseAgentLP): 
    pass 
