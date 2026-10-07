# TODO make prompts interchangable along with caching handling!
from abc import abstractmethod

from openai import OpenAI
from anthropic import Anthropic
from pydantic import BaseModel

class Changes(BaseModel): 
    reasoning: str
    changes: list[str]

class BaseAgent():
    def __init__(self, model: str, reasoning: str, t: float | None, top_p: float | None):
        self.model = model 
        self.reasoning = reasoning 
        self.t = t
        self.top_p = top_p

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
        with open(self.ctx_path + 'ROOT_old.md', 'r') as f: 
            return f.read() 

    # older iterative optimization version
    def get_prefix_context_old(self) -> str: 
        return F'''
SUPPLY-SHOCK CHAIN ENVIRONMENT CONTEXT: 

{self._get_root_ctx()}

INSTRUCTIONS: 

You are an AI assistant. Your task is to iteratively improve the RL policy for the supply-shock chain environment.

The initial policy is defined below:

{self._get_init_policy()}

The initial policy is only a minimal working example that shows the required interface. The goal is the highest possible score, not staying close to this example. You are free to propose a completely different policy: redesign the decision logic, use any information available in `config` and `observation`, keep internal state between steps, and so on.

Rules:
1. Every policy you propose must adhere strictly to the interface of the initial policy.
2. You may propose multiple changes in a single iteration, for example to compare several alternative policy changes or to sweep over policy parameters.
3. Structure your response in this order: first your reasoning, then your proposed changes.
4. Every proposed change is the full content of a new policy file (only Python source, no Markdown code fences). Changes do not accumulate between iterations.
5. In each iteration, at least one proposed change must try a fundamentally different strategy from everything tried so far, not a parameter tweak of an earlier idea.

Each proposed change will be evaluated, and its score will be returned to you in the following format so that you can continue improving the policy:

results: change[0] [score], change[1] [score], ..., change[N-1] [score]

where:
- the change number is the position of that change in the list you proposed in this iteration, starting from 0;
- the score is the numerical result of evaluating that change.
'''.strip()

    # sample optimization version 
    def get_prefix_context(self) -> str: 
        return F'''
SUPPLY-SHOCK CHAIN ENVIRONMENT CONTEXT: 

{self._get_root_ctx()}

INSTRUCTIONS: 

You are an AI assistant. Your task is to iteratively improve the RL policy for the supply-shock chain environment.

You are provided with a set of policies along with their scores. The first line of each policy is its score in the `# {{score}}` format (higher is better). Your task is to propose new policies that score higher than the provided ones.

The provided policies are only examples. The goal is the highest possible score, not staying close to them. You are free to propose a completely different policy: redesign the decision logic, use any information available in `config` and `observation`, keep internal state between steps, and so on.

Rules:
1. Every policy you propose must adhere strictly to the interface of the provided policies.
2. You may propose multiple changes in a single iteration, for example to compare several alternative policy changes or to sweep over policy parameters.
3. Structure your response in this order: first your reasoning, then your proposed changes.
4. Every proposed change is the full content of a new policy file (only Python source, no Markdown code fences). Changes do not accumulate between iterations.
5. Never hard-code shapes (numbers of nodes, routes, goods, weeks): a policy must work on every task variant (tiny, small, full), so read them from `config` and `observation`.
'''.strip()

class AgentOpenAI(BaseAgent): 
    def __init__(self, model: str, reasoning: str, t: float = 1.0, top_p: float = 0.98): 
        super().__init__(model, reasoning, t, top_p)
        self.client = OpenAI()
        self.t = t
        self.top_p = top_p
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
            temperature=self.t, 
            top_p=self.top_p
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

class AgentAnthropic(BaseAgent):
    def __init__(self, t: float = 1.0, top_p: float = 0.98):
        super().__init__(t,top_p)
        self.client = Anthropic()
        self.model = 'claude-haiku-4-5'  # claude-haiku-4-5 $1 input, $0.1 cached input, $1.25 cache writes (5m), $5 output; no thinking unless enabled
        self.max_tokens = 16000          # required by the api; above ~21k the sdk requires streaming
        self.ctx_path = './evolve/context/'
        self.system = self.init_system()
        self.history = []
        self.input_tokens, self.output_tokens, self.cached_tokens,  self.cache_write_tokens = 0, 0, 0, 0

    def next_change(self) -> Changes:
        res = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.history,
            output_format=Changes,
            cache_control={'type': 'ephemeral'},  # implicit: moves the breakpoint to the end of the history every call
            extra_body={'temperature': self.t},   # sdk 1.x dropped sampling args, haiku 4.5 still accepts them; top_p is not sent: 4.5 models take one or the other
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
        return {
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'cached_tokens': self.cached_tokens,
            'cache_write_tokens': self.cache_write_tokens
        }

    def upd_usage(self, usage):
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cached_tokens += usage.cache_read_input_tokens or 0
        self.cache_write_tokens += usage.cache_creation_input_tokens or 0
