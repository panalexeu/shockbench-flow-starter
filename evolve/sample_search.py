# sampling approach closer to the one presented AlphaEvolve 
import os 
import hashlib 
from pathlib import Path

from rich import print 
import numpy as np 
from dotenv import load_dotenv
from sbf_starter import  scoring

from .agent import AgentOpenAI, AgentAnthropic

class ReplacmentError(Exception): 
    def __init__(self, *args):
        super().__init__(*args)

_policy_dir = 'evolve/policies/policy1'
def create_policy_dir(): 
    os.makedirs(_policy_dir, exist_ok=True)

_candidates_dir = './evolve/policies/candidate_pool'
def create_candidate_dir(): 
    os.makedirs(_candidates_dir, exist_ok=True)

def get_candidates_pool(size: int) -> str: 
    files_str = ''
    paths = [path for path in Path(_candidates_dir).iterdir() if path.is_file()][:size]
    for path in paths: 
        with open(path, 'r') as f:
            files_str += f.read() + '\n\n'
    return files_str

_base_policy_file = './evolve/context/policy.py'
def _get_def_policy(): 
    with open(_base_policy_file, 'r') as f: 
        return f.read()

def get_path(id_: str) -> str: 
    return _policy_dir + '/' + id_ + '.py'

_error_prefix = '# error: '
def _is_score_line(line: str) -> bool:
    if not line.startswith('#'):
        return False
    try:
        float(line.lstrip('#').strip())
        return True
    except ValueError:
        return False

def hash(text: str) -> str:
    # remove the header add_score writes (score line, then an optional error line); keep every other byte
    first, _, rest = text.partition('\n')
    if _is_score_line(first):
        text = rest
        second, _, rest = text.partition('\n')
        if second.startswith(_error_prefix):
            text = rest
    return hashlib.blake2s(text.encode(), digest_size=6).hexdigest()

def add_score(id_: str, score: float, e: str | None):
    score_str = f'# {score}\n'
    with open(get_path(id_), 'r') as f: lines = f.readlines()
    lines.insert(0, score_str)
    if e is not None:
        e_str = _error_prefix + ' '.join(e.split()) + '\n'  # one line, so the header stays two lines
        lines.insert(1, e_str)
    with open(get_path(id_), 'w') as f: f.write(''.join(lines)) 

def get_score(id_: str) -> float: 
    with open(get_path(id_), 'r') as f: lines = f.readlines()
    score = float(lines[0].lstrip("#").strip())
    return score 

def write_change(id_: str, change: str):
    with open(get_path(id_), 'w') as f: 
        f.write(change)

def fitness(id_: str, episodes): 
    res = episodes.score(get_path(id_), cpu_budget=True)
    return res.rss

def load_policies() -> list[str]: 
    policies = []
    for path in Path(_policy_dir).iterdir():
        if path.is_file(): 
            with open(path, 'r') as f: 
                policies.append(f.read())
    return policies

def _scores_to_probs(scores, t=1.0):
    s = np.asarray(scores, dtype=float)
    s = (s - s.mean()) / (s.std() + 1e-8)  
    z = s / t
    z -= z.max()                           
    p = np.exp(z)
    return p / p.sum()
    
def sample(rng, policies: list[str], sample_n: int, sample_t: float) -> list[int]: 
    scores = [get_score(hash(policy)) for policy in policies]
    probs = _scores_to_probs(scores, sample_t)
    idx = rng.choice(len(probs), size=sample_n, replace=False, p=probs)
    return idx

def form_prompt(ids: list[int], policies: list[str]) -> list[str]: 
    prompt = ''
    for id_ in ids: prompt += policies[id_] + '\n\n'
    return prompt 

if __name__ == '__main__': 
    load_dotenv()
    create_policy_dir()

    # env params 
    task: str = "small"
    entropy: int = 2004
    train_episodes: int = 16
    holdout: str | int | list[int] = "dev"
    quick = False 
    n_jobs = -1 
    train = scoring.episode_set(task, train_episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
    held_out = scoring.episode_set(task, holdout, quick=quick, n_jobs=n_jobs)

    # TODO make this cli 
    # search params  
    iters = 20  
    alpha_model = False
    alpha = 0.25 # factor of proposal by strong (alpha) model                 
    alpha_i = int(iters / (iters * alpha))
    sample_n = 5
    sample_t = 0.01
    lm_t = 1.0
    top_p = 0.98
    all_policies = load_policies() 
    rng = np.random.default_rng(entropy)

    # sample => score 
    for i in range(iters): 
        if ((i % alpha_i) == 0) and alpha_model: 
            agent = AgentOpenAI('gpt-6.1-sol', 'low', None, None)        
        else: 
            agent = AgentOpenAI('gpt-6-luna', 'none', lm_t, top_p)   
        sample_ids = sample(rng, all_policies, sample_n, sample_t)
        sampled_policies = [all_policies[id_] for id_ in sample_ids]
        state = 'policies: ' + '\n\n'.join([policy for policy in sampled_policies])
        agent.inj_history(state)
        state = f'begin! current task is {task}'
        agent.inj_history(state)
        proposed_change, _ = agent.next_change() 
        all_policies.extend(proposed_change.changes)

        # evaluate fitness of changes 
        for change in proposed_change.changes: 
            id_ = hash(change)
            write_change(id_, change)
            try:
                score = fitness(id_, train)
                add_score(id_, score, None)
            except (ReplacmentError, Exception) as e: 
                score = -999 
                add_score(id_, score, str(e))

        # log
        print(f'iter: {i}, alpha: {i % alpha_i == 0}')
        print('sampled policies: ', [hash(policy) for policy in sampled_policies])
        print('their scores: ', [get_score(hash(policy)) for policy in sampled_policies])
        print(proposed_change.reasoning)
        print('proposed policeis: ', [hash(change) for change in proposed_change.changes])
        print('their scores: ', [get_score(hash(change)) for change in proposed_change.changes])
        print(agent.get_usage())