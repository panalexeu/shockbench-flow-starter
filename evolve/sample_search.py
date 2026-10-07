# sampling approach closer to the one presented AlphaEvolve 
import os 
import hashlib 
from pathlib import Path

from rich import print 
import numpy as np 
from dotenv import load_dotenv
from sbf_starter import  scoring

from .agent import AgentOpenAI, AgentAnthropic

_policy_dir = 'evolve/policies/policy1/'
_fail_policy_dir = _policy_dir + 'fails'
_error_prefix = '# error: '

def create_policy_dir(): 
    os.makedirs(_policy_dir, exist_ok=True)
    os.makedirs(_fail_policy_dir, exist_ok=True)

def get_path(id_: str) -> str: 
    return _policy_dir + '/' + id_ + '.py'

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

def load_fail_policies() -> list[str] :
    policies = []
    for path in Path(_fail_policy_dir).iterdir():
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
    
def sample_policies(rng, policies: list[str], sample_n: int, sample_t: float) -> list[int]: 
    if len(policies) == 0: return []
    if len(policies)-1 < sample_n: sample_n = len(policies)  

    scores = [get_score(hash(policy)) for policy in policies]
    probs = _scores_to_probs(scores, sample_t)
    idx = rng.choice(len(probs), size=sample_n, replace=False, p=probs)
    return idx

def sample_fail_policies(rng, policies: list[str], sample_e: int) -> list[int]:
    if len(policies) == 0: return []
    if len(policies)-1 < sample_e: sample_e = len(policies)  
    
    return rng.choice(len(policies), size=sample_e, replace=False)  # uniform sampling 

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
    iters = 12  
    alpha_model = True
    alpha = 0.25                 
    alpha_i = int(iters / (iters * alpha))
    sample_n = 3
    sample_e = 1 
    sample_t = 1.0
    lm_t = 1.0
    top_p = 0.98
    all_policies = load_policies() 
    fail_poilicies = load_fail_policies()
    rng = np.random.default_rng(entropy)

    # sample => score 
    for i in range(iters): 
        # model selection 
        is_alpha = ((i % alpha_i) == 0) and alpha_model
        if is_alpha: 
            agent = AgentOpenAI('gpt-6.1-sol', 'low', None, None)        
        else: 
            agent = AgentOpenAI('gpt-6-luna', 'none', lm_t, top_p)   

        #  sample policies => update state 
        sample_ids = sample_policies(rng, all_policies, sample_n, sample_t)
        sampled_policies = [all_policies[id_] for id_ in sample_ids]
        fail_sample_ids = sample_fail_policies(rng, fail_poilicies, sample_e)
        fail_sampled_policies = [fail_poilicies[id_] for id_ in fail_sample_ids]
        sampled_policies.extend(fail_sampled_policies)
        state = 'policies: ' + '\n\n'.join([policy for policy in sampled_policies])
        agent.inj_history(state)
        
        state = f'begin! current task is {task}'
        agent.inj_history(state)

        # evaluate fitness of changes
        proposed_change, _ = agent.next_change()  
        for change in proposed_change.changes: 
            id_ = hash(change)
            write_change(id_, change)
            try:
                score = fitness(id_, train)
                add_score(id_, score, None)
                all_policies.append(change)
            except (Exception) as e: 
                score = -999 
                add_score(id_, score, str(e))
                fail_poilicies.append(change)

        # log
        print(f'iter: {i}, alpha: {is_alpha}')
        print('sampled policies: ', [hash(policy) for policy in sampled_policies])
        print('their scores: ', [get_score(hash(policy)) for policy in sampled_policies])
        print(proposed_change.reasoning)
        print('proposed policeis: ', [hash(change) for change in proposed_change.changes])
        print('their scores: ', [get_score(hash(change)) for change in proposed_change.changes])
        print(agent.get_usage())
        