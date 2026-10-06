import os 
import hashlib 
import datetime

from rich import print 
from dotenv import load_dotenv
from sbf_starter import env_id, scoring

from .agent import AgentOpenAI, Change
from .changes import ChangesStorage

class ReplacmentError(Exception): 
    def __init__(self, *args):
        super().__init__(*args)

_policy_dir = './evolve/policies/' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
def create_policy_dir(): 
    os.makedirs(_policy_dir, exist_ok=True)

_base_policy_file = './evolve/context/policy.py'
def _get_def_policy(): 
    with open(_base_policy_file, 'r') as f: 
        return f.read()

def get_path(id_: str) -> str: 
    return _policy_dir + '/' + id_ + '.py'

def hash(text: str) -> str:
    return hashlib.blake2s(text.encode(), digest_size=6).hexdigest() 

def write_change(id_: str, change: Change):
    with open(get_path(id_), 'w') as f: 
        policy = _get_def_policy()
        new_policy = policy.replace(change.old_string, change.new_string)
        if policy == new_policy: 
            raise ReplacmentError('Content of files did not change and stays the same.')
        f.write(new_policy)

def fitness(id_: str, episodes): 
    return episodes.score(get_path(id_), cpu_budget=True).rss

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

    # search params 
    iters = 32

    # search loop 
    agent = AgentOpenAI()
    base_policy_score = train.score(_base_policy_file, cpu_budget=True).rss
    state = f'begin! current task is {task}, base policy score: {base_policy_score:.2f}'
    print(state)
    storage = ChangesStorage()
    all_changes = [] 
    for i in range(iters): 
        agent.inj_history(state)
        proposed_change, _ = agent.next_change() 
        all_changes.extend(proposed_change)
       
        # evaluate fitness of changes 
        results: list[str | int] = []
        for change in proposed_change.changes: 
            try: 
                id_ = hash(change.new_string) # create a hash based on the replacement string 
                write_change(id_, change)
                score = fitness(id_, train)
                results.append(score) 
            # todo maybe delete this 
            except (ReplacmentError, Exception) as e: 
                results.append(str(e))
                
        # return scores for changes 
        state = 'results: ' + ' '.join([f'change{i}: {res:.2f}' if isinstance(res, float) else f'change{i}: {res}' for i, res in enumerate(results)])

        print(f'iter: {i}')
        print(proposed_change.reasoning)
        print(f'changes count: {len(proposed_change.changes)}')
        print([hash(change.new_string) for change in proposed_change.changes])
        print(state)
    
    print(agent.get_usage())
