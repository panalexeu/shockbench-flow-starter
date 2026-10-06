import os 
import shutil 
import hashlib 
import datetime
from pathlib import Path

from rich import print 
from dotenv import load_dotenv
from sbf_starter import  scoring

from .agent import AgentOpenAI, Change

class ReplacmentError(Exception): 
    def __init__(self, *args):
        super().__init__(*args)

_policy_dir = './evolve/policies/' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
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

def hash(text: str) -> str:
    return hashlib.blake2s(text.encode(), digest_size=6).hexdigest() 

def add_score(id_: str, score: float,):  
    score_str = f'# {score}\n'
    with open(get_path(id_), 'r') as f: lines = f.readlines()
    lines.insert(0, score_str)
    with open(get_path(id_), 'w') as f: f.write(''.join(lines)) 

def get_score(id_: str) -> float: 
    with open(get_path(id_), 'r') as f: lines = f.readlines()
    score = float(lines[0].lstrip("#").strip())
    return score 

def write_change(id_: str, change: Change):
    with open(get_path(id_), 'w') as f: 
        policy = _get_def_policy()
        new_policy = policy.replace(change.old_string, change.new_string)
        f.write(new_policy)
        if policy == new_policy: 
            raise ReplacmentError('Content of the files did not change and stays the same.')

def fitness(id_: str, episodes): 
    return episodes.score(get_path(id_), cpu_budget=True).rss

if __name__ == '__main__': 
    load_dotenv()
    create_policy_dir()
    create_candidate_dir() 

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
    iters = 16                # how many iterations a lm is provided to improve the policy  
    keep_candidates = 1       # how many best scoring candidates are stored in the candidates pool after all iterations are completed 
    provide_pool = False      # wether to provide candidates from the best candidates pool to the lm
    pool_size = 5             # how many pool candidates are provided to the lm 
    score_init_policy = False

    # search loop 
    agent = AgentOpenAI()
    if provide_pool: 
        pool = get_candidates_pool(pool_size)
        agent.inj_history(f'below are the candidates that scored the best so far: {pool}')
    state = f'begin! current task is {task}'
    if score_init_policy: 
        init_policy_score = train.score(_base_policy_file, cpu_budget=True).rss
        state += f'init policy score: {init_policy_score:.2f}'

    all_changes = [] 
    for i in range(iters): 
        agent.inj_history(state)
        proposed_change, _ = agent.next_change() 
        all_changes.extend(proposed_change.changes)

        # evaluate fitness of changes 
        results: list[str | int] = []
        for change in proposed_change.changes: 
            try: 
                id_ = hash(change.new_string) # create a hash based on the replacement string 
                write_change(id_, change)
                score = fitness(id_, train)
                add_score(id_, score)
                results.append(score) 
            except (ReplacmentError, Exception) as e: 
                score = -1
                add_score(id_, score)
                results.append(str(e))
                
        # update  state with changes scores 
        state = 'results: ' + ' '.join([f'change{i}: {res:.2f}' if isinstance(res, float) else f'change{i}: {res}' for i, res in enumerate(results)])

        # log
        print(f'iter: {i}')
        print(proposed_change.reasoning)
        print(f'changes count: {len(proposed_change.changes)}')
        print([hash(change.new_string) for change in proposed_change.changes])
        print(state)
        print(agent.get_usage())

    # save the best candidates into the candidates pool 
    changes_dict = {} 
    for change in all_changes: 
        id_ = hash(change.new_string)
        score = get_score(id_) 
        changes_dict[id_] = score 
    sorted_changes = sorted(changes_dict.items(), key=lambda kv: kv[1], reverse=True)
    best_candidates = sorted_changes[:keep_candidates]
    for cnd in best_candidates: 
        id_ = cnd[0]
        shutil.move(get_path(id_), _candidates_dir)
