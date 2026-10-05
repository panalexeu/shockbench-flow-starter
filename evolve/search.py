import os 
import shutil
import hashlib 

from rich import print 
from dotenv import load_dotenv
from sbf_starter import env_id, scoring

from .agent import AgentOpenAI, Change
from .changes import ChangesStorage

_tmp_dir = './evolve/tmp/'
def create_tmp_dir(): 
    os.makedirs(_tmp_dir, exist_ok=True)

def del_tmp_dir(): 
    shutil.rmtree(_tmp_dir) 

def _get_def_policy(): 
    with open('./evolve/context/policy.py', 'r') as f: 
        return f.read()

def get_path(id_: str) -> str: 
    return _tmp_dir + id_ + '.py'

def hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def write_change(id_: str, change: Change):
    with open(get_path(id_), 'w') as f: 
        policy = _get_def_policy()
        policy = policy.replace(change.string, change.string_replace)
        f.write(policy)

def fitness(id_: str, episodes): 
    return episodes.score(get_path(id_), cpu_budget=True).rss

if __name__ == '__main__': 
    load_dotenv()
    create_tmp_dir()

    # env params 
    task: str = "tiny"
    entropy: int = 2004
    train_episodes: int = 16
    holdout: str | int | list[int] = "dev"
    quick = False 
    n_jobs = -1 
    train = scoring.episode_set(task, train_episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
    held_out = scoring.episode_set(task, holdout, quick=quick, n_jobs=n_jobs)

    # search params 
    iters = 2

    # search loop 
    agent = AgentOpenAI()
    results = f'begin! current task is {task}'
    storage = ChangesStorage()
    all_changes = [] 
    for i in range(iters): 
        agent.inj_history(results)
        proposed_change, _ = agent.next_change() 
        all_changes.extend(proposed_change)

        # evaluate fitness of changes 
        scores = []
        for change in proposed_change.changes: 
            try: 
                id_ = hash(change.string_replace) # create a hash based on the replacement string 
                write_change(id_, change)
                score = fitness(id_, train)
            except Exception as e:
                print(str(e)) 
                score = -1
            finally: 
                scores.append(score)

        # return scores for changes 
        results = 'results: ' + ' '.join([f'change{i}: {score}' for i, score in enumerate(scores)])

    storage.dump(all_changes)
    print(agent.get_usage())

