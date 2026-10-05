from rich import print 
from dotenv import load_dotenv

from .agent import AgentOpenAI
from .changes import ChangesStorage

if __name__ == '__main__': 
    load_dotenv()

    # search params 
    iters = 2

    # search loop 
    agent = AgentOpenAI()
    results = 'begin!'
    storage = ChangesStorage()
    all_changes = [] 
    for i in range(iters): 
        agent.inj_history(results)
        changes, _ = agent.next_changes() 
        all_changes.extend(changes)
        results = 'results: '
    storage.dump(all_changes)

    print(agent.get_usage())
