from rich import print 
from dotenv import load_dotenv

from .agent import AgentOpenAI

if __name__ == '__main__': 
    load_dotenv()

    # search params 
    iters = 2

    # search loop 
    agent = AgentOpenAI()
    agent.inj_history('begin!')
    for i in range(iters): 
        changes, usage = agent.next_changes() 
        breakpoint()
        agent.inj_history('')
