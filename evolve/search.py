from rich import print 
from dotenv import load_dotenv

from .agent import AgentOpenAI

if __name__ == '__main__': 
    load_dotenv()
    agent = AgentOpenAI()
    agent.inj_history('pls remember that my name is oleksii')
    changes, usage = agent.next_changes() 
    agent.inj_history('what is my name?')
    changes, usage = agent.next_changes() 
    print(agent.history)
    print(usage)