# deprecated for now 
import os 
import pickle 
import datetime 

from .agent import Changes

class ChangesStorage:
    def __init__(self):
        self._data_dir = './evolve/data/'
        os.makedirs(self._data_dir, exist_ok=True)

    def dump(self, changes: list[Changes]):
        name = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.pkl'
        with open(os.path.join(self._data_dir, name), 'wb') as f:
            pickle.dump(changes, f)

    def load(self, file: str) -> list[Changes]:
        with open(file, 'rb') as f:
            return pickle.load(f)
        