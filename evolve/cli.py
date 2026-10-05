import sys 

from rich import print

from .changes import ChangesStorage

if __name__ == '__main__': 
    file = sys.argv[1]
    storage = ChangesStorage()
    print(storage.load(file))
