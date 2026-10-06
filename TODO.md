1. track token usage
2. use smaller model for polishing and bigger model for initial solution. e.g. bigger model 5 iterations than 30 iterations of smaller model.
3. add branching the best individual pool 
4. highlight adaptivity of this approach  
5. switch models 
6. evolution of a search algorithm: "whereas for problems with non-symmetric solutions it works better to evolve customized search algorithms"
7. add critique that steers agent in the iterative loop
8. prompt sampler (it shuold not contain achieved scores)

```text
The construction directly: the code literally spells out the object, such as a list of coordinates.
A constructor function: a short program that builds the object from a rule. This suits symmetric solutions.
A search heuristic: a program that searches for the object within a time budget. This suits irregular solutions.
```