# High-Performance-Computing-for-the-TSP

The code in tsp_python.pyx contains the code to run 3-opt heuristic for the TSP(Travelling Salesperon Problem). The use of Cython improves the speed of code. In addition, I applied multiprocessing to improve the speed of the code when applying the "Recurrent Nearest Neighbour" (RNN) algorithm 

There two main parts of the code. First the RNN algorithm is applied. To further minimise the tour 3-opt is then applied. 3-opt works by deleting 3 edges of the tour then reconnecting the tour by adding the best possible set of edges that will reduce the cost.

Interesting technical notes
- The number pathways to consider when applying k-opt is (k-1)!*2^(k-1)
- 3-opt includes 2-opt (this can be seen from the code). But despite this, 2-opt might outperform 3-opt due to matter of luck although this is unlikely.

### Execution Instructions:
The solver.py shows how to import the function and how to run execute the code on a command-line interface like Anaconda Prompt. Please note the tsp_python.pyx needs to build first before being used. This will result in a file called tsp_python.so. In addition, the compilation instructions are provided in setup.py file which will be used in the build process. For convience I provided the tsp_python.so file so that no building is required.     

### Data used
tsp_51_1 contains information on the node points that need to be visited in TSP. For more testing data I recommend access the Coursera course on Discrete Optimisation and downloading the relevant files there.   
