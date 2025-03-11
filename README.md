# High-Performance-Computing-for-the-TSP

The code in tsp_python.pyx contains the code to run 3-opt heuristic for the TSP(Travelling Salesperon Problem). The use of Cython improves the speed of code.

There two main parts of the code. First the recurrent nearest neighbour algorithm is applied. To further minimise the tour 3-opt is then applied. 3-opt works by deleting 3 edges of the tour then reconnecting the tour by adding the best possible set of edges that will reduce the cost.

Interesting technical notes
- The number pathways to consider when applying k-opt is (k-1)!*2^(k-1)
- 3-opt includes 2-opt (this can be seen from the code). But despite this it 2-opt might outperform due to matter of luck although this is unlikely.  
