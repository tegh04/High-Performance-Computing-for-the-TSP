# cython: language_level=3, boundscheck=True
import numpy as np
import random
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow, exp
from collections import namedtuple
from multiprocessing import Pool, cpu_count, shared_memory
import time

np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def calculate_distance_matrix(np.ndarray[DTYPE_t, ndim=2] points):
    """Calculate the distance matrix for all points."""
    cdef unsigned long int num_points = len(points)
    cdef unsigned int i
    cdef unsigned int j
    # Shared memory of double precision (float64)
    shm = shared_memory.SharedMemory(create=True, size=num_points * num_points * 8)  
    cdef np.ndarray shared_distance_matrix = np.ndarray((num_points, num_points), dtype=DTYPE, buffer=shm.buf)

    # Calculate and fill the distance matrix
    for i in range(num_points):
        for j in range(i + 1, num_points):  # Only calculate the upper triangle (symmetric matrix)
            dist = sqrt(pow(points[i, 0] - points[j, 0], 2) + pow(points[i, 1] - points[j, 1], 2))
            shared_distance_matrix[i, j] = dist
            shared_distance_matrix[j, i] = dist  # Symmetric entry
    return shm, shared_distance_matrix
    
def nearest_neighbor(start_index, shm_name, shape):
    """Generate a tour using the nearest neighbor heuristic starting from a given point using a shared memory distance matrix."""
    cdef unsigned int num_points = shape[0]
    
    # Access the shared memory for the distance matrix
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cdef np.ndarray[DTYPE_t, ndim=2] distance_matrix = np.ndarray(shape, dtype=DTYPE, buffer=existing_shm.buf)
    
    visited = np.zeros(num_points, dtype=bool)
    tour = [start_index]
    current_index = start_index
    visited[start_index] = True
    total_distance = 0.0
    
    for _ in range(num_points - 1):
        distances_to_unvisited = np.where(visited, np.inf, distance_matrix[current_index])
        nearest_index = np.argmin(distances_to_unvisited)
        nearest_distance = distances_to_unvisited[nearest_index]
        
        tour.append(nearest_index)
        total_distance = total_distance + nearest_distance
        visited[nearest_index] = True
        current_index = nearest_index

    # Return to the start point
    total_distance += distance_matrix[current_index][tour[0]]

    return tour, total_distance
    
"""
path: list variable
length: number i.e.(length of the given path)
shm_name: used to access shared distance matrix
"""    
def three_opt(path, length, shm_name):
    
    # create distance matrix
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    cdef unsigned int n = len(path)
    shape = (n, n)
    
    cdef np.ndarray[DTYPE_t, ndim=2] dist_matrix = np.ndarray(shape, buffer=existing_shm.buf)
    
    improved = True
    cdef unsigned int i, j, k 
    
    num_iterations = 0
    while improved and num_iterations<2_000_000:
        improved = False
        
        for i in range(1, n - 5):
            for j in range(i + 2, n - 3):
                for k in range(j + 2, n):
                    num_iterations = num_iterations + 1
                    # compute the cost for each of 8 possible paths that can be produced by k-opt
                    costs = [dist_matrix[path[i-1],path[i]] + dist_matrix[path[j-1],path[j]] + dist_matrix[path[k-1],path[k]], 
                             dist_matrix[path[i-1],path[j-1]] + dist_matrix[path[i],path[j]] + dist_matrix[path[k-1],path[k]],
                             dist_matrix[path[i-1],path[i]] + dist_matrix[path[j-1],path[k-1]] + dist_matrix[path[j],path[k]],
                             dist_matrix[path[i-1],path[j-1]] + dist_matrix[path[i],path[k-1]] + dist_matrix[path[j],path[k]],
                             dist_matrix[path[i-1],path[j]] + dist_matrix[path[k-1],path[i]] + dist_matrix[path[j-1],path[k]],
                             dist_matrix[path[i-1],path[k-1]] + dist_matrix[path[j],path[i]] + dist_matrix[path[j-1],path[k]],
                             dist_matrix[path[i-1],path[j]] + dist_matrix[path[k-1],path[j-1]] + dist_matrix[path[i],path[k]],
                             dist_matrix[path[i-1],path[k-1]] + dist_matrix[path[j],path[j-1]] + dist_matrix[path[i],path[k]]]
                    #print(f'costs: {costs}')
                    
                    best_index = np.argmin(costs)
                    # select the path with the best possible cost
                    
                    # if best_index = 0 no change to the path is made meaning no better path was found
                    if best_index != 0:
                        if best_index == 1:
                            # reverse first segment
                            # dist_matrix[i-1][j-1] + dist_matrix[i][j] + dist_matrix[k-1][k]
                            path = path[:i] + path[i:j][::-1] + path[j:k] + path[k:]
                        elif best_index == 2:
                            # reverse second segment
                            # dist_matrix[i-1][i] + dist_matrix[j-1][k-1] + dist_matrix[j][k]
                            path = path[:i] + path[i:j] + path[j:k][::-1] + path[k:]
                        elif best_index == 3:
                            # reverse both segments
                            # dist_matrix[i-1][j-1] + dist_matrix[i][k-1] + dist_matrix[j][k]
                            path = path[:i] + path[i:j][::-1] + path[j:k][::-1] + path[k:]
                        elif best_index == 4:
                            # swap both segments
                            # dist_matrix[i-1][j] + dist_matrix[k-1][i] + dist_matrix[j-1][k]
                            path = path[:i] + path[j:k] + path[i:j] + path[k:]
                        elif best_index == 5:
                            # swap both segments and reverse second segment
                            # dist_matrix[i-1][k-1] + dist_matrix[j][i] + dist_matrix[j-1][k]
                            path = path[:i] + path[j:k][::-1] + path[i:j] + path[k:]
                        elif best_index == 6:
                            # swap both segments and reverse first segment
                            # dist_matrix[i-1][j] + dist_matrix[k-1][j-1] + dist_matrix[i][k]
                            path = path[:i] + path[j:k] + path[i:j][::-1] + path[k:]
                        else:
                            # swap both segments and reverse both segments
                            # dist_matrix[i-1][k-1] + dist_matrix[j][j-1] + dist_matrix[i][k]
                            path = path[:i] + path[j:k][::-1] + path[i:j][::-1] + path[k:]
                        improved = True
                        length = length + costs[best_index] - costs[0]
                        break
                         
                if improved:
                    break

    return path, length
    
"""
This is the full algorithm used to solve the TSP problem
The code proceeds by applying the recurrent nearest-neighbor heuristic. Thinking about how to apply simulated annealing and three-opt
"""
def solve_it(input_data):

    # Process the data to get an array of points
    Point = namedtuple("Point", ['x', 'y'])
    # parse the input
    lines = input_data.split('\n')
    node_count = int(lines[0])
    print(f'node_count: {node_count}')
    list_of_points = []
    for i in range(1, node_count+1):
        line = lines[i]
        parts = line.split()
        list_of_points.append(Point(float(parts[0]), float(parts[1])))
    cdef np.ndarray[DTYPE_t, ndim=2] points = np.array(list_of_points)
    
    # create distance matrix
    start = time.perf_counter()
    shm, shared_distance_matrix = calculate_distance_matrix(points)
    shape = (node_count, node_count)
    finish = time.perf_counter()
    print(f"The finish time of creating the distance matrix: {finish - start}")
    del points
    
    # Prepare arguments for parallel processing
    args_list = [(start_index, shm.name, shape) for start_index in range(node_count)[:2000]]

    start = time.perf_counter()

    # Run nearest-neighbor in parallel using starmap
    with Pool(processes=8) as pool:
        results = pool.starmap(nearest_neighbor, args_list)
    finish = time.perf_counter()
    print(f"Finish time recurrent nearest neighbors: {finish - start}")
    
    # Find the best tour from recurrent nearest-neighbor heuristic 
    best_path, best_length = min(results, key=lambda x: x[1])
    
    """
    # can't uncomment the line below because three opt uses path as a list not array
    #current_path = np.array(current_path, dtype=np.uint16)
    """
    # Run three-opt
    if node_count<=10_000:
        start = time.perf_counter()
        best_path, best_length = three_opt(best_path, best_length, shm.name)
        finish = time.perf_counter()
        print(f"The finish time of executing three opt: {finish - start}")

    # Clean up shared memory
    shm.close()
    shm.unlink()

    output_data = f'{best_length:.2f} 0\n'
    output_data += ' '.join(map(str, best_path))
    return output_data
