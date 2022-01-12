# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 11:56:42 2021

@author: asyoo
The following are functions used in helping to test and plot maze.py
"""

import maze as m 
from statistics import mean
import matplotlib.pyplot as plt

'''Function that returns boolean, "True" indicating success in finding solution to maze
    Arguments: strategy should take an int, either 1 or 2, so as to indicate which strategy it's in regards to;
    q is flammability of the maze.
'''
def success(strategy, p, q):   
    ma = m.Maze()
    grid = 10
    ma.generateMaze(grid, p)
    start = (0,0)
    goal = (grid-1, grid-1)
    fire_init = ma.random_fire_init()
    if not ma.valid_maze(fire_init):
       # print("error")
        return 'error'
    if strategy == 1 or strategy == 2:     
        
        #while loop to check for valid maze       
        if strategy == 1: 
           if ma.solve_firemaze_strat1(fire_init, q, False) == True:
               return 'True'
        elif strategy == 2: 
           if ma.solve_firemaze_strat2(fire_init, q, False) == True:
               return 'True'     
    elif strategy == 3:
       fire_init = ma.random_fire_init()
       
       if ma.solve_firemaze_strat3(fire_init, q, True, ma.generate_sim_cube2) == True:
            return 'True'
    elif strategy == 'BFS':
        if ma.BFS(start, goal) is not None:
            return 'True'
        else:
            return 'False'
    elif strategy == 'DFS':
        if ma.DFS(start, goal) is not None:
            return 'True'
        else:
            return 'False'
    elif strategy == 'a':
        if ma.A_star(start,goal) is not None:
            return 'True'
        else:
            return 'False'
    else: return 'False'



'''Prints the number of successes in 50 runs of a strategy given some flammability, q'''
def runxtimes(strategy, p, q, x):
    y = 0
    e = 0
    f = 0
    if strategy == 1 or strategy == 2:            
        for i in range(0, x, 1):
            if success(strategy,p, q) == 'True':
               # print('success')
                y+=1  
            elif success(strategy, p, q) == 'error':
                e+=1
                #print("error")
            else:
                f += 1
                #print('fail')
        print('Number of successes: ', y, 'Number of errors: ', e, 'Number of fails: ', f)
    elif strategy == 3:
        for i in range(0, x, 1):
            if success(strategy,p, q) == 'True':
                #print('success')
                y+=1  
            elif success(strategy, p, q) == 'error':
                e+=1
                #print("error")
            else:
                f += 1
                #print('fail')
        print('Number of successes: ', y, 'Number of errors: ', e, 'Number of fails: ', f)
    elif strategy == 'BFS':
        for i in range(0, x, 1):
            if success(strategy,p, q) == 'True':
                #print('success')
                y+=1  
            elif success(strategy, p, q) == 'error':
                e+=1
                #print("error")
            else:
                f += 1
                #print('fail')
        print('Number of successes: ', y, 'Number of errors: ', e, 'Number of fails: ', f)
    elif strategy == 'DFS':
        for i in range(0, x, 1):
            if success(strategy,p, q) == 'True':
                #print('success')
                y+=1  
            elif success(strategy, p, q) == 'error':
                e+=1
                #print("error")
            else:
                f += 1
                #print('fail')
        print('Number of successes: ', y, 'Number of errors: ', e, 'Number of fails: ', f)
    elif strategy == 'a':
        for i in range(0, x, 1):
            if success(strategy,p, q) == 'True':
                #print('success')
                y+=1  
            elif success(strategy, p, q) == 'error':
                e+=1
                #print("error")
            else:
                #print('fail')
                f += 1
        print('Number of successes: ', y, 'Number of errors: ', e, 'Number of fails: ', f)
        
    return None
   
#script to run a given algorithm x number of times with given obstacle probability p 
def prob3(strategy, p, x):
    y = 0
    e = 0
    f = 0
    nodes = []
    grid = 100
    ma = m.Maze()
   
    start = (0,0)
    goal = (grid-1, grid-1)
    if strategy == 'BFS':            
        for i in range(0, x, 1):
            ma.generateMaze(grid, p)
            if success(strategy,p, 0) == 'True':
               # print('success')
                nodes.append(ma.BFS(start, goal)[1])  
                y+=1  
            elif success(strategy, p, 0) == 'error':
                e+=1
                #print("error")
            else:
                f += 1
                #print('fail')
        print('Number of successes: ', y, 'Number of errors: ', e, 'Number of fails: ', f)
        nodeAvg = mean(nodes)
        print('List of nodes: ', nodes)
        print('Average: ', nodeAvg)
    elif strategy == 'a':
        for i in range(0, x, 1):
            if success(strategy,p, 0) == 'True':
                #print('success')
                nodes.append(ma.BFS(start, goal)[1])  
                y+=1  
            elif success(strategy, p, 0) == 'error':
                e+=1
                #print("error")
            else:
                #print('fail')
                f += 1
        nodeAvg = mean(nodes)
        print('Average: ', nodeAvg)
        print('List of nodes: ', nodes)
        print('Number of successes: ', y, 'Number of errors: ', e, 'Number of fails: ', f)
        
'''Ended up not using the code below to produce plots.
We found that graphs produced through excel were more aesthetically pleasing and so switched.'''
        
def main():
    '''
    For each strategy and every q, will generate 10 mazes, restarting with randomized new initial fire locations. 
    After plots are generated, results will be plotted on 'average successes vs flammability q'
    '''
    s1 = []
    s2 = []
    flammability = []
    
    for r in range(1, 1, 10): 
        q = float(r/10)
        flammability[r] = q
        s1[r-1]=success(1, q)
        print('hello')
        print('s1: ', s1)
        s2[r-1]=success(1, q)
        print('s2: ', s2)

    #strategy 1: generate and solve at least 10 mazes restarting each 10 times with new initial fire locations
    #generate 10 plots and store the success/failure in list 
    
    '''for every q (increment by 0.1) until q=1
            find if there is a solution, repeat 10x, increment number of successes, store total successes in s1
    '''
    
    for q in range (0.1, 1, 0.1):
        temp = 0
        for i in range(1, 10, 1):
            if success() == True:
                temp+=1
        s1[q*10] = temp/10      
    
    #strategy 1, generate plot of 'average successes vs flammability q' 
    
    plt.plot(flammability, s1)
    plt.title('Average Number of Successes vs Flammability, q in Strategy 1')
    plt.xlabel('Flammability, q')
    plt.ylabel('Average Successes')
    plt.show
    
    #strategy 2: generate and solve at least 10 mazes restarting each 10 times with new initial fire locations
    
    for q in range (0.1, 1, 0.1):
        temp = 0
        for i in range(1, 10, 1):
            if success() == True:
                temp+=1
        s2[q*10] = temp/10      
    
    #strategy 2, generate plot of 'average successes vs flammability q' 
    
    plt.plot(flammability, s2)
    plt.title('Average Number of Successes vs Flammability, q in Strategy 1')
    plt.xlabel('Flammability, q')
    plt.ylabel('Average Successes')
    plt.show

    
    #code below is for replotting A_star and BFS, num_nodes vs p
    
    m = Maze()
    p_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    total_nodes_Astar = [0 for x in range(10)]
    total_nodes_BFS = [0 for x in range(10)]

    for i in range(10):
        print(i)
        for j in range(100):
            m.generateMaze(100, p_list[i])
        
            _,num_nodes_A,_ = m.A_star((0,0), (99,99))
            total_nodes_Astar[i]+=num_nodes_A

    print("Starting BFS now")      

    for i in range(10):
        print(i)
        for j in range(100):
            m.generateMaze(100, p_list[i])
        
            _,num_nodes_BFS,_ = m.BFS((0,0), (99,99))
            total_nodes_BFS[i]+=num_nodes_BFS
    