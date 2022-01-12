import numpy as np
import heapq as pq
import math
import random

class Maze:
    maze = None
    
    def generateMaze(self, grid, prob):
        self.maze = np.zeros((grid, grid), dtype=int)
        for i in range(grid):
            for j in range(grid):
                self.maze[i][j] = self.determineWall(prob)
        self.maze[0][0] = 0
        self.maze[grid-1][grid-1] = 0
    
    def determineWall(self, prob):
        #generate 1 with probability prob
        res = np.random.choice(np.arange(0,2), p =[1-prob, prob])
        return res   
    
    '''
    DFS algorithm utilizes FILO stack when prioritizing children to visit. 
    Returns the path found, number of nodes explored, and the set of visited nodes.
    ''' 
    def DFS(self, start, goal):
        fringe = [start]
        fringe_set = {start}
        visited = set()
        parent_map = [[None for i in range(self.maze.shape[0])] for i in range(self.maze.shape[0])]
        num_nodes = 0
        while len(fringe) > 0: 
            
            #get parent/newest cell and pop 
            curr = fringe.pop()
            visited.add(curr)
            if curr == goal:
                return self.reconstruct_path(goal, parent_map), num_nodes, visited
            num_nodes+=1
                
            #generate children up and down
            down = (curr[0] + 1, curr[1])
            up = (curr[0] -1, curr[1])
            if down[0] <= self.maze.shape[0] and down[1] <= self.maze.shape[0] and up[0] <= self.maze.shape[0] and up[1] <= self.maze.shape[0]:
                
                #if this is 0 only go down
                if curr[0] == 0:
                    down = (curr[0]+1, curr[1])
                    if (down not in visited) and self.maze[down[0]][down[1]] != 1 and down not in fringe_set:
                        fringe.append(down)
                        fringe_set.add(down)
                        #update parent_map so that the parent of down is curr                       
                        parent_map[down[0]][down[1]] = curr 
                        
                #if this is at the bottom, only go up
                elif curr[0] == self.maze.shape[0]-1:
                    up = (curr[0]-1, curr[1])
                    if (up not in visited) and self.maze[up[0]][up[1]] != 1 and up not in fringe_set:
                        fringe.append(up)
                        fringe_set.add(down)
                        #update parent_map so that the parent of up is curr                      
                        parent_map[up[0]][up[1]] = curr                        
                else: 
                    if (down not in visited) and self.maze[down[0]][down[1]] != 1 and down not in fringe_set:
                        fringe.append(down)
                        fringe_set.add(down)
                        parent_map[down[0]][down[1]] = curr                       
                    if (up not in visited) and self.maze[up[0]][up[1]] != 1 and up not in fringe_set:
                        fringe.append(up)      
                        fringe_set.add(up)
                        parent_map[up[0]][up[1]] = curr                  
        
            #generate children left and right
            right = (curr[0], curr[1]+1)
            left = (curr[0], curr[1]-1)
            if right[0] <= self.maze.shape[0] and right[1] <= self.maze.shape[0] and left[0] <= self.maze.shape[0] and left[1] <= self.maze.shape[0]:
                if curr[1] == 0:
                    if (right not in visited) and self.maze[right[0]][right[1]] != 1 and right not in fringe_set:
                        fringe.append(right) 
                        fringe_set.add(right)
                        parent_map[right[0]][right[1]] = curr                
                elif curr[1] == self.maze.shape[0]-1:
                    if (left not in visited) and self.maze[left[0]][left[1]] != 1 and left not in fringe_set:
                        fringe.append(left)
                        fringe_set.add(left)
                        parent_map[left[0]][left[1]] = curr          
                else:
                    if (left not in visited) and self.maze[left[0]][left[1]] != 1 and left not in fringe_set:
                        fringe.append(left)
                        fringe_set.add(left)
                        parent_map[left[0]][left[1]] = curr
                    right = (curr[0], curr[1]+1)
                    if (right not in visited) and self.maze[right[0]][right[1]] != 1 and right not in fringe_set:
                        fringe.append(right) 
                        fringe_set.add(right)
                        parent_map[right[0]][right[1]] = curr
                        
        return None, num_nodes, visited
    
    '''
    BFS algorithm utilizes FIFO queue when prioritizing children to visit. 
    Returns the path found, number of nodes traversed, and the set of visited nodes.
    ''' 
    def BFS(self, start, goal):
        fringe = [start]
        fringe_set = {start}
        visited = set()
        parent_map = [[None for i in range(self.maze.shape[0])] for i in range(self.maze.shape[0])]
        num_nodes = 0
        while len(fringe) > 0: 
            
            #get parent/newest cell and pop 
            curr = fringe.pop(0)
            visited.add(curr)
            if curr == goal:
                return self.reconstruct_path(goal, parent_map), num_nodes, visited    
            num_nodes+=1

            #generate children up and down
            down = (curr[0] + 1, curr[1])
            up = (curr[0] -1, curr[1])
            if down[0] <= self.maze.shape[0] and down[1] <= self.maze.shape[0] and up[0] <= self.maze.shape[0] and up[1] <= self.maze.shape[0]:
                
                #if this is 0 only go down
                if curr[0] == 0:
                    if (down not in fringe_set) and self.maze[down[0]][down[1]] != 1 and down not in visited:
                        fringe.append(down)
                        fringe_set.add(down)
                        #update parent_map so that the parent of down is curr                       
                        parent_map[down[0]][down[1]] = curr   
                        
                #if this is at the bottom, only go up
                elif curr[0] == self.maze.shape[0]-1:
                    if (up not in fringe_set) and self.maze[up[0]][up[1]] != 1 and up not in visited:
                        fringe.append(up)
                        fringe_set.add(up)
                        #update parent_map so that the parent of up is curr                      
                        parent_map[up[0]][up[1]] = curr                        
                else: 
                    if (down not in fringe_set) and self.maze[down[0]][down[1]] != 1 and down not in visited:
                        fringe.append(down)
                        fringe_set.add(down)
                        parent_map[down[0]][down[1]] = curr                       
                    if (up not in fringe_set) and self.maze[up[0]][up[1]] != 1 and up not in visited:
                        fringe.append(up)
                        fringe_set.add(up)
                        parent_map[up[0]][up[1]] = curr                  

            #generate children left and right
            right = (curr[0], curr[1]+1)
            left = (curr[0], curr[1]-1)
            if right[0] <= self.maze.shape[0] and right[1] <= self.maze.shape[0] and left[0] <= self.maze.shape[0] and left[1] <= self.maze.shape[0]:
                if curr[1] == 0:
                    if (right not in visited) and self.maze[right[0]][right[1]] != 1 and right not in fringe_set:
                        fringe.append(right) 
                        fringe_set.add(right)
                        parent_map[right[0]][right[1]] = curr                
                elif curr[1] == self.maze.shape[0]-1:
                    if (left not in visited) and self.maze[left[0]][left[1]] != 1 and left not in fringe_set:
                        fringe.append(left)
                        fringe_set.add(left)
                        parent_map[left[0]][left[1]] = curr          
                else:
                    if (left not in visited) and self.maze[left[0]][left[1]] != 1 and left not in fringe_set:
                        fringe.append(left)
                        fringe_set.add(left)
                        parent_map[left[0]][left[1]] = curr
                    right = (curr[0], curr[1]+1)
                    if (right not in visited) and self.maze[right[0]][right[1]] != 1 and right not in fringe_set:
                        fringe.append(right)     
                        fringe_set.add(right)
                        parent_map[right[0]][right[1]] = curr
                        
        return None, num_nodes, visited  
    
    '''
    Given node in either the (r,c) form or (priority, dist_to, (r,c)) form, and sets representing nodes that have been visited or are in the fringe, returns a list of valid, unvisited children.
    '''
    def generate_children(self, node, visited, fringe_set):
        if len(node) == 3:
            tple =  node[2]
        else:
            tple = node
            
        dim = self.maze.shape[0]
        
        neighbors = [(tple[0]+1, tple[1]), #up
                    (tple[0]-1, tple[1]), #down
                    (tple[0], tple[1]+1), #left
                    (tple[0], tple[1]-1)] #right
        children = []
        
        for tup in neighbors:
            if (tup[0]<0 or tup[0]>=dim or tup[1]<0 or tup[1]>=dim #out of bounds
                or self.maze[tup[0]][tup[1]] == 1 #filled
                or tup in visited #already explored
                or tup in fringe_set #already put in the fringe
               ):
                continue
            
            #creates child in the same dimension as node
            if len(node) == 3:
                child = (0,node[1]+1,tup) #dist_to = dist_to of parent + 1
            else:
                child = tup
                
            children.append(child)           
        
        return children

    '''
    Reconstructs the path to node to node using parent_map, which was generated by a search algorithm.
    A parent map is an array the same dimensions as the maze, where each spot denotes the parent through which the location was explored.
    The resulting path is a list of (r,c) tuples and includes the start and the goal.
    '''
    def reconstruct_path(self, node, parent_map):
        path = []
        crnt = node
        
        while crnt is not None:
            path.append(crnt)
            crnt = parent_map[crnt[0]][crnt[1]]
        
        path.reverse()
        
        return path

    '''
    Calculates the euclidean distance between loc1 and loc2.
    loc1 and loc2 are (r,c) tuples.
    '''
    def eucl_dist(self, loc1, loc2):
        return math.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)
   

    '''
    A_star uses a priority queue, ordered by the sum of the distance to the node so far (dist_to) and the euclidean distance.
    Returns the path found, number of nodes explored, and the set of visited nodes.
    Python's minheap module heapq is used for implementing the priority queue.
    '''
    def A_star(self, start, goal):
        start_node = (self.eucl_dist(start, goal), 0, start)
        parent_map = [[None for i in range(self.maze.shape[0])] for i in range(self.maze.shape[0])]
        visited = set()
        num_nodes = 0
        
        fringe = []
        pq.heappush(fringe, start_node)
        #set created to avoid duplicate insertions into the fringe
        ##if child is already in fringe, the one in the fringe will have a lower priority. Since the distances are all 1, it is not possible to find another shorter path to that child than the one already found and inserted into the fringe.
        fringe_set = {start_node[2]}
        
        while (len(fringe) > 0):
            crnt = pq.heappop(fringe)
            if crnt[2] == goal:
                return self.reconstruct_path(crnt[2], parent_map), num_nodes, visited

            children = self.generate_children(crnt, visited, fringe_set)
            for child in children:
                parent_map[child[2][0]][child[2][1]] = crnt[2]
                #update priority = eucl_dist + dist_to
                updated_child = (self.eucl_dist(child[2],goal) + child[1], child[1],child[2])
                pq.heappush(fringe, child)
                fringe_set.add(child[2])

            visited.add(crnt[2])
            num_nodes+=1

        return None, num_nodes, visited

    '''
----------------------------------------------------------------------------------------------------------
    Fire-maze functions
    
    Randomly select open block to be on fire
    Then agent moves, fire moves, agent moves, etc.
    
    Agent's movement:
        agent can move between open cells or stay in place
        agent cannot move into fire, and if fire moves to where agent is, agent dies
        
    Fire's movement:
        a free cell with no burning neighbors stays free, and a cell thats on fire stays on fire
        a free cell that has k burning neighbors, it will be on fire with probability 1-(1-q)^k
        fire can advance in the up, down, left, right directions
            
    How to use these functions: 
        Create maze using generateMaze
        Pick random fire init using random_fire_init
        While loop until a valid maze is obtained
        Solve the maze using one of the strategies
----------------------------------------------------------------------------------------------------------
    '''
    
    '''
    Returns random free spot in the maze to start the fire in.
    Does not include upper left or bottom right.
    '''
    def random_fire_init(self):
        possible_spots = []
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[0]):
                if (i == 0 and j == 0) or (i == self.maze.shape[0]-1 and j == self.maze.shape[0]-1):
                    continue
                if self.maze[i][j] == 0:
                    possible_spots.append((i,j))
        return random.choice(possible_spots)
    
    '''
    Checks whether the maze and location of the start of the fire is valid.
    Valid if there is a path from the start to the goal, and from the start to the fire.
    '''
    def valid_maze(self, fire_init):
        path,_,_ = self.DFS((0,0), (self.maze.shape[0]-1, self.maze.shape[0]-1))
        path_to_fire,_,_ = self.DFS((0,0), fire_init)
        return (path is not None and path_to_fire is not None)
    
    '''
    Given set of fire locations, fire_set, and flammability, q, returns a new set with the fire after advancing one step.
    '''
    def advance_fire(self, fire_set, q):
        new_fire_set = fire_set.copy()
        
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[0]):
                
                #if free and not on fire
                if self.maze[i][j] == 0 and self.maze[i][j] not in fire_set:
                    
                    neighbors = [(i+1,j), #up
                                 (i-1,j), #down
                                 (i,j-1), #left
                                 (i,j+1)] #right
                    
                    #count the neighbors that are on fire
                    k=0
                    for tup in neighbors:
                        if (tup[0]>=0 and tup[0]<self.maze.shape[0] and tup[1]>=0 and tup[1]<self.maze.shape[0] #in bounds
                            and self.maze[tup[0]][tup[1]] == 0 #free
                            and tup in fire_set #on fire
                           ):
                            k += 1
                    
                    #randomly assign whether location is on fire
                    prob = 1 - (1-q)**k
                    on_fire = np.random.choice([0,1], p =[1-prob, prob])
                    if on_fire == 1:
                        new_fire_set.add((i,j))                
                
        return new_fire_set
    
    '''
    Gets the shortest path from a given start (r,c) tuple to lower left goal, treating the fire as filled.
    Used in strategies 1 and 2.
    '''
    def get_path(self, start, fire_set):
        for fire_tup in fire_set:
            self.maze[fire_tup[0]][fire_tup[1]] = 1
        path, _, _ = self.A_star(start, (self.maze.shape[0]-1, self.maze.shape[0]-1))
        for fire_tup in fire_set:
            self.maze[fire_tup[0]][fire_tup[1]] = 0
        return path
    
    '''
    Given fire_init, solves the maze while modeling a fire with flammibility q using strategy 1
    If agent reaches goal, and then in the next time step, the fire takes the goal, agent is considered to have succeeded.
    When return_history is True, returns the path planned and a list representing the state of the fire through time.
    '''
    def solve_firemaze_strat1(self, fire_init, q, return_history):
        if not self.valid_maze(fire_init):
            raise Exception("The maze and fire_init is not valid.")
        
        fire_set = {fire_init}
        if return_history:
            fire_history = [fire_set]
        
        path = self.get_path((0,0), fire_set)
        if path is None or path[0] in fire_set:
            return (False, path, fire_history) if return_history else False
        
        #follow path with fire simulated
        t = 0
        while path[t] != (self.maze.shape[0]-1, self.maze.shape[0]-1):
            t+=1
            if path[t] in fire_set:
                return (False, path, fire_history) if return_history else False
            fire_set = self.advance_fire(fire_set, q)
            if return_history:
                fire_history.append(fire_set)
        
        return (True, path, fire_history) if return_history else True
     
    '''
    Given fire_init, solves the maze while modeling a fire with flammibility q using strategy 2
    If agent reaches goal, and then in the next time step, the fire takes the goal, agent is considered to have succeeded.
    When return_history is True, returns a list of the paths planned and a list representing the state of the fire through time.
    '''
    def solve_firemaze_strat2(self, fire_init, q, return_history):
        if not self.valid_maze(fire_init):
            raise Exception("The maze and fire_init is not valid.")
        
        fire_set = {fire_init}
        if return_history:
            fire_history = [fire_set]
            path_history = []   
        agent_loc = (0,0)
        
        while agent_loc != (self.maze.shape[0]-1, self.maze.shape[0]-1):
            if agent_loc in fire_set:
                return (False, path_history, fire_history) if return_history else False
            
            path = self.get_path(agent_loc, fire_set)
            if path is not None:
                agent_loc = path[1] #guarenteed to not be on fire at this point
            
            fire_set = self.advance_fire(fire_set, q)
            
            if return_history:
                path_history.append(path)
                fire_history.append(fire_set)
        
        return (True, path_history, fire_history) if return_history else True
     
    '''
    Not used in final report.
    Generates a cube that can be used as a heuristic in strategy 3. The first dimension represents time, while the other two represent the (r,c) location in the cube.
    depth indicates how many timesteps deep generate cube will be.
    Simulates a fire and fills cube accordingly:
        0 - free
        0.5 - simulated fire 
        1 - filled or on fire
    '''
    def generate_sim_cube1(self, fire_set, q, depth):
        fire_set_sim = fire_set.copy()
        
        #create cube as 3d array of 0's with cube[0] being copy of self.maze with elements of fire_set made into 1's
        cube = np.zeros((depth, self.maze.shape[0], self.maze.shape[0]))
        cube[0] = self.maze.copy()
        for fire in fire_set:
            cube[0][fire[0]][fire[1]] = 1
        
        for t in range(1, depth):
            fire_set_sim = self.advance_fire(fire_set_sim, q)
            
            for i in range(self.maze.shape[0]):
                for j in range(self.maze.shape[0]):
                    
                    if cube[t-1][i][j] == 0 and (i,j) in fire_set_sim:
                        cube[t][i][j] = 0.5
                    elif cube[t-1][i][j] == 0:
                        cube[t][i][j] = 0
                    elif cube[t-1][i][j] == 0.5:
                        cube[t][i][j] = 0.5
                    else:
                        cube[t][i][j] = 1    
        
        return cube
    
    '''
    Used in final report.
    Generates a cube that can be used as a heuristic in strategy 3. The first dimension represents time, while the other two represent the (r,c) location in the cube.
    depth indicates how many timesteps deep generate cube will be.
    Simulates a fire multiple times and averages the results together.
    After generation, each spot (t,r,c) in the cube is the probability of (r,c) being on fire (or filled) at a particular timestep t.
    '''
    def generate_sim_cube2(self, fire_set, q, depth):
        fire_set_sim = fire_set.copy()
        num_samples = 20
        
        #an array of cubes
        cubes = np.zeros((num_samples,depth,self.maze.shape[0], self.maze.shape[0]))
        
        for s in range(num_samples):
            fire_set_sim = fire_set.copy()
            #initialize the first layer of the ith cube 
            cubes[s][0] = self.maze.copy()
            for fire in fire_set:
                cubes[s][0][fire[0]][fire[1]] = 1
 
            for t in range(1, depth):
                fire_set_sim = self.advance_fire(fire_set_sim, q)
            
                for i in range(self.maze.shape[0]):
                    for j in range(self.maze.shape[0]):
                        if cubes[s][t-1][i][j] == 1 or (cubes[s][t-1][i][j] == 0 and (i,j) in fire_set_sim):
                            cubes[s][t][i][j] = 1
                            
        #average the cubes together (spots that are filled or definitely on fire should be 1)
        cube = np.average(cubes, axis = 0)
                            
        return cube
    
    '''
    Given a node (which includes the timestep), generates the valid children in the next timestep, initializing their priorities.
    A cube is a 3D representation of the maze over time. The first dimension represents time, while the other two represent the (r,c) location in the cube. Each spot is an indication how likely that location and time is of being on fire or filled.
        0 - least likely to run into fire
        1 - filled or definitely on fire
    Utilizes the FireNode class defined below the Maze class.
    Used in strategy 3.
    Assumes the t+1 layer exists.
    '''
    def generate_children_cube(self, node, cube):
        goal = (self.maze.shape[0]-1, self.maze.shape[0]-1)
        t = node.t + 1
        tple = node.r_c
        
        dim = self.maze.shape[0]
        
        neighbors = [(tple[0]+1, tple[1]), #up
                    (tple[0]-1, tple[1]), #down
                    (tple[0], tple[1]+1), #left
                    (tple[0], tple[1]-1), #right
                    (tple[0], tple[1])] #still
        
        children = []
        
        for tup in neighbors:
            if (tup[0]>=0 and tup[0]<dim and tup[1]>=0 and tup[1]<dim and #in bounds
                cube[t][tup[0]][tup[1]] < 1 #not on fire or filled
               ):
                priority = self.eucl_dist(tup, goal) + 3*cube[t][tup[0]][tup[1]]
                child = Fnode(priority, tup, t, node)
                children.append(child)
        
        return children
    
    '''
    Reconstructs the path that leads to node following the parent field down the linked list.
    Used in strategy 3.
    '''
    def reconstruct_path_cube(self, node):
        path = []
        crnt = node
        
        while crnt is not None:
            path.append(crnt.r_c)
            crnt = crnt.parent
            
        path.reverse()
        
        return path    
    
    '''
    Returns a path from start that avoids the predicted fire while getting closer to the goal.
    Uses a priority queue, ordered by a linear combination of the euclidean distance to the goal and fire heuristic given by the cube.
    generate_cube is a function pointer indicating which kind of cube should be generated to search through.
    Python's minheap module heapq is used for implementing the priority queue.
    Used in strategy 3.
    '''
    def plan(self, start, fire_set, q, depth, generate_cube):
        goal = (self.maze.shape[0]-1, self.maze.shape[0]-1)
        cube = generate_cube(fire_set, q, depth)
        start_priority = self.eucl_dist(start, goal) + 3*cube[0][start[0]][start[1]]
        start_node = Fnode(start_priority, start, 0, None)
        
        fringe = []
        pq.heappush(fringe, start_node)
        
        while len(fringe)>0:
            crnt = pq.heappop(fringe)
            if crnt.r_c == goal or crnt.t+1>=len(cube): 
                return self.reconstruct_path_cube(crnt)
            children = self.generate_children_cube(crnt, cube)
            for child in children:
                pq.heappush(fringe, child)
        
        return None 
    
    
    '''
    Given fire_init, solves the maze while modeling a fire with flammibility q using strategy 3.
    If agent reaches goal, and then in the next time step, the fire takes the goal, agent is considered to have succeeded.
    When return_history is True, returns the path planned and a list representing the state of the fire through time.
    generate_cube is a function pointer indicating which kind of cube should be generated for the planning step.
    '''
    def solve_firemaze_strat3(self, fire_init, q, return_history, generate_cube):
        if not self.valid_maze(fire_init):
            raise Exception("The maze and fire_init is not valid.")
        
        fire_set = {fire_init}
        if return_history:
            fire_history = [fire_set]
            path_history = []   
        agent_loc = (0,0)
        
        while agent_loc != (self.maze.shape[0]-1, self.maze.shape[0]-1):
            if agent_loc in fire_set:
                return (False, path_history, fire_history) if return_history else False
            
            path = self.plan(agent_loc, fire_set, q, 10, generate_cube)
            if path is not None:
                agent_loc = path[1]
            
            fire_set = self.advance_fire(fire_set, q)
            
            if return_history:
                path_history.append(path)
                fire_history.append(fire_set)
        
        return (True, path_history, fire_history) if return_history else True
    
'''
priority(t,r,c) = euclidean distance to goal + w*cube(t,r,c)
r_c is a (r,c) tuple picking out a location on the maze
t is the time step
parent is an Fnode
'''
class Fnode:
    def __init__(self, priority, r_c, t, parent):
        self.priority = priority
        self.r_c = r_c
        self.t = t
        self.parent = parent
    
    def __lt__(self, other):
        return self.priority < other.priority
