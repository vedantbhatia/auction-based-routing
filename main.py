import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree as MST
from scipy.sparse import csr_matrix
from tqdm import tqdm

num_robots = 0
num_targets = 0
verbose = False

printif = print if verbose else lambda *a, **k: None


class robot:
    def __init__(self,id,targets,pos=None,mode='ave'):
        self.id = id
        self.pos = np.random.uniform(0,1,2) if pos is None else pos
        self.targets = targets
        self.allocated = np.zeros((len(targets)))
        self.mode = mode
        self.my_tasks = []
    def cost(self,a, b):
        return (np.linalg.norm(a - b))

    def bid(self):
        printif("Current allocation table for robot {0} = ".format(self.id),self.allocated)
        bids = []
        current_unallocated = np.where(self.allocated==0)[0]
        for i in current_unallocated:
            if(self.mode=='sum'):
                all_costs = [self.cost(self.pos,self.targets[i])]
                for j in self.my_tasks:
                    all_costs.append(self.cost(self.targets[j],self.targets[i]))
                bids.append(min(all_costs))
            elif(self.mode=='ave'):
                bids.append(self.cost(self.pos,self.targets[i]))
            else:
                tree_cost = self.computeMST()
                all_costs = [self.cost(self.pos, self.targets[i])]
                for j in self.my_tasks:
                    all_costs.append(self.cost(self.targets[j], self.targets[i]))
                bids.append(min(all_costs)+tree_cost)
        bids = np.asarray(bids)
        index = bids.argmin()
        val = bids.min()
        bids = np.ones((bids.shape[0]))*np.inf
        bids[index]=val
        printif("bidding {0} for {1}".format(bids.tolist(),current_unallocated))
        return(bids)

    def computeMST(self):
        graph = np.zeros((len(self.my_tasks) + 1,len(self.my_tasks) + 1))
        for j,i in enumerate(self.my_tasks):
            graph[0, j + 1] = self.cost(self.pos, self.targets[i])
            graph[j+1,0]=graph[0,j+1]
            for j2,i2 in enumerate(self.my_tasks):
                if (i == i2):
                    continue
                graph[j + 1, j2 + 1] = self.cost(self.targets[i], self.targets[i2])
                graph[j2+1,j+1] = graph[j+1,j2+1]
        printif(graph.tolist())
        graph = csr_matrix(graph.tolist())
        mst = MST(graph).toarray()
        printif(mst)
        return (np.sum(mst))

    def winningBids(self,bids):
        current_unallocated = np.where(self.allocated==0)[0]
        printif("current unallocated robot {0} = ".format(self.id),current_unallocated)
        assert len(current_unallocated)==bids.shape[1]
        t = np.argmin(bids)
        min_robot = t//bids.shape[1]+1
        min_bid = t%bids.shape[1]
        printif("winning robot {0}, winning bid id {1} for target {2} ".format(min_robot,min_bid,current_unallocated[min_bid]))

        self.allocated[current_unallocated[min_bid]]=min_robot
        if(min_robot==self.id):
            self.my_tasks.append(current_unallocated[min_bid])
            printif("robot {0}'s tasks now: {1}".format(self.id,self.my_tasks))


def constructTestGrid(n,beta):
    pass

def test1(mode,epsilon=0.5):

    global num_robots
    global num_targets
    num_robots = 2
    num_targets = 2

    targets = []
    target1 = np.zeros((2))
    target2 = np.zeros((2))
    target1[0] = 1 + epsilon
    target2[0] = 3 + 2 * epsilon
    targets.append(target1)
    targets.append(target2)

    pos1 = np.zeros((2))
    robot1 = robot(1,targets,pos1,mode)

    pos2 = np.zeros((2))
    pos2[0] = 2+epsilon
    robot2 = robot(2,targets,pos2,mode)

    return([robot1,robot2])

def test2(mode,m=5,n=5,epsilon = 0.1, beta = 0.1):

    global num_robots
    global num_targets
    num_robots = m
    num_targets = n
    targets = [np.asarray([beta*i,1]) for i in range(m)]
    targets[0][1]=1-epsilon
    printif(targets)
    robots_pos = [np.asarray([beta*i,0]) for i in range(n)]
    robots = [robot(i+1,targets,robots_pos[i],mode) for i in range(n)]
    return(robots)

def test3(mode,n=2,beta=1):

    global num_robots
    global num_targets
    num_robots = n
    num_targets = n*n*n

    targets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                t = np.array([1+i*(beta),1+j*(beta+1)])
                targets.append(t)
    print(targets)
    robots = [robot(i+1,targets,np.ones((2)),mode) for i in range(n)]
    return(robots)

def driver():
    mode = 'max'
    # robots = test1(mode)
    robots = test2(mode)
    robots = test3(mode)
    for i in tqdm(range(num_targets)):
        all_bids = []
        for robot in robots:
            all_bids.append(robot.bid())
        all_bids = np.asarray(all_bids).reshape(num_robots,len(all_bids[0]))
        for robot in robots:
            robot.winningBids(all_bids)

    allocation = robots[0].allocated
    print(allocation)


driver()




