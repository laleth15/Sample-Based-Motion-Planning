import typing
import numpy as np
from networkx import Graph
from robot import Simple_Manipulator as Robot
from scipy.spatial import KDTree


from M1 import M1

def M2(robot: Robot, num_samples: int, num_neighbors: int) -> typing.Tuple[np.array, Graph]:
    """ Implement the PRM algorithm

    Parameters
    ----------
    robot : Robot
        our pybullet robot class
    num_samples : int
        number of samples in PRM
    num_neighbors : int
        number of closest neighbors to consider in PRM

    Returns
    -------
    typing.Tuple[np.array, Graph]
        np.array: 
            num_samples x 4 numpy array, sampled configurations in the roadmap (vertices)
        G: 
            a NetworkX graph object with weighted edges indicating the distance between connected nodes in the joint configuration space.
            This should be impelemented as an undirected graph.
    """
    
    samples = M1(robot.lower_lims, robot.upper_lims, num_samples)
    s = KDTree(samples)
    r = Graph()
    for i in range(len(samples)):
        r.add_node(i, pos=samples[i])
    for i, l in enumerate(samples):
        a, n = s.query(l, k=num_neighbors + 1)
        for j, m in enumerate(n[1:], start=1):  
            if robot.check_edge(samples[i], samples[m]):
                r.add_edge(i, m, weight=a[j])
    return samples, r