from networkx import Graph, shortest_path
import numpy as np
from robot import Simple_Manipulator as Robot
import typing
from scipy.spatial import KDTree


def M3(robot: Robot, samples: np.array, G: Graph, q_start: np.array, q_goal: np.array) -> typing.Tuple[np.array, bool]:
    """ Find a path from q_start to q_goal using the PRM roadmap

    Parameters
    ----------
    robot : Robot
        our robot object
    samples : np.array
        num_samples x 4 numpy array of nodes/vertices in the roadmap
    G : Graph
        An undirected NetworkX graph object with the number of nodes equal to num_samples, 
        and weighted edges indicating collision free connections in the robot's configuration space
    q_start : np.array
        1x4 numpy array denoting the start configuration
    q_goal : np.array
       1x4 numpy array denoting the goal configuration

    Returns
    -------
    typing.Tuple[np.array, bool]
        np.array:
            Nx4 numpy array containing a collision-free path between
            q_start and q_goal, if a path is found. The first row
            should be q_start, the final row should be q_goal.
        bool:
            Boolean denoting whether a path was found
    """


    s = KDTree(samples)
    start = s.query(q_start, k=1)[1]
    goal = s.query(q_goal, k=1)[1]
    if not (robot.check_edge(q_start, samples[start]) and robot.check_edge(samples[goal], q_goal)):
        return np.array([]), False

    path = networkx.has_path(G, source=start, target=goal)
    if path:
        index = shortest_path(G, source=start, target=goal)
        new_path = [q_start] + [samples[i] for i in index] + [q_goal]
        return np.array(new_path), True
    else:
        return np.array([]), False