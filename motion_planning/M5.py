import numpy as np
from robot import Simple_Manipulator as Robot

def M5(robot: Robot, path: np.array) -> np.array:
    """Smooth the given path

    Parameters
    ----------
    robot : Robot
        our robot object
    path : np.array
        Nx4 numpy array containing a collision-free path between q_start and q_goal

    Returns
    -------
    np.array
        Nx4 numpy array containing a smoothed version of the
        input path, where some unnecessary intermediate
        waypoints may have been removed
    """

    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if robot.check_edge(path[i], path[j]):
                path = np.concatenate((path[:i+1], path[j:]), axis=0)
                break 
            j -= 1
        i += 1
    return path