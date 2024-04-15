from M1 import M1
import numpy as np
from robot import Simple_Manipulator as Robot
import typing

def M4(robot: Robot, q_start: np.array, q_goal: np.array) -> typing.Tuple[np.array, bool]:
    """Implement RRT algorithm to find a path from q_start to q_goal

    Parameters
    ----------
    robot : Robot
        our robot object
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

    max_iterations = 1000
    step_size = 0.05
    a, b = [q_start], [-1]

    for i in range(max_iterations):
        randq = q_goal if np.random.rand() < 0.1 else np.random.uniform(robot.lower_lims, robot.upper_lims, size=q_start.shape)
        near = np.argmin([np.linalg.norm(q - randq) for q in a])
        nearq = a[near]

        direction = (randq - nearq) / np.linalg.norm(randq - nearq)
        newq = nearq + step_size * direction

        if robot.check_edge(nearq, newq):
            a.append(newq)
            b.append(near)

            if np.linalg.norm(newq - q_goal) < step_size:
                path = [q_goal]
                while near != -1:
                    path.append(a[near])
                    near = b[near]
                return np.array(path[::-1]), True

    return np.array([]), False
