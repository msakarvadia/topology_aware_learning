import numpy as np

"""
In this file we create a few sample network topologies for testing

"""

topology = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
np.savetxt("topology/topo_1.txt", topology, fmt="%d")

topology = np.array([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
np.savetxt("topology/topo_2.txt", topology, fmt="%d")

topology = np.array(
    [
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0],
    ]
)
np.savetxt("topology/topo_3.txt", topology, fmt="%d")

topology = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ]
)
np.savetxt("topology/topo_4.txt", topology, fmt="%d")
