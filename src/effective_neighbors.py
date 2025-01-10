""" 
This code is based on the paper: https://arxiv.org/abs/2206.03093

The code is mostly borrowed and adapted from the original paper's research code base: https://github.com/epfml/topology-in-decentralized-learning
"""

from math import sqrt
import scipy.linalg
import networkx as nx
import math
from functools import lru_cache
import networkx
import torch


class Topology:
    num_workers: int

    def __init__(self, num_workers):
        self.num_workers = num_workers

    def neighbors(self, worker: int) -> list[int]:
        raise NotImplementedError()

    def degree(self, worker: int) -> int:
        return len(self.neighbors(worker))

    @property
    def workers(self) -> list[int]:
        return list(range(self.num_workers))

    @property
    def max_degree(self) -> int:
        return max(self.degree(w) for w in self.workers)

    def gossip_matrix(self, weight=None) -> torch.Tensor:
        m = torch.zeros([self.num_workers, self.num_workers])
        for worker in self.workers:
            for neighbor in self.neighbors(worker):
                max_degree = max(self.degree(worker), self.degree(neighbor))
                m[worker, neighbor] = 1 / (max_degree + 1) if weight is None else weight
            # self weight
            m[worker, worker] = 1 - m[worker, :].sum()

        return m

    def to_networkx(self) -> networkx.Graph:
        g = networkx.Graph()
        g.add_nodes_from(range(self.num_workers))
        for worker in range(self.num_workers):
            g.add_edges_from(
                [(worker, neighbor) for neighbor in self.neighbors(worker)]
            )
        return g

    @property
    def max_delay(self):
        g = self.to_networkx()
        distances = dict(networkx.all_pairs_shortest_path_length(g))
        return max(distances[i][j] for i in g.nodes for j in g.nodes)


class AveragingScheme:
    # Some defaults
    period = 1
    n = 1

    def init(self):
        return None

    @property
    def state_size(self):
        return self.n

    def w(self, t=0, params=None):
        return torch.eye(1)

    def show_weights(self, params=None, **kwargs):
        fig, axes = plt.subplots(ncols=self.period)
        if self.period == 1:
            axes = [axes]
        for t in range(self.period):
            axes[t].set_title(f"t={t}")
            axes[t].matshow(self.w(params=params, t=t), **kwargs)

        for ax in axes:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

        return fig


class FullyConnectedTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        return [j for j in range(n) if j != i]


class TwoCliquesTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        if i < n // 2:
            neighbors = [j for j in range(0, n // 2) if j != i]
        else:
            neighbors = [j for j in range(n // 2, n) if j != i]

        if i == 0:
            neighbors.append(n // 2)
        elif i == n // 2:
            neighbors.append(0)

        return neighbors


class DisconnectedTopology(Topology):
    def neighbors(self, worker):
        return []


class MixTopology(FullyConnectedTopology):
    r"""
    Symmetric doubly-stochastic gossip matrix with all \lambda_{2...} equal.
    """

    def __init__(self, num_workers, spectral_gap):
        super().__init__(num_workers)
        self.spectral_gap = spectral_gap

        ii = torch.eye(num_workers)
        ee = torch.ones_like(ii) / num_workers
        self.W = spectral_gap * ee + (1 - spectral_gap) * ii

    def gossip_matrix(self):
        return self.W


class StarTopology(Topology):
    def neighbors(self, worker):
        i = worker
        if i == 0:
            n = self.num_workers
            return [j for j in range(n) if j != i]
        else:
            return [0]


class ChainTopology(Topology):
    def neighbors(self, worker):
        if worker < 1:
            return [1]
        elif worker >= self.num_workers - 1:
            return [worker - 1]
        else:
            return [worker - 1, worker + 1]


class RingTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        if n == 1:
            return []
        elif n == 2:
            return [(i + 1) % n]
        else:
            return [(i - 1) % n, (i + 1) % n]


class UnidirectionalRingTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers
        return [(i + 1) % n]


class HyperCubeTopology(Topology):
    def neighbors(self, worker):
        i = worker
        n = self.num_workers

        d = int(math.log2(n))
        assert 2**d == n

        return [i ^ (2**j) for j in range(0, d)]


class TorusTopology(Topology):
    def __init__(self, n, m):
        self.num_workers = n * m
        self.n = n
        self.m = m

    def neighbors(self, worker):
        # i = col + row * m
        i = worker
        col = i % self.m
        row = i // self.m

        idx = lambda row, col: (col + row * self.m) % self.num_workers

        return [
            idx(row - 1, col),
            idx(row + 1, col),
            idx(row, col - 1),
            idx(row, col + 1),
        ]


class TreeTopology(Topology):
    """A tree that divides nodes such that nodes have the same degree if they are not (close to) leaves"""

    num_workers: int
    max_degree: int

    def __init__(self, num_workers, max_degree):
        super().__init__(num_workers=num_workers)
        self._max_degree = max_degree

    def max_workers_up_to_depth(self, layer_number: int) -> int:
        d = self._max_degree
        n = layer_number
        return int(1 + d * ((d - 1) ** n - 1) / (d - 2))

    def depth_of_worker(self, worker_number: int) -> int:
        # TODO: optimize / give direct formula
        depth = 0
        while True:
            if self.max_workers_up_to_depth(depth) > worker_number:
                return depth
            depth += 1

    def parent(self, worker_number: int) -> int:
        depth = self.depth_of_worker(worker_number)
        if depth == 0:
            return None
        index_within_layer = worker_number - self.max_workers_up_to_depth(depth - 1)
        if depth == 1:
            parent_within_layer = index_within_layer // (self._max_degree)
        else:
            parent_within_layer = index_within_layer // (self._max_degree - 1)
        return parent_within_layer + self.max_workers_up_to_depth(depth - 2)

    def children(self, worker_number: int) -> list[int]:
        if worker_number == 0:
            children = [1 + x for x in range(self._max_degree)]
        else:
            depth = self.depth_of_worker(worker_number)
            start_idx_my_depth = self.max_workers_up_to_depth(depth - 1)
            start_idx_next_depth = self.max_workers_up_to_depth(depth)
            i = worker_number - start_idx_my_depth
            d = self._max_degree
            children = [start_idx_next_depth + (d - 1) * i + x for x in range(d - 1)]
        return [c for c in children if c < self.num_workers]

    def neighbors(self, worker: int) -> list[int]:
        if worker == 0:
            return self.children(worker)
        else:
            return [self.parent(worker)] + self.children(worker)


class NetworkxTopology(Topology):
    def __init__(self, nx_graph):
        super().__init__(num_workers=len(nx_graph.nodes))
        self.graph = networkx.relabel.convert_node_labels_to_integers(nx_graph)

    def neighbors(self, worker: int) -> list[int]:
        return list(self.graph.neighbors(worker))


class SocialNetworkTopology(NetworkxTopology):
    def __init__(self):
        nx_graph = networkx.davis_southern_women_graph()
        super().__init__(nx_graph)


class BinaryTreeTopology(Topology):
    def __init__(self, num_workers, reverse=False):
        super().__init__(num_workers=num_workers)
        self.reverse = reverse

    def neighbors(self, worker):
        if self.num_workers < 2:
            return []
        elif worker >= self.num_workers or worker < 0:
            raise ValueError(
                f"worker number {worker} is out of range [0, {self.num_workers})"
            )
        elif worker == 0 and not self.reverse:
            return [1]
        elif worker == self.num_workers - 1 and self.reverse:
            return [self.num_workers - 2]
        elif not self.reverse:
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [c for c in children if c < self.num_workers]
            return [parent, *children]
        elif self.reverse:
            worker = self.num_workers - 1 - worker
            parent = worker // 2
            children = [worker * 2, worker * 2 + 1]
            children = [
                self.num_workers - 1 - c for c in children if c < self.num_workers
            ]
            parent = self.num_workers - 1 - parent
            return [parent, *children]


class Matrix(AveragingScheme):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.matrix = matrix
        self.n = len(matrix)

    def w(self, t=0, params=None):
        return self.matrix


class TimeVaryingExponential(AveragingScheme):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.d = int(math.log(n, 2))
        self.period = self.d
        assert 2**self.d == self.n

    def w(self, t=0, params=None):
        offset = 2 ** (t % self.d)
        return self._w(offset)

    @lru_cache(maxsize=10)
    def _w(self, offset):
        w = torch.eye(self.n)
        w = (w + torch.roll(w, -offset, 0)) / 2
        return w


class LocalSteps(AveragingScheme):
    def __init__(self, n, period):
        super().__init__()
        self.n = n
        self.period = period
        self.avg = torch.ones([n, n]) / n
        self.no_avg = torch.eye(n)

    def w(self, t=0, params=None):
        if (t + 1) % self.period == 0:
            return self.avg
        else:
            return self.no_avg


def scheme_for_string(topology: str, num_workers: int) -> AveragingScheme:
    if topology == "Ring":
        return Matrix(RingTopology(num_workers).gossip_matrix())
    if topology == "Uni-ring":
        return Matrix(UnidirectionalRingTopology(num_workers).gossip_matrix())
    if topology == "Torus (4x8)":
        return Matrix(TorusTopology(4, 8).gossip_matrix())
    if topology == "Torus (2x16)":
        return Matrix(TorusTopology(2, 16).gossip_matrix())
    if topology == "Torus (8x8)":
        return Matrix(TorusTopology(8, 8).gossip_matrix())
    if topology == "Binary tree":
        return Matrix(BinaryTreeTopology(num_workers).gossip_matrix())
    if topology == "Two cliques":
        return Matrix(TwoCliquesTopology(num_workers).gossip_matrix())
    if topology == "Hypercube":
        return Matrix(HyperCubeTopology(num_workers).gossip_matrix())
    if topology == "Star":
        return Matrix(StarTopology(num_workers).gossip_matrix())
    if topology == "Social network":
        assert num_workers == 32
        return Matrix(SocialNetworkTopology().gossip_matrix())
    if topology == "Fully connected":
        return Matrix(FullyConnectedTopology(num_workers).gossip_matrix())
    if topology == "Solo":
        return Matrix(DisconnectedTopology(num_workers).gossip_matrix())
    if topology == "Time-varying exponential":
        return TimeVaryingExponential(num_workers)
    if topology.startswith("Local steps"):
        for i in range(100):
            if topology == f"Local steps ({i})":
                return LocalSteps(num_workers, i)

    raise ValueError(f"Unknown topology {topology}")


class AveragingScheme:
    # Some defaults
    period = 1
    n = 1

    def init(self):
        return None

    @property
    def state_size(self):
        return self.n

    def w(self, t=0, params=None):
        return torch.eye(1)

    def show_weights(self, params=None, **kwargs):
        fig, axes = plt.subplots(ncols=self.period)
        if self.period == 1:
            axes = [axes]
        for t in range(self.period):
            axes[t].set_title(f"t={t}")
            axes[t].matshow(self.w(params=params, t=t), **kwargs)

        for ax in axes:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

        return fig


class Matrix(AveragingScheme):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.matrix = matrix
        self.n = len(matrix)

    def w(self, t=0, params=None):
        return self.matrix


def solve_discrete_lyapunov(
    A: torch.Tensor, Q: torch.Tensor, method=None
) -> torch.Tensor:
    """
    Solves the discrete Lyapunov equation `A X A^T - X + Q = 0`.
    This is a wrapper around scipy that accepts torch tensors.
    """
    device = A.device
    assert Q.device == A.device
    A = A.cpu().numpy()
    Q = Q.cpu().numpy()
    out = scipy.linalg.solve_discrete_lyapunov(A, Q, method)
    return torch.from_numpy(out).to(device)


def simulate_random_walk(
    scheme: AveragingScheme, gamma: float, num_steps: int, num_reps: int
) -> torch.Tensor:
    x = torch.zeros([scheme.n, num_reps])

    for t in range(num_steps):
        x = scheme.w(t) @ (sqrt(gamma) * x + torch.randn_like(x))

    return x


def effective_number_of_neighbors(
    scheme: AveragingScheme, gamma: float, t: int = 0, mode="mean", start_at: int = 1
):
    var_per_worker = random_walk_covariance(scheme, gamma, t, start_at=start_at).diag()
    if mode == "mean":
        return 1 / (1 - gamma) / var_per_worker.mean()
    elif mode == "worst":
        return 1 / (1 - gamma) / var_per_worker.max()
    elif mode == "all":
        return 1 / (1 - gamma) / var_per_worker
    else:
        raise ValueError("Unknown mode")


def random_walk_covariance_static(
    gossip_matrix: torch.Tensor, gamma: float, start_at: int = 1
) -> torch.Tensor:
    """
    Asymptotic covariance `E[x x^T]` in the random walk process
    `x <- W @ (sqrt(gamma) x + n)`,
    where `x` is a vector containing one scalar per worker
    and `n` is i.i.d. standard normal noise.
    """
    W = gossip_matrix
    w_is_symmetric = W.allclose(W.T)
    if w_is_symmetric:
        L, Q = torch.linalg.eigh(W)
        numerator = L.square() if start_at == 1 else 1
        diag = numerator / (1 - gamma * L.square())
        return (Q * diag) @ Q.T  # = Q @ torch.diag(diag) @ Q.T
    else:
        rhs = W @ W.T if start_at == 1 else torch.eye(len(W))
        return solve_discrete_lyapunov(sqrt(gamma) * W, rhs)


def random_walk_covariance(
    scheme: AveragingScheme, gamma: float, t: int = 0, start_at: int = 1
) -> torch.Tensor:
    """
    Asymptotic covariance `E[x x^T]` in the (periodically time-varying) random walk process
    `x <- W[t] @ (sqrt(gamma) x + n)`,
    where `x` is a vector containing one scalar per worker
    and `n` is i.i.d. standard normal noise.
    The covariance is evaluated after averaging with `W[n * period + t]`.
    """
    if scheme.period == 1:
        return random_walk_covariance_static(
            gossip_matrix=scheme.w(0), gamma=gamma, start_at=start_at
        )

    period = scheme.period
    n = scheme.n

    # We split all the terms by their length mod period,
    # and compute contributions from each of them
    cumulative_cov = torch.zeros([n, n])
    for len_mod_period in range(period):
        # Compute the transition matrix
        T = torch.eye(n)
        for s in range(
            t - period - len_mod_period - start_at, t - len_mod_period - start_at
        ):
            T = scheme.w(s) @ T

        cov = random_walk_covariance_static(
            gossip_matrix=T, gamma=gamma**period, start_at=0
        )
        for s in range(t - len_mod_period - start_at, t):
            cov = scheme.w(s) @ cov @ scheme.w(s).T
        cumulative_cov += cov * gamma ** (len_mod_period)

    return cumulative_cov


def get_n_placement_locations(
    graph,
    gamma,
    n,
):
    """
    This funcation will give you n nodes on by which to sort your information placement decision based on each  node's
    # of effective neighbors

    graph --> network x graph
    gamma --> parameter to vary how aggressive the random walk is for the # of effetive neighbors calculations
    n --> # of placement nodes in the graph you want

    """
    avg_effective_neighbors = torch.zeros(len(graph))
    nx_topo = NetworkxTopology(graph)
    matrix = Matrix(nx_topo.gossip_matrix())
    for i in range(len(graph)):

        avg_effective_neighbors += effective_number_of_neighbors(
            scheme=matrix, gamma=0.9, t=0, mode="all", start_at=i
        )

    avg_effective_neighbors /= len(graph)

    start = 0
    interval = len(graph) // n
    l = list(range(start, (interval + 1) * n, interval))

    val, ind = torch.sort(avg_effective_neighbors)
    placement_neighbors = torch.index_select(ind, 0, torch.tensor(l))

    return placement_neighbors.tolist()
