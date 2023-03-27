from primefac import primefac
import torch_geometric.utils as geom_utils
import torch
import numpy as np
from copy import deepcopy as dc
from enum import Enum


class ExpanderConfig(Enum):
    NONE = 0
    DEFAULT = 1
    WITH_LEARNABLE_EDGE_FEATS = 2


def cayley_size(n):
    """
        Calculate the number of elements in the Cayley graph (Cay(SL(2, Z_n); S_n)).
    """
    n = int(n)
    return round(n*n*n*np.prod([1 - 1.0/(p * p) for p in list(set(primefac(n)))]))


def get_suitable_cayley_n(num_nodes):
    """
        Get a suitable natural number `n` such that the size of the Cayley graph (Cay(SL(2, Z_n); S_n)) is at least as big as the target `num_nodes`.
    """
    n = 1
    while cayley_size(n) < num_nodes:
        n += 1
    return n


def index_of_tensor(tensor_list, tensor):
    for i, item in enumerate(tensor_list):
        if torch.equal(item, tensor):
            return i
    raise Exception("Tensor not found in list.")


def tensor_in_list(tensor_list, tensor):
    return any([torch.equal(tensor, n) for n in tensor_list])


def get_cayley_graph(cayley_n, generators, track_layers=False):
    node_zero = torch.eye(2).long()
    nodes = [node_zero]
    added_nodes = [node_zero]
    edges = []
    layers = [[node_zero]]
    while len(added_nodes) != 0:
        new_added_nodes = []
        for node in added_nodes:
            resulting_nodes = [(torch.remainder(
                node @ gen, cayley_n), dir) for gen, dir in generators]
            for resulting_node, forward in resulting_nodes:
                if not tensor_in_list(nodes, resulting_node):
                    nodes.append(resulting_node)
                    new_added_nodes.append(resulting_node)

                if forward:
                    edges.append((index_of_tensor(nodes, node),
                                  index_of_tensor(nodes, resulting_node)))
                else:
                    edges.append((index_of_tensor(nodes, resulting_node),
                                  index_of_tensor(nodes, node)))
        added_nodes = new_added_nodes
        if track_layers and len(added_nodes) != 0:
            layers.append(dc(added_nodes))

    if track_layers:
        return nodes, edges, layers
    else:
        return nodes, edges


def get_cayley_graph_breadth_first(cayley_n):
    # Forward generators
    s1 = torch.tensor([[1, 1], [0, 1]]).long()
    s2 = torch.tensor([[1, 0], [1, 1]]).long()
    # Backward generators
    s1_inv = torch.remainder(torch.inverse(s1.float()), cayley_n).long()
    s2_inv = torch.remainder(torch.inverse(s2.float()), cayley_n).long()

    generators = [(s1, True), (s2, True), (s1_inv, False), (s2_inv, False)]
    return get_cayley_graph(cayley_n, generators)


def get_undirected_cayley_graph_ordered_adj(cayley_n):
    """
        Get the adjacency matrix associated with the Cayley graph (Cay(SL(2, Z_n); S_n)) in a breadth-first manner
    """
    cayley_nodes, edges = get_cayley_graph_breadth_first(cayley_n)
    adj = torch.zeros((len(cayley_nodes), len(cayley_nodes))).long()
    for a, b in edges:
        adj[a, b] = 1
        adj[b, a] = 1
    return adj


def get_edge_index_from_adj(adj):
    return geom_utils.dense_to_sparse(adj)[0]


def _EXPERIMENTAL_shuffle(data):
    def get_overlap(edge_index, expander_edge_index):
        exp = expander_edge_index.T.tolist()
        normal = edge_index
        c = 0.0
        for edge in exp:
            if edge in normal:
                c += 1
        return c/len(exp)

    expander_edge_index = data['expander_edge_index']
    edge_index = data['edge_index'].T.tolist()
    cur = get_overlap(edge_index, expander_edge_index)
    for i in range(30):
        perm = np.random.permutation(data['num_nodes'])
        new_ee = expander_edge_index.clone()
        for i in range(new_ee.shape[1]):
            new_ee[0][i] = perm[new_ee[0][i]]
            new_ee[1][i] = perm[new_ee[1][i]]
        new_overlap = get_overlap(edge_index, new_ee)
        if new_overlap < cur:
            cur = new_overlap
            expander_edge_index = new_ee

    data['expander_edge_index'] = expander_edge_index


class ExpanderGraphStore:
    def __init__(self):
        self.cayley_graph_adj_store = {}
        self.expander_edge_index_store = {}

    def _add_expander_to_datapoint(self, datapoint):
        for store in datapoint.stores:
            if 'edge_index' in store:
                assert 'expander_edge_index' not in store
                num_nodes = store['num_nodes']
                expander_edge_index_entry = self.expander_edge_index_store.get(
                    num_nodes)
                if expander_edge_index_entry is not None:
                    expander_edge_index = expander_edge_index_entry.clone()
                else:
                    cayley_n = get_suitable_cayley_n(num_nodes)
                    cayley_graph_adj_entry = self.cayley_graph_adj_store.get(
                        cayley_n)
                    if cayley_graph_adj_entry is None:
                        cayley_graph_adj_entry = get_undirected_cayley_graph_ordered_adj(
                            cayley_n)
                        self.cayley_graph_adj_store[cayley_n] = cayley_graph_adj_entry

                    expander_edge_index = get_edge_index_from_adj(
                        cayley_graph_adj_entry[:num_nodes, :num_nodes])
                    # Shuffle indices to remove potential bias from original dataset node ordering
                    perm = np.random.permutation(num_nodes)
                    for i in range(expander_edge_index.shape[1]):
                        expander_edge_index[0][i] = perm[expander_edge_index[0][i]]
                        expander_edge_index[1][i] = perm[expander_edge_index[1][i]]
                    self.expander_edge_index_store[num_nodes] = expander_edge_index

                store['expander_edge_index'] = expander_edge_index
                # _EXPERIMENTAL_shuffle(store)
        return datapoint

    def transform_fn(self):
        return lambda datapoint: self._add_expander_to_datapoint(datapoint)


if __name__ == '__main__':
    def test_edge_index(adj, target_edge_index):
        assert get_edge_index_from_adj(adj).equal(target_edge_index)

    test_edge_index(torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]).to(
        dtype=int), torch.tensor([[0, 1, 2, 3], [2, 3, 0, 1]]).to(dtype=int))
