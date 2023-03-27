from typing import Union
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph, InteractiveGraph
from netgraph._artists import EdgeArtist
import netgraph
import random
import torch

def visualize_batch(batch, file_name_root: str, n: Union[int, None] = None, overlap_rewiring=True):
    counts = {i: 0 for i in np.unique(batch.batch)}
    for i, v in enumerate(batch.batch):
        counts[v.item()] += 1

    num_graphs = min(len(counts), n) if n is not None else len(counts)
    total_node_count = sum([counts[i] for i in range(min(len(counts), n))])

    total_edge_count = 0
    for j, k in enumerate([batch.edge_index[:, i] for i in range(batch.edge_index.shape[1])]):
        if k[0] >= total_node_count:
            total_edge_count = j
            break

    total_expander_edge_count = 0
    for j, k in enumerate([batch.expander_edge_index[:, i] for i in range(batch.expander_edge_index.shape[1])]):
        if k[0] >= total_node_count:
            total_expander_edge_count = j
            break

    colors = [
        f"#{''.join([random.choice('0123456789ABCDEF') for j in range(6)])}" for i in range(num_graphs)]

    if not overlap_rewiring:
        G = nx.Graph()
        H = nx.Graph()
        for i in range(total_node_count):
            G.add_node(i)
            H.add_node(i)

        for i in range(total_edge_count):
            k = batch.edge_index[:, i]
            a, b = k[0].item(), k[1].item()
            G.add_edge(a, b)

        for i in range(total_expander_edge_count):
            k = batch.expander_edge_index[:, i]
            a, b = k[0].item(), k[1].item()
            H.add_edge(a, b)

        fig, axs = plt.subplots(ncols=2, figsize=(14, 8))

        g = Graph(G, ax=axs[0])
        h = Graph(H, ax=axs[1], node_layout=g.node_positions)
        for j, i in enumerate(batch.batch[:total_node_count]):
            g.node_artists[j].set_facecolor(colors[i.item()])
            h.node_artists[j].set_facecolor(colors[i.item()])
        fig.canvas.draw()
    else:
        G = nx.Graph()
        edges = []
        for i in range(total_edge_count):
            k = batch.edge_index[:, i]
            a, b = k[0].item(), k[1].item()
            if (b, a, True) not in edges:
                edges.append((a, b, True))
        for i in range(total_expander_edge_count):
            k = batch.expander_edge_index[:, i]
            a, b = k[0].item(), k[1].item()
            if (b, a, False) not in edges:
                edges.append((a, b, False))

        fig = plt.figure()
        fig.set_size_inches(15, 15)
        node_color = {j: colors[i.item()] for j, i in enumerate(
            batch.batch[:total_node_count])}
        edge_width = {(u, v): 0.5 if is_orig else 0.1 for (
            u, v, is_orig) in edges}
        edge_color = {(u, v): 'blue' if is_orig else 'red' for (
            u, v, is_orig) in edges}
        Graph([(a, b) for (a, b, is_orig) in edges], edge_width=edge_width,
              edge_color=edge_color, node_color=node_color)

    plt.savefig(f"{file_name_root}.png")


def add_master_node(datapoint):
    for store in datapoint.stores:
        if 'edge_index' in store:
            num_nodes = store['num_nodes']
            idxs = torch.arange(0, num_nodes)[None, :]
            additional_edges = (torch.cat((torch.ones_like(idxs)*num_nodes, idxs), dim=0).to(dtype=store['edge_index'].dtype),
                                torch.cat((idxs, torch.ones_like(idxs)*num_nodes), dim=0).to(dtype=store['edge_index'].dtype))

            store['x'] = torch.cat(
                (store['x'], torch.zeros((1, store['x'].shape[1])).to(dtype=store['x'].dtype)), dim=0)
            store['edge_index'] = torch.cat(
                (store['edge_index'], *additional_edges), dim=1)
            store['edge_attr'] = torch.cat((store['edge_attr'], torch.zeros(
                (num_nodes * 2, store['edge_attr'].shape[1])).to(dtype=store['edge_attr'].dtype)), dim=0)
            store['num_nodes'] += 1
    return datapoint
