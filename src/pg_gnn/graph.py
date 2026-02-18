from typing import List, Tuple

import torch


def build_chain_edges(station_order: List[str]) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
    edges = []
    edge_names = []
    for i in range(len(station_order) - 1):
        src = i
        dst = i + 1
        edges.append([src, dst])
        edge_names.append((station_order[src], station_order[dst]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, edge_names


def build_upstream_mask(station_order: List[str]) -> torch.Tensor:
    count = len(station_order)
    mask = torch.zeros((count, count), dtype=torch.bool)
    for dst in range(count):
        for src in range(count):
            if src <= dst:
                mask[dst, src] = True
    return mask
