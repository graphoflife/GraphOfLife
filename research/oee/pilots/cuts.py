"""The largest piece one node's removal cuts off — the node version of cutRisk."""


def node_cut_risk(G):
    """Largest min(piece, rest) / n over every articulation point and the pieces it separates."""
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    disc, low, size = {}, {}, {}
    worst = 0
    timer = 0
    for root in G.nodes():
        if root in disc:
            continue
        disc[root] = low[root] = timer
        timer += 1
        size[root] = 1
        stack = [(root, None, iter(G[root]))]
        while stack:
            node, parent, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[node])
                    size[u] += size[node]
                    if low[node] >= disc[u]:
                        piece = size[node]
                        worst = max(worst, min(piece, n - 1 - piece))
                continue
            if nxt == parent:
                continue
            if nxt in disc:
                low[node] = min(low[node], disc[nxt])
                continue
            disc[nxt] = low[nxt] = timer
            timer += 1
            size[nxt] = 1
            stack.append((nxt, node, iter(G[nxt])))
    return worst / n
