import re

import torch


def remove_orange_nodes_from_dot(dot):
    """Torchviz generates some orange nodes that are annoying.

    ```
    dot = make_dot(dist, {"verts": verts}, show_attrs=False, show_saved=False)
    dot = remove_orange_nodes_from_dot(dot)
    dot.view()
    ```

    Args:
        dot (Digraph): graph visualization from torchviz.make_dot

    Returns:
        Digraph: visualization without annoying orange nodes.
    """
    from graphviz import Digraph

    node_pattern = re.compile(r"^\s*([\w\d_]+)\s*(\[(.+)\])?;?$")
    edge_pattern = re.compile(r"^\s*([\w\d_]+)\s*->\s*([\w\d_]+)\s*(\[(.+)\])?;?$")

    nodes = {}
    edges = []

    # Parse dot.body safely
    for line in dot.body:
        line = line.strip()
        if not line or line.startswith("//"):
            continue

        # Edge line
        m = edge_pattern.match(line)
        if m:
            src, dst, _, attr_str = m.groups()
            attrs = {}
            if attr_str:
                for pair in re.findall(
                    r'(\w+)\s*=\s*"(.*?)"|(\w+)\s*=\s*([^",\s]+)', attr_str
                ):
                    if pair[0]:
                        k, v = pair[0], pair[1]
                    else:
                        k, v = pair[2], pair[3]
                    attrs[k] = v
            edges.append((src, dst, attrs))
            continue

        # Node line
        m = node_pattern.match(line)
        if m:
            name, _, attr_str = m.groups()
            attrs = {}
            if attr_str:
                for pair in re.findall(
                    r'(\w+)\s*=\s*"(.*?)"|(\w+)\s*=\s*([^",\s]+)', attr_str
                ):
                    if pair[0]:
                        k, v = pair[0], pair[1]
                    else:
                        k, v = pair[2], pair[3]
                    attrs[k] = v
            nodes[name] = attrs

    # Filter out orange nodes
    kept_nodes = {
        n
        for n, a in nodes.items()
        if a.get("color") != "orange" and a.get("fillcolor") != "orange"
    }

    # Build new Digraph
    new_dot = Digraph(comment=dot.comment)
    for n in kept_nodes:
        new_dot.node(n, **nodes[n], shape="box")

    for src, dst, attrs in edges:
        if src in kept_nodes and dst in kept_nodes:
            new_dot.edge(src, dst, **attrs)

    return new_dot


def print_identity(x: torch.Tensor, name: str) -> torch.Tensor:
    """Passes forward a tensor and prints stuff in fwd and bwd passes.

    In the forward pass, it prints the tensor's name.
    In the backward pass it prints whether the tensor is receiving gradients.
    Does so in a way that the tensor name shows up in torchviz.

    Args:
        x (torch.Tensor): Tensor to forward.
        name (str): Tensor's name.

    Returns:
        torch.Tensor: Same tensor as input.
    """

    class _PrintFn(torch.autograd.Function):
        @staticmethod
        def setup_context(ctx, inputs, outputs):
            x, name = inputs
            ctx.name = name

        @staticmethod
        def forward(x, name: str):
            print(f"forward passthrough: {name}")
            return x

        @staticmethod
        def backward(ctx, grad_output):
            print(f"backward passthrough: {ctx.name}: {grad_output}")
            return grad_output, None

    cls = type(name + "_pt_", (_PrintFn,), {})
    if x.is_sparse:
        return cls.apply(x.to_dense(), name).to_sparse_coo()
    else:
        return cls.apply(x, name)
