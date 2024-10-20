from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Implemented for Task 1.1.
    # assume that inputs are well formed
    epsvals = list(vals)
    epsvals[arg] = vals[arg] + epsilon
    return (f(*epsvals) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """See Scalar.accumulate_derivative"""
        ...

    @property
    def unique_id(self) -> int:
        """See Scalar.unique_id"""
        ...

    def is_leaf(self) -> bool:
        """See Scalar.is_leaf"""
        ...

    def is_constant(self) -> bool:
        """See Scalar.is_constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """See Scalar.parents"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """See Scalar.chain_rule"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.
    The graph must not have cycles.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    out = []
    blockers = {}
    visited = set()

    def _visit(v: Variable) -> None:
        for p in v.parents:
            if p.unique_id not in blockers:
                blockers[p.unique_id] = set()
            blockers[p.unique_id].add(v.unique_id)
            if p.unique_id not in visited:
                visited.add(p.unique_id)
                _visit(p)

    # build a graph
    _visit(variable)

    # run the actual topological sort algorithm
    poppable = [variable]
    while len(poppable) > 0:
        item = poppable.pop()
        out.append(item)
        for p in item.parents:
            # item.unique_id may not be in blockers if previously removed,
            # e.g. if p was in item.parents multiple times.
            if item.unique_id in blockers[p.unique_id]:
                blockers[p.unique_id].remove(item.unique_id)
                if len(blockers[p.unique_id]) == 0:
                    poppable.append(p)

    return out


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable.
        deriv: Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None: Updates the derivative values of each leaf through accumulate_derivative`.

    """
    schedule = topological_sort(variable)
    # print(f"schedule {[ (var, var.unique_id) for var in schedule]}")
    dLdKey = {variable.unique_id: deriv}
    for var in schedule:
        # note that the first var should be variable
        if var.is_leaf():
            # print(f"dLdKey {dLdKey}")
            var.accumulate_derivative(dLdKey[var.unique_id])
        else:
            for parent, dLdParent in var.chain_rule(dLdKey[var.unique_id]):
                dLdKey[parent.unique_id] = dLdKey.get(parent.unique_id, 0.0) + dLdParent


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values"""
        return self.saved_values
