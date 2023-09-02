from .typing import *
from .state.environments import *
from .state.mps import MPS
from .state.environments import scprod


def expectation1(state: MPS, O: Operator, i: int) -> Weight:
    """Compute the expectation value :math:`\\langle\\psi|O_i|\\psi\\rangle`
    of an operator O acting on the `i`-th site

    Parameters
    ----------
    state : MPS
        Quantum state :math:`\\psi` used to compute the expectation value.
    O : Operator
        Local observable acting onto the `i`-th subsystem
    i : int
        Index of site, in the range `[0, state.size)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    return state.expectation1(O, i)


def expectation2(
    state: MPS, O: Operator, Q: Operator, i: int, j: Optional[int] = None
) -> Weight:
    """Compute the expectation value :math:`\\langle\\psi|O_i Q_j|\\psi\\rangle`
    of two operators `O` and `Q` acting on the `i`-th and `j`-th subsystems.

    Parameters
    ----------
    state : MPS
        Quantum state :math:`\\psi` used to compute the expectation value.
    O, Q : Operator
        Local observables
    i : int
    j : int, default=`i+1`
        Indices of sites, in the range `[0, state.size)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    return state.expectation2(O, Q, i, j)


def all_expectation1(state: MPS, O: Union[list[Operator], Operator]) -> Vector:
    """Vector of expectation values :math:`v_i = \\langle\\psi|O_i|\\psi\\rangle`
    of local operators acting on individual sites of the MPS.

    Parameters
    ----------
    state: MPS
        State :math:`\\psi` onto which the expectation values are computed.
    operator : Operator | list[Operator]
        If `operator` is an observable, it is applied on each possible site.
        If it is a list, the expectation value of `operator[i]` is computed
        on the i-th site.

    Returns
    -------
    Vector
        Numpy array of expectation values.
    """
    return state.all_expectation1(O)


def product_expectation(state: MPSLike, operator_list: list[Operator]) -> Weight:
    """Expectation value of a product of local operators
    :math:`\\langle\\psi|O_0 O_1 \cdots O_{N-1}|\\psi\\rangle`.

    Parameters
    ----------
    state : MPSLike
        State :math:`\\psi` onto which the expectation values are computed.
    operator_list : list[Operator]
        List of operators, with the same length `len(operator_list) == len(state)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    rho = begin_environment()
    for Ai, opi in zip(state, operator_list):
        rho = update_left_environment(Ai.conj(), Ai, rho, operator=opi)
    return end_environment(rho)
