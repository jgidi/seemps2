import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from libcpp cimport bool
import enum
# something.pxd
cdef extern from "limits.h":
    cdef int INT_MAX

MAX_BOND_DIMENSION = INT_MAX

class Truncation:
    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2
    ABSOLUTE_SINGULAR_VALUE = 3

cdef class Strategy:
    cdef int method
    cdef double tolerance
    cdef int max_bond_dimension
    cdef int max_sweeps
    cdef bool normalize
    cdef bool simplify

    def __init__(self,
                 method: int = Truncation.RELATIVE_SINGULAR_VALUE,
                 tolerance: float = 1e-8,
                 max_bond_dimension: Optional[int] = INT_MAX,
                 normalize: bool = False,
                 simplify: bool = False,
                 max_sweeps: int = 16):
        if method < 0 or method > 3:
            raise AssertionError("Invalid method argument passed to Strategy")
        self.method = method
        if self.tolerance < 0 or self.tolerance >= 1.0:
            raise AssertionError("Invalid tolerance argument passed to Strategy")
        self.tolerance = tolerance
        if max_bond_dimension is None:
            self.max_bond_dimension = INT_MAX
        elif max_bond_dimension <= 0:
            raise AssertionError("Invalid bond dimension in Strategy")
        else:
            self.max_bond_dimension = max_bond_dimension
        self.normalize = normalize
        self.simplify = simplify
        if max_sweeps < 0:
            raise AssertionError("Negative or zero number of sweeps in Strategy")
        self.max_sweeps = max_sweeps

    def replace(self,
                 method: Optional[Truncation] = None,
                 tolerance: Optional[float] = None,
                 max_bond_dimension: Optional[int] = None,
                 normalize: Optional[bool] = None,
                 simplify: Optional[bool] = None,
                 max_sweeps: Optional[int] = None):
        return Strategy(method = self.method if method is None else method,
                        tolerance = self.tolerance if tolerance is None else tolerance,
                        max_bond_dimension = self.max_bond_dimension if max_bond_dimension is None else max_bond_dimension,
                        normalize = self.normalize if normalize is None else normalize,
                        simplify = self.simplify if simplify is None else simplify,
                        max_sweeps = self.max_sweeps if max_sweeps is None else max_sweeps)

    def get_method(self) -> int:
        return self.method

    def get_tolerance(self) -> float:
        return self.tolerance

    def get_max_bond_dimension(self) -> int:
        return self.max_bond_dimension

    def get_normalize_flag(self) -> bool:
        return self.normalize

    def get_max_sweeps(self) -> int:
        return self.max_sweeps

    def get_simplify_flag(self) -> bool:
        return self.simplify

    def __str__(self) -> str:
        if self.method == 0:
            method="None"
        elif self.method == 1:
            method="RelativeSVD"
        elif self.method == 2:
            method="RelativeNorm"
        else:
            method="AbsoluteSVD"
        return f"Strategy(method={method}, tolerance={self.tolerance}, " \
               f"max_bond_dimension={self.max_bond_dimension}, normalize={self.normalize}, " \
               f"simplify={self.simplify}, max_sweeps={self.max_sweeps})"

DEFAULT_TOLERANCE = np.finfo(np.float64).eps

DEFAULT_STRATEGY = Strategy(method = Truncation.RELATIVE_NORM_SQUARED_ERROR,
                            tolerance = np.finfo(np.float64).eps,
                            max_bond_dimension = INT_MAX,
                            normalize = False)

NO_TRUNCATION = DEFAULT_STRATEGY.replace(method = Truncation.DO_NOT_TRUNCATE)

cdef cnp.float64_t[::1] errors_buffer = np.zeros(1024, dtype=np.float64)

cdef cnp.float64_t *get_errors_buffer(Py_ssize_t N) noexcept:
    global errors_buffer
    if errors_buffer.size <= N:
        errors_buffer = np.zeros(2 * N, dtype=np.float64)
    return &errors_buffer[0]

def truncate_vector(cnp.ndarray[cnp.float64_t, ndim=1] s,
                    Strategy strategy):
    global errors_buffer

    cdef:
        Py_ssize_t i, final_size, N = s.size
        double total, max_error, new_norm
        cnp.float64_t *errors = get_errors_buffer(N)
        cnp.float64_t *data

    if strategy.method == 0:
        max_error = 0.0
    elif strategy.method == 2:
        #
        # Compute the cumulative sum of the reduced density matrix eigen values
        # in reversed order. Thus errors[i] is the error we make when we drop
        # i singular values.
        #
        total = 0.0
        data = &s[N-1]
        for i in range(N):
            errors[i] = total
            total += data[0]*data[0]
            data -= 1
        errors[N] = total

        max_error = total * strategy.tolerance
        final_error = 0.0
        for i in range(N):
            if errors[i] >= max_error:
                i -= 1
                break
        final_size = min(N - i, strategy.max_bond_dimension)
        max_error = errors[i]
        if final_size < N:
            s = s[:final_size]
        if strategy.normalize:
            new_norm = sqrt(total - max_error)
            data = &s[0]
            for i in range(final_size):
                data[i] /= new_norm
    else:
        data = &s[0]
        if strategy.method == 1:
            max_error = strategy.tolerance * data[0]
        else:
            max_error = strategy.tolerance
        final_size = N
        for i in range(N):
            if data[i] <= max_error:
                final_size = i
                break
        if final_size <= 0:
            final_size = 1
        elif final_size > strategy.max_bond_dimension:
            final_size = strategy.max_bond_dimension
        max_error = 0.0
        for i in range(final_size, N):
            max_error += data[i] * data[i]
        s = s[:final_size]
    return s, max_error