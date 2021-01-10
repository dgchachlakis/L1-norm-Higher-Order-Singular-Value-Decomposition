import numpy as np
import algorithm as alg
tensor = np.random.randn(3, 3, 3)
tensor_ranks = (2, 2, 2)
# Example 1:
# ====L1-HOSVD with fixed-point underlying L1-PCA solver
core1, factors1 = alg.l1hosvd(tensor, tensor_ranks, solver = 'fixedpoint')

# Example 2:
# ====L1-HOSVD with bitflipping underlying L1-PCA solver
core2, factors2 = alg.l1hosvd(tensor, tensor_ranks, solver = 'bitflipping')

# Example 3:
# ====L1-HOSVD with exactpoly underlying L1-PCA solver
core3, factors3 = alg.l1hosvd(tensor, tensor_ranks, solver = 'exactpoly')

# Example 4:
# ====L1-HOSVD with exact underlying L1-PCA solver
core4, factors4 = alg.l1hosvd(tensor, tensor_ranks, solver = 'exact')