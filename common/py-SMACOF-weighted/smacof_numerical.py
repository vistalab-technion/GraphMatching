import numpy as np

def smacof_embed(numvertices, target_dimension, weights, lengths, Vplus, maximum_iterations):
    # input
    # dimension
    # iterations cap
    # target dimension
    # the edge lengths
    # the edge weights
    # storage during calculation and output
    # V_+ is the moore-penrose inverse of the V matrix (the 'hessian') that holds the transformed edges, weights
    # calculated (once) using numpy for convenience
    # previous step
    # B = B(Z)
    B = np.ones((numvertices, numvertices))
    # output
    # array of resulting coordinates
    np.random.seed(0)
    X = np.random.random((numvertices, target_dimension))
    print(f"initial stress = {FindStress(numvertices, target_dimension, weights, lengths, X)}")

    for step in range(1, maximum_iterations+1):
        Z = X.copy()
        B = BTransform(numvertices, target_dimension, weights, lengths, Z)
        X = np.matmul(Vplus, np.matmul(B, Z))
        stress = FindStress(numvertices, target_dimension, weights, lengths, X)
        print("step:", step, "stress:", stress)
    return X

def FindStress(numvertices, target_dimension, weights, lengths, Z):
    # input
    # intermediate
    # output
    stress = 0.0
    # i != j
    for j in range(1, numvertices):
        for i in range(1, j):
            distance = 0.0
            for k in range(1, target_dimension):
                distance += (Z[i, k] - Z[j, k])**2
            distance = np.sqrt(distance)
            stress += (distance - lengths[i, j])**2 * weights[i, j]
    return stress

def BTransform(numvertices, target_dimension, weights, lengths, Z):
    # input
    # intermediate
    # output
    B = np.zeros((numvertices, numvertices))
    # i != j
    for j in range(1, numvertices):
        for i in range(1, j):
            distance = 0.0
            for k in range(1, target_dimension):
                distance += (Z[i, k] - Z[j, k])**2
            distance = np.sqrt(distance)
            if distance > 0:
                B[i, j] = - weights[i, j] * lengths[i, j] / distance
                B[j, i] = - weights[j, i] * lengths[j, i] / distance
            else:
                B[i, j] = 0
                B[j, i] = 0
    # i = j
    for i in range(1, numvertices):
        for j in range(1, i):
            B[i, i] -= B[i, j]
        for j in range(i+1, numvertices):
            B[i, i] -= B[i, j]
    return B


