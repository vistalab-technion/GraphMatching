
import numpy as np
Nfeval = 1
evnr = 1


# well = 0#np.abs(L_full-L0_full)


def create_orig_support(full_size, size_part, well):
    # original support. High values on diagonal of rest, negative values on part diagonal and +1 for edge elements
    v = np.zeros((full_size, full_size))
    for i in range(0, full_size):
        for j in range(0, full_size):
            if (i == j and i >= size_part):
                v[i, j] = 1.0
            if (i != j and well[i, j] >= 0.01):
                v[i, j] = well[i, j]
                v[i, i] = -well[i, i]
                v[j, i] = well[j, i]
    return v


def create_diag_support_orig(full_size, size_part, well):
    # diagonal for rest has large values, negative values on part of diagonal that correspond to edges
    v = np.zeros((full_size, full_size))
    for i in range(0, full_size):
        if (i >= size_part):
            v[i, i] = 10.0
        else:
            v[i, i] = -well[i, i]

    return v


def create_diag_support_full(full_size, size_part, well):
    # diagonal has some values
    v = np.zeros((full_size, full_size))

    for i in range(0, full_size):
        v[i, i] = 1.0

    return v


def create_diag_support_rest(full_size, size_part, well):
    # diagonal for rest has large values
    v = np.zeros((full_size, full_size))
    for i in range(0, full_size):
        if (i >= size_part):
            v[i, i] = 10.0

    return v


def bounds_orig(mask, full_size,maski):
    bnds = []
    for i in range(0, full_size * full_size):
        if (maski[i] == 0):
            bnds.append((0, 0))
        if (maski[i] > 0):
            bnds.append((0, 10))
        if (maski[i] < 0):
            bnds.append((-10, 0))

    return tuple(bnds)


def bounds_anything(mask, full_size,maski):
    bnds = []
    for i in range(0, full_size * full_size):
        if (maski[i] == 0):
            bnds.append((0, 0))
        else:
            bnds.append((-10, 10))

    return tuple(bnds)


def bounds_positive(mask, full_size,maski):
    bnds = []
    for i in range(0, full_size * full_size):
        if (maski[i] == 0):
            bnds.append((0, 0))
        else:
            bnds.append((0, 100000))

    return tuple(bnds)