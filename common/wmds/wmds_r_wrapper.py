import os
from pathlib import Path
import numpy as np

# Defining the R script and loading the instance in Python
from matplotlib import pyplot as plt

os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.3.2'
# os.environ['R_HOME'] = 'C:\\Users\\kogan\\AppData\\Local\\R\\win-library\\4.3'
try:
    import rpy2.robjects as robjects
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    import rpy2.robjects.numpy2ri as rpyn
    r = robjects.r
    rpy2.robjects.numpy2ri.activate()
    
    from rpy2.robjects.packages import importr
    smacof = importr('smacof')
except ModuleNotFoundError:
    print(f"!!! WARNING: COULD NOT IMPORT rpy2, check R_HOME path exists: {os.environ['R_HOME']} !!!")

def load_r_source_code(directory):
    # iterate over files in
    # that directory
    files = Path(directory).glob('*.R')
    for file in files:
        # print(file)
        r['source'](str(file))

def apply_wmds(x, w, dim):
    wcmdscale_r = smacof.smacofSym

    nr, nc = x.shape
    Xr = ro.r.matrix(x, nrow=nr, ncol=nc)
    Wr = ro.r.matrix(w, nrow=nr, ncol=nc)
    # ro.r.assign("B", Br)

    # Invoking the R function and getting the result
    res_r = wcmdscale_r(Xr, dim, weightmat=Wr)
    res = dict(zip(res_r.names, list(res_r)))

    embeddings = res['conf']
    embeddings = np.array(embeddings)

    stress_val = res['stress']
    print(embeddings)
    print(stress_val)

    return embeddings

# load_r_source_code('C:\\Program Files\\R\\smacof\\R\\')
# load_r_source_code('C:\\Program Files\\R\\laeken\\R\\')
# load_r_source_code('C:\\Users\\kogan\\AppData\\Local\\R\\win-library\\4.3\\smacof')
# import pandas as pd
# from rpy2.robjects import pandas2ri
# r['source']('C:\\Program Files\\R\\vegan\\R\\wcmdscale.R')
# Loading the function we have defined in R.
# wcmdscale_r = robjects.globalenv['smacofSym']

if __name__ == "__main__":
    B = np.ones((4,4))
    for i in range(B.shape[0]):
        B[i][i] = 0

    # for i in range(B.shape[0]):
    #     if i == 2:
    #         continue
    #     B[i][2] = 2
    #     B[2][i] = 2

    weightmat = np.ones(B.shape)
    weightmat[0][1] = 0
    weightmat[0][2] = 0
    # weightmat[1][0] = 0

    embeddings = apply_wmds(B, weightmat, dim=2)

    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=['red', 'blue', 'yellow', 'purple'])#, 'black'])
    plt.grid()
    plt.title('Feature space')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.plot()
    plt.show()