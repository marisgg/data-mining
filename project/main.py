#!/usr/bin/env python3
import sklearn.decomposition
import numpy as np

popsize = 100
dimension = np.array([100, 500, 1000])
run_times = 25
maxFEs = 5000 * dimension

seed = 0

np.random.seed(seed)

def custom_pca(matrix, numOfComponents=2):
    if len(matrix.shape) >= 3:
        return None
    # calculate the mean of each column
    M = np.mean(matrix.T, axis=1)
    # center columns by subtracting column means
    C = matrix - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    _, vectors = np.linalg.eig(V)
    # project data
    P = vectors.T.dot(C.T)
    return P.T

def industry_pca(matrix, numOfComponents=2):
    if len(matrix.shape) == 3:
        samples, nx, ny = matrix.shape
        matrix = matrix.reshape(samples, nx*ny)
        numOfComponents = 3
    pca = sklearn.decomposition.PCA(numOfComponents)
    pca = pca.fit(matrix)
    return pca.transform(matrix)

def init():
    pass

def update_velocity():
    pass

def update_position():
    pass

def position_pca():
    pass

def algorithm():
    init() # 
    for t in range(run_times):
        for i in range(popsize):
            r = np.random.uniform(0, 1)
            bats[i] = self.Qmin + (self.Qmax - self.Qmin) * rnd
        update_velocity() # For every bat
        update_position() # For every bat
        position_pca() # For every bat

        if 1 > 0:
            pass

def main():
    rng = np.random.default_rng(seed)
    algorithm()
    M = np.array([[1, 2], [3, 4], [5, 6]])
    print(tuple(dimension))
    M = rng.integers(1000, size=tuple(dimension))
    pca = industry_pca(M)
    print(pca)
    pca = custom_pca(M)
    print(pca)

if __name__ == "__main__":
    main()