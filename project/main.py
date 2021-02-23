#!/usr/bin/env python3
import sklearn.decomposition
import numpy as np

popsize = 100
dimension = np.array([100, 500, 1000])
run_times = 25
maxFEs = 5000 * dimension

seed = 0

np.random.seed(seed)

class PCABat():
    def __init__(self, function):
        self.popsize = popsize
        self.dimension = dimension
        self.run_times = run_times
        self.maxFEs = maxFEs
        self.function = function

        self.Lb = np.zeros(self.dimension)                   # lower bound
        self.Ub = np.zeros(self.dimension)                   # upper bound
        self.Q = np.zeros(self.popsize)                      # frequency

        self.v = np.zeros(self.dimension, self.popsize)      # velocity
        self.sol = np.zeros(self.dimension, self.popsize)    # population of solutions
        self.fitness = np.zeros(self.popsize)                # fitness
        self.best = np.zeros(self.dimension)                 # best solution
        self.f = function

    def custom_pca(self, matrix, numOfComponents=2):
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

    def industry_pca(self, matrix, numOfComponents=2):
        if len(matrix.shape) == 3:
            samples, nx, ny = matrix.shape
            matrix = matrix.reshape(samples, nx*ny)
            numOfComponents = 3
        pca = sklearn.decomposition.PCA(numOfComponents)
        pca = pca.fit(matrix)
        return pca.transform(matrix)

    def init(self):
        pass

    def update_velocity(self):
        pass

    def update_position(self):
        pass

    def position_pca(self):
        pass

    def algorithm(self):
        x_min = np.ninf
        x_max = np.inf
        for t in range(run_times):
            for i in range(popsize):
                r = np.random.uniform(0, 1)
                self.Q[i] = x_min + (x_max - x_min) * r
                # v[i] += (sol[i] - xb) * Q[i]
                # if self.rand() > self.r: 
                #     S[i] = self.localSearch(best=xb, task=task, i=i, Sol=Sol)
                # else: 
                #     S[i] = task.repair(Sol[i] + v[i], rnd=self.Rand)
                # Fnew = task.eval(S[i])
                # if (Fnew <= Fitness[i]) and (self.rand() < self.A): 
                #     Sol[i], Fitness[i] = S[i], Fnew
                # if Fnew <= fxb: 
                #     xb, fxb = S[i].copy(), Fnew

def function(x):
    return sum((-x[i] * np.sin(np.sqrt(np.abs(x[i])))) for i in range(x.shape[0]))

def main():
    rng = np.random.default_rng(seed)
    bat = PCABat(function)
    bat.algorithm()
    M = np.array([[1, 2], [3, 4], [5, 6]])
    print(tuple(dimension))
    M = rng.integers(1000, size=tuple(dimension))
    pca = bat.industry_pca(M)
    print(pca)
    pca = bat.custom_pca(M)
    print(pca)
    print(function(M))

if __name__ == "__main__":
    main()