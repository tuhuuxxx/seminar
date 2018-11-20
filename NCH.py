import numpy as np
from scipy.optimize import minimize

class NCHClassifier():
    def __init__(self, points, test, C):
        self.points = np.concatenate((points, test), axis=0)
        self.l = self.points.shape[0]
        self.y = np.asarray([1 for _ in range(self.l-1)] + [-1])
        self.bnds = [(0, C) for i in range(self.l-1)] + [(None, None)]
        
    def solve(self):
        def func(x):
            gt = 0
            for i in range(self.l):
                for j in range(self.l):
                    gt += x[i]*x[j]*self.y[i]*self.y[j]*(np.sum(self.points[i]*self.points[j]))
            return -(x[self.l-1] - gt/2)
        xinit = [0.1 for _ in range(self.l)]
        bnds = self.bnds
        cons = [{"type": "eq", "fun": lambda x: np.sum(self.y*x)}]
        sol = minimize(func, x0=xinit, bounds=bnds, constraints=cons)
        
        return -sol.fun

if __name__ == "__main__":
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    test = np.array([[0, 0]])
    C = 1
    nch_clf = NCHClassifier(points, test, C)
    print(nch_clf.solve())
