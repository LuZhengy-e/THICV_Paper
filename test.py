import numpy as np
import geatpy as ea
from deployment.methods.LuZhengye import ProblemwithConstraint


if __name__ == '__main__':
    A = np.ones((5, 5))
    f = [
        lambda x: np.expand_dims(np.sum(x.T, axis=0), axis=1),
        lambda x: np.expand_dims(np.sum(np.dot(A, x.T), axis=0), axis=1)
    ]
    cons = [lambda x: (1 - np.dot(A, x.T)).T]
    params = {
        "name": "deploy",
        "M": 2,
        "maxormins": [1, -1],
        "Dim": 5,
        "varTypes": [1] * 5,
        "lb": [0] * 5,
        "ub": [1] * 5,
        "lbin": [1] * 5,
        "ubin": [1] * 5
    }
    prob = ProblemwithConstraint(f, cons, **params)
    algorithm = ea.moea_NSGA2_templet(prob,
                                      ea.Population(Encoding='RI', NIND=50),
                                      MAXGEN=200,  # 最大进化代数
                                      logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=1, outputMsg=True, drawLog=False, saveFlag=False,
                      dirName='result')
