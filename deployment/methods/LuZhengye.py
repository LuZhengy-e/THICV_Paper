import geatpy as ea
import json
import numpy as np
from loguru import logger
from tqdm import tqdm
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from ..utils.maps import LoaclMap, Line, Point, Pos
from ..utils.sensors import Camera2D
from ..registry import DP_OBJECT

OPEN = True
PLOT = False


def plotHeatMap(vars, poles, Hs, phis, min_x, min_y, max_x, max_y, cameras_info):
    x_, y_ = np.linspace(min_x, max_x, 200), np.linspace(min_y, max_y, 200)
    x, y = np.meshgrid(x_, y_)
    prob = np.ones((200, 200))
    for r in range(200):
        for c in range(200):
            pt = Point(x[r, c], y[r, c], 0, "0")
            for i in range(len(vars)):
                if vars[i] == 1:
                    pole_idx, h_idx, phi_idx, theta_idx = get_theta_and_H(i, len(poles), len(phis), len(Hs), 2)
                    pole = poles[pole_idx]

                    pole_pos = pole["pos"]
                    h = Hs[h_idx]
                    phi = phis[phi_idx]
                    theta = pole["theta"] if theta_idx == 0 else -pole["theta"]

                    camera = Camera2D.create(phi=phi, H=h, **cameras_info)
                    if camera.is_point_vis(pt,
                                           np.array([pole_pos.x, pole_pos.y]),
                                           theta):
                        prob[r, c] = prob[r, c] * (1 - camera.calculate_prob(pt,
                                                                             pole_pos,
                                                                             theta,
                                                                             phi, h,
                                                                             k=cameras_info["k"] * cameras_info[
                                                                                 "delta_x"] * cameras_info["fx"]))

    fig = plt.figure(figsize=(16, 9))
    ax = Axes3D(fig)
    ax.plot_surface(x, y, 1 - prob, cmap='Blues')
    cset = ax.contour(x, y, 1 - prob, zdir='z', offset=1.25, cmap=cm.coolwarm)
    ax.set_xlabel("x/m")
    ax.set_ylabel('y/m')
    ax.set_zlabel('prob')

    fig = plt.figure()
    ax = sns.heatmap(1 - prob, xticklabels=[], yticklabels=[], cmap='Blues')
    ax.set_xlabel("x/m")
    ax.set_ylabel('y/m')

    plt.show()


class objectFunction:
    def __init__(self, A):
        self.A = A.copy()
        self.k = -2.5

    def __call__(self, x):
        # self.k = min(1500, self.k + 5)
        a = np.expand_dims(np.sum(x.T, axis=0), axis=1)
        # b = np.expand_dims(np.array(
        #     [
        #         np.sum(np.prod(1 - self.A * u, axis=1), axis=0) for u in x
        #     ]
        # ), axis=1)
        # return a + self.k * b
        return a


class objectFunction2:
    def __init__(self, P, k, pt_size):
        self.P = P
        self.k = k
        self.pt_size = pt_size

    def __call__(self, x):
        size = x.shape[0]
        a = np.expand_dims(np.array(
            [
                np.sum(
                    1 - np.prod(1 - self.P * u, axis=1), axis=0
                ) for u in x
            ]
        ) / self.pt_size, axis=1)

        if self.k == 0:
            return a

        b = np.expand_dims(np.array(
            [
                np.sum(
                    (1 - np.prod(1 - self.P * x[i], axis=1) - a[i]) ** 2
                ) for i in range(size)
            ],
        ) / self.pt_size, axis=1)

        return a - self.k * b


def init_pop(NIND, var_num):
    np.random.seed(157)
    tmp = np.zeros((NIND, var_num))
    prob = 0
    gap = 1 / (2 * NIND)
    for i in range(NIND):
        for j in range(var_num):
            if np.random.rand() < prob:
                tmp[i, j] = 1

        prob += gap

    return tmp.copy()


class ProblemwithConstraint(ea.Problem):
    def __init__(self, f, constraint, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin, init=None):
        self.f = f
        self.constraint = constraint
        self.init = init
        self.flag = 0
        super(ProblemwithConstraint, self).__init__(name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        if self.flag == 0 and self.init is not None:
            pop.Phen = self.init()
            self.flag += 1
            logger.debug("init by self successfully")

        vars = pop.Phen
        res = []
        for fun in self.f:
            res.append(fun(vars))

        cons = []
        for fun in self.constraint:
            cons.append(fun(vars))

        pop.ObjV = np.hstack(res)
        pop.CV = np.hstack(cons)


def get_theta_and_H(index, num_pole, num_phi, num_H, num_theta):
    tmp = num_phi * num_H * num_theta
    pole_idx = index // tmp
    res = index % tmp

    tmp = num_phi * num_theta
    H_idx = res // tmp
    res = res % tmp

    phi_idx = res // num_theta
    theta_idx = res % num_theta

    return pole_idx, H_idx, phi_idx, theta_idx


@DP_OBJECT.register
def deployment_lu(map: LoaclMap, Hs, phis, gap_pole, gap_road, cameras_info):
    pts, poles = [], []
    logger.info("---------------start discretization----------------")
    for road_id in map.get_lines():
        road: Line = map.get_line(road_id)
        # != "secondary" and road.get_tag("highway") != "tertiary"
        if road.get_tag("highway") is None:
            continue

        pt_list = road.get_pts()
        for i in range(len(pt_list) - 1):
            cur_pt, next_pt = pt_list[i:i + 2]
            pos = Pos.copy_from_point(cur_pt)

            direct = np.array(
                [next_pt.x - cur_pt.x, next_pt.y - cur_pt.y, next_pt.z - cur_pt.z]
            )
            direct /= np.linalg.norm(direct)
            direct2D = direct[0:3] / np.linalg.norm(direct[0:3])
            theta = np.arccos(direct2D[0])
            if direct2D[1] < 0:
                theta = 0 - theta

            pts.append(Point(pos.x, pos.y, pos.z, idx="0"))
            while pos.can_walk(direct, gap_road, next_pt):
                pos.walk(direct, gap_road)
                pts.append(Point(pos.x, pos.y, pos.z, idx="0"))

            pos = Pos.copy_from_point(cur_pt)
            poles.append(
                {"pos": Point(pos.x, pos.y, pos.z, idx="0"),
                 "theta": theta}
            )
            while pos.can_walk(direct, gap_pole, next_pt):
                pos.walk(direct, gap_pole)
                poles.append(
                    {"pos": Point(pos.x, pos.y, pos.z, idx="0"),
                     "theta": theta}
                )

    pole_size = len(poles) * (len(Hs)) * len(phis) * 2
    pt_size = len(pts)

    A = np.zeros((pt_size, pole_size), dtype=int)
    P = np.zeros((pt_size, pole_size), dtype=float)

    logger.info("---------------start calculate----------------")
    if OPEN:
        for i in tqdm(range(pt_size)):
            pt = pts[i]
            for j in range(pole_size):
                pole_idx, h_idx, phi_idx, theta_idx = get_theta_and_H(j, len(poles), len(phis), len(Hs), 2)
                pole = poles[pole_idx]

                pole_pos = pole["pos"]
                h = Hs[h_idx]
                phi = phis[phi_idx]
                theta = pole["theta"] if theta_idx == 0 else -pole["theta"]

                camera = Camera2D.create(phi=phi, H=h, **cameras_info)
                if camera.is_point_vis(pt,
                                       np.array([pole_pos.x, pole_pos.y]),
                                       theta):
                    A[i, j] = 1
                    P[i, j] = camera.calculate_prob(pt,
                                                    pole_pos,
                                                    theta,
                                                    phi, h,
                                                    k=cameras_info["k"] * cameras_info["delta_x"] * cameras_info["fx"])

    # save and load
    if OPEN:
        np.savetxt("result/A.txt", A)
        np.savetxt("result/P.txt", P)
    else:
        A = np.loadtxt("result/A.txt", dtype=float)
        P = np.loadtxt("result/P.txt", dtype=float)

    if PLOT:
        vars = np.loadtxt("result/ga/vars.txt")
        vars_ = sorted(vars.tolist(),
                       key=lambda x: np.sum(x))
        var = vars_[0]
        plotHeatMap(var, poles, Hs, phis, -40, -20, 130, 100, cameras_info)

        return not PLOT

    f = [
        objectFunction(A),
        objectFunction2(P, 0, pt_size)
    ]
    cons = [
        lambda x: (1 - np.dot(A, x.T)).T
    ]

    params = {
        "name": "deploy",
        "M": 2,
        "maxormins": [1, -1],
        "Dim": pole_size,
        "varTypes": [1] * pole_size,
        "lb": [0] * pole_size,
        "ub": [1] * pole_size,
        "lbin": [1] * pole_size,
        "ubin": [1] * pole_size
    }

    NIND = 5000
    init = partial(init_pop, NIND, pole_size)

    problem = ProblemwithConstraint(f, cons, init=None, **params)
    algorithm = ea.moea_NSGA2_templet(problem,
                                      ea.Population(Encoding='BG', NIND=NIND),
                                      MAXGEN=600,  # 最大进化代数
                                      logTras=20)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # 求解
    logger.info("---------------start optimization----------------")
    res = ea.optimize(algorithm, seed=157, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True,
                      dirName='result/ga_ori')

    np.savetxt("result/ga_ori_s/vars_single.txt", res["Vars"])
    vars_ = sorted(res["Vars"].tolist(),
                   key=lambda x: np.sum(x))
    vars = vars_[0]

    plt.figure()
    for i in range(len(vars)):
        if vars[i] == 1:
            pole_idx, h_idx, phi_idx, theta_idx = get_theta_and_H(i, len(poles), len(phis), len(Hs), 2)
            pole = poles[pole_idx]

            pole_pos = pole["pos"]
            h = Hs[h_idx]
            phi = phis[phi_idx]
            theta = pole["theta"] if theta_idx == 0 else -pole["theta"]

            camera = Camera2D.create(phi=phi, H=h, **cameras_info)
            camera.deployment(map,
                              np.array([pole_pos.x, pole_pos.y]),
                              theta,
                              pole_pos.z)

    f1 = objectFunction(A)
    f2 = objectFunction2(P, 0, pt_size)
    objs = np.hstack([f1(np.array(vars_)), f2(np.array(vars_))])

    np.savetxt("result/single_ori.txt", objs)

    plt.show()
