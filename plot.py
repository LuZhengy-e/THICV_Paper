import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from deployment.utils.sensors import Camera2D

rad = np.pi / 180

config = dict(
    HFOV=25 * rad,
    VFOV=23 * rad,
    fx=2183.375019,
    k=0.05217551,
    prob_thresh=0.6,
    delta_x=2.5
)


def plotMinMax(Hs, phis, HFOV, VFOV, fx, k, prob_thresh, delta_x, is_max=False):
    Hs, phis = np.meshgrid(Hs, phis)
    Ds = np.zeros(shape=phis.shape)
    for i in range(phis.shape[0]):
        for j in range(phis.shape[1]):
            sensor = Camera2D.create(HFOV, VFOV, phis[i, j], Hs[i, j], fx, k, delta_x, prob_thresh)

            if is_max:
                Ds[i, j] = sensor.max_l

            else:
                Ds[i, j] = sensor.min_l

    return Hs, phis, Ds


plot_max = partial(plotMinMax, is_max=True, **config)
plot_min = partial(plotMinMax, is_max=False, **config)

if __name__ == '__main__':
    phis = np.arange(0, np.pi / 2, 5 * rad)
    Hs = np.arange(3, 7, 0.5)
    Hs, phis, Max_Ds = plot_max(Hs, phis)

    fig = plt.figure(figsize=(12, 8))
    ax1 = Axes3D(fig)
    ax1.plot_surface(phis / rad, Hs, Max_Ds, cmap='rainbow')
    ax1.set_xlabel(r"$\phi$")
    ax1.set_ylabel('H/m')
    ax1.set_zlabel('Max field/m')
    plt.savefig("result/max_l.pdf")

    phis = np.arange(0, np.pi / 2, 5 * rad)
    Hs = np.arange(3, 7, 0.5)
    Hs, phis, Min_Ds = plot_min(Hs, phis)
    fig = plt.figure(figsize=(12, 8))
    ax2 = Axes3D(fig)
    ax2.plot_surface(phis / rad, Hs, Min_Ds, cmap='rainbow')
    ax2.set_xlabel(r"$\phi$")
    ax2.set_ylabel('H/m')
    ax2.set_zlabel('Min field/m')

    plt.savefig("result/min_l.pdf")

    # plot deployment
    # plt.figure()
    # ori = np.loadtxt("result/ori.txt", dtype=float)
    # for pt in ori:
    #     if pt[0] == 0 and pt[1] == 0:
    #         continue
    #     plt.scatter(pt[0], pt[1], c="black", s=4)

    # plot stability
    tianMa = np.load("stability.npy", allow_pickle=True)
    tianMa = tianMa.T

    xianDai = np.load("stabilityx.npy", allow_pickle=True)
    xianDai = xianDai.T

    dingXiang = np.load("stabilityb.npy", allow_pickle=True)
    dingXiang = dingXiang.T

    plt.figure(figsize=(16, 9))
    plt.scatter(tianMa[0].tolist(), tianMa[1].tolist(), s=15, c="red", label="Student Apartment")
    plt.scatter(dingXiang[0].tolist(), dingXiang[1].tolist(), s=15, c="blue", label="Dingxiang Garden")
    plt.scatter(xianDai[0].tolist(), xianDai[1].tolist(), s=15, c="green", label="Training Center")
    plt.legend()
    plt.savefig("result/repeat.pdf")

    cost = {
        "Student Apartment": tianMa[0].tolist(),
        "Training Center": xianDai[0].tolist(),
        "Dingxiang Garden": dingXiang[0].tolist(),
    }
    prob = {
        "Student Apartment": tianMa[1].tolist(),
        "Training Center": xianDai[1].tolist(),
        "Dingxiang Garden": dingXiang[1].tolist(),
    }

    c = pd.DataFrame(cost)
    c.plot.box(title="cost of cameras")
    plt.grid(linestyle="--", alpha=0.3)
    plt.ylim([10, 40])
    plt.savefig("result/cost_box.pdf")

    p = pd.DataFrame(prob)
    p.plot.box(title="total pd")
    plt.grid(linestyle="--", alpha=0.3)
    plt.savefig("result/prob_box.pdf")

    plt.show()
