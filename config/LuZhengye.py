import numpy as np

rad = np.pi / 180

params = dict(
    method="deployment_lu",
    Hs=np.arange(3, 5, 1),
    phis=np.arange(0, np.pi / 2, 15 * rad),
    gap_pole=50,
    gap_road=2.5,
    cameras_info=dict(
        HFOV=15,
        VFOV=23,
        fx=2183.375019,
        k=0.05217551,
        delta_x=1.75,
        prob_thresh=0.6
    )
)