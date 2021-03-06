import numpy as np
from ..config import config

rad = np.pi / 180

params = config.register_from_file("LuZhengye",
                                   dict(
                                       method="deployment_lu",
                                       Hs=np.arange(3, 5, 1),
                                       phis=np.arange(0, np.pi / 2, 15 * rad),
                                       gap_pole=50,
                                       gap_road=2.5,
                                       cameras_info=dict(
                                           HFOV=15 * rad,
                                           VFOV=23 * rad,
                                           fx=2183.375019,
                                           k=0.05217551,
                                           delta_x=1.75,
                                           prob_thresh=0.6
                                       )
                                   )
                                   )
