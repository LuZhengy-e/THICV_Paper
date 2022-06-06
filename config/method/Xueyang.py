import numpy as np
from ..config import config

rad = np.pi / 180

params = config.register_from_file("Xueyang",
                                   dict(
                                       method="deployment_one_road",
                                       gap_pole=25,
                                       cost_pole=5,
                                       cost_cameras=10,
                                       cameras_info=[0, 86, 3, 30],
                                       road_list=["secondary", "tertiary"]
                                   )
                                   )
