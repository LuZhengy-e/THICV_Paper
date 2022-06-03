from .registry import DP_OBJECT


def build_from_dp(name, **kwargs):
    return DP_OBJECT.solve(name, **kwargs)
