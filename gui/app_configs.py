import numpy as np

model_slider_configs = {
    "sersic galaxy model": {
        "x": (0, 1, 0.5),
        "y": (0, 1, 0.5),
        "q": (0, 1, 0.5),
        "PA": (0, np.pi, np.pi / 2),
        "n": (0.4, 8, 2),
        "Re": (0, 1, 0.1),
        "Ie": (0, 1, 0.5),
    },
    "flat sky model": {
        "F": (0, 1, None),
    },
    "point model": {
        "x": (0, 1, 0.5),
        "y": (0, 1, 0.5),
        "flux": (0, 1, None),
    },
}
