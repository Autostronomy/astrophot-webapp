import numpy as np

model_slider_configs = {
    "Sersic": {
        "n": (0.4, 8, 2),
        "Re": (0, 1, 0.1),
        "q": (0, 1, 0.5),
        "PA": (0, np.pi, np.pi / 2),
        "x": (0, 1, 0.5),
        "y": (0, 1, 0.5),
    },
}
