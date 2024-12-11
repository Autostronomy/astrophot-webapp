import os

import matplotlib.pyplot as plt
import streamlit as st
from streamlit_theme import st_theme
import torch
import numpy as np
from astropy.io import fits
import AstroPhot as ap

from app_configs import model_slider_configs

# Build AstroPhot model
#####################################################################


if "target" not in st.session_state:
    st.session_state.target = ap.image.Target_Image(
        data=np.array(fits.open("init_image.fits")[0].data, dtype=np.float64),
        pixelscale=0.262,
        zeropoint=22.5,
        variance="auto",
    )

if "models" not in st.session_state:
    target = st.session_state.target
    st.session_state.models = [
        ap.models.AstroPhot_Model(
            model_type="flat sky model",
            target=target,
            parameters={"F": np.nanmedian(target.data.numpy()) / target.pixelscale**2},
        ),
    ]

if "group" not in st.session_state:
    st.session_state.group = ap.models.AstroPhot_Model(
        model_type="group model", target=st.session_state.target, models=st.session_state.models
    )


def set_target(value):
    try:
        data = np.array(fits.open(value)[0].data, dtype=np.float64)
    except:
        return
    if data.shape == st.session_state.target.data.shape:
        if np.all(data == st.session_state.target.data.numpy()):
            return
    st.session_state.target = ap.image.Target_Image(
        data=data,
        pixelscale=1.0,
        variance="auto",
    )
    target = st.session_state.target
    st.session_state.models = [
        ap.models.AstroPhot_Model(
            model_type="flat sky model",
            target=target,
            parameters={"F": np.nanmedian(target.data.numpy()) / target.pixelscale.item() ** 2},
        ),
    ]


def add_sersic():
    model = ap.models.AstroPhot_Model(
        model_type="sersic galaxy model",
        target=st.session_state.target,
    )
    model.initialize()
    st.session_state.models.append(model)
    st.session_state.group.add_model(model)


def add_nuker():
    model = ap.models.AstroPhot_Model(
        model_type="nuker galaxy model",
        target=st.session_state.target,
    )
    model.initialize()
    st.session_state.models.append(model)
    st.session_state.group.add_model(model)


def add_star():
    model = ap.models.AstroPhot_Model(
        model_type="moffat star model",
        target=st.session_state.target,
    )
    st.session_state.models.append(model)
    st.session_state.group.add_model(model)


# Sidebar
#####################################################################
st.set_page_config(layout="wide")
# css = """
# <style>
#     section.main > div {max-width:75rem}
# </style>
# """
# st.markdown(css, unsafe_allow_html=True)
theme = st_theme()
if theme is None or theme["base"] == "dark":
    logo_url = "https://github.com/Autostronomy/AstroPhot/raw/main/media/AP_logo_white.png?raw=true"
else:
    logo_url = "https://github.com/Autostronomy/AstroPhot/raw/main/media/AP_logo.png?raw=true"
st.sidebar.image(logo_url)
docs_url = "https://astrophot.readthedocs.io/"
st.sidebar.write("Check out the [documentation](%s)!" % docs_url)

st.sidebar.write("Upload your own image:")

uploaded_file = st.sidebar.file_uploader(
    "Choose a FITS file", type=["fits"], accept_multiple_files=False, on_change=set_target
)
set_target(uploaded_file)

st.sidebar.write("Select a model to add:")
sersic_button = st.sidebar.button("add Sersic", on_click=add_sersic)
nuker_button = st.sidebar.button("add Nuker", on_click=add_nuker)
star_button = st.sidebar.button("add Moffat star", on_click=add_star)

st.sidebar.write("Optimize the models")
optimize_button = st.sidebar.button("Optimize")

st.sidebar.write("Get it for yourself: pip install astrophot")

st.title("AstroPhot fitting demo")
# Create a two-column layout
col1, col2 = st.columns([3, 5])

# Sliders for model parameters in the first column
with col1:
    st.header(r"$\textsf{\tiny Model Parameters}$", divider="blue")
    for model in st.session_state.models:
        st.write(f"{model.name}: {model.model_type}")
        for label, bounds in model_slider_configs[model.model_type].items():
            if label in ["Re", "x"]:
                scale = st.session_state.target.pixelscale * st.session_state.target.data.shape[0]
            elif label in ["y"]:
                scale = st.session_state.target.pixelscale * st.session_state.target.data.shape[1]
            else:
                scale = 1.0
            if label == "x":
                if model["center"].value is None:
                    model["center"].value = (bounds[2] * scale, 0.5 * scale)
                val = st.slider(
                    label,
                    min_value=bounds[0] * scale,
                    max_value=bounds[1] * scale,
                    value=model["center"].value[0].numpy(),
                )
                model["center"].value = (val, model["center"].value[1])
            elif label == "y":
                val = st.slider(
                    label,
                    min_value=bounds[0] * scale,
                    max_value=bounds[1] * scale,
                    value=model["center"].value[1].numpy(),
                )
                model["center"].value = (model["center"].value[0], val)
            else:
                if model[label].value is None:
                    model[label].value = bounds[2] * scale
                val = st.slider(
                    label,
                    min_value=bounds[0] * scale,
                    max_value=bounds[1] * scale,
                    value=model[label].value.numpy(),
                )
                model[label].value = val


# Display the fit
with col2:
    st.header(r"$\textsf{\tiny Visualization}$")

    fig, axarr = plt.subplots(3, 1, figsize=(5, 15))
    ap.plots.target_image(fig, axarr[0], st.session_state.target)
    ap.plots.model_image(fig, axarr[1], st.session_state.group)
    ap.plots.residual_image(fig, axarr[2], st.session_state.group)
    st.pyplot(fig)
