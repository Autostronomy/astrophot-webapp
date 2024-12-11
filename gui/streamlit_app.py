import os

import matplotlib.pyplot as plt
import streamlit as st
from streamlit_theme import st_theme
import torch
import numpy as np
from astropy.io import fits
import astrophot as ap
from scipy.stats import iqr
import yaml

from app_configs import model_slider_configs

# Build AstroPhot model
#####################################################################

if "skytoggle" not in st.session_state:
    st.session_state.skytoggle = False

if "fitmessage" not in st.session_state:
    st.session_state.fitmessage = ""

if "target" not in st.session_state:
    st.session_state.target = ap.image.Target_Image(
        data=np.array(fits.open("init_image.fits")[0].data, dtype=np.float64),
        pixelscale=0.262,
        zeropoint=22.5,
        variance="auto",
    )


def initialize_psf():
    target = st.session_state.target
    psf_size = int(max(target.data.shape) // 4)
    psf_size = psf_size + 1 if psf_size % 2 == 0 else psf_size
    psf = ap.models.AstroPhot_Model(
        model_type="moffat psf model",
        target=ap.image.PSF_Image(
            data=np.zeros((psf_size, psf_size)),
            pixelscale=target.pixelscale,
        ),
        parameters={"n": 2.0, "Rd": target.pixel_length.item() * 2},
    )
    psf.initialize()
    st.session_state.psf = psf


if "psf" not in st.session_state:
    initialize_psf()

if "models" not in st.session_state:
    target = st.session_state.target
    if st.session_state.skytoggle:
        sky = ap.models.AstroPhot_Model(
            model_type="flat sky model",
            target=target,
            parameters={"F": np.nanmedian(target.data.numpy()) / target.pixel_area.item()},
        )
        sky.initialize()
        st.session_state.models = [sky]
    else:
        st.session_state.models = []

if "group" not in st.session_state:
    st.session_state.group = ap.models.AstroPhot_Model(
        model_type="group model", target=st.session_state.target, models=st.session_state.models
    )


def reset_models():
    target = st.session_state.target
    if st.session_state.skytoggle:
        sky = ap.models.AstroPhot_Model(
            model_type="flat sky model",
            target=target,
            parameters={"F": np.nanmedian(target.data.numpy()) / target.pixel_area.item()},
        )
        sky.initialize()
        st.session_state.models = [sky]
    else:
        st.session_state.models = []
    st.session_state.group = ap.models.AstroPhot_Model(
        model_type="group model", target=st.session_state.target, models=st.session_state.models
    )


def change_sky_toggle():
    st.session_state.skytoggle = not st.session_state.skytoggle
    if st.session_state.skytoggle:
        sky = ap.models.AstroPhot_Model(
            model_type="flat sky model",
            target=st.session_state.target,
            parameters={
                "F": np.nanmedian(st.session_state.target.data.numpy())
                / st.session_state.target.pixel_area.item()
            },
        )
        sky.initialize()
        st.session_state.models = [sky] + st.session_state.models
        st.session_state.group = ap.models.AstroPhot_Model(
            model_type="group model", target=st.session_state.target, models=st.session_state.models
        )
    else:
        while any([model.model_type == "flat sky model" for model in st.session_state.models]):
            for i in range(len(st.session_state.models)):
                if st.session_state.models[i].model_type == "flat sky model":
                    break
            st.session_state.models.pop(i)
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
    reset_models()
    initialize_psf()


def add_sersic():
    target = st.session_state.target
    model = ap.models.AstroPhot_Model(
        model_type="sersic galaxy model",
        target=target,
    )
    model.initialize(target=target - st.session_state.group())
    st.session_state.models.append(model)
    st.session_state.group.add_model(model)


def add_star():
    target = st.session_state.target

    model = ap.models.AstroPhot_Model(
        model_type="point model",
        target=target,
        psf=st.session_state.psf,
    )
    model.initialize(target=target - st.session_state.group())
    st.session_state.models.append(model)
    st.session_state.group.add_model(model)


def optimize_model():
    result = ap.fit.LM(st.session_state.group, verbose=0).fit()
    st.session_state.fitmessage = result.message


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
st.sidebar.write("Get it for yourself:")
st.sidebar.code("pip install astrophot", language=None)

st.sidebar.write("Upload your own image:")

uploaded_file = st.sidebar.file_uploader(
    "Choose a FITS file", type=["fits"], accept_multiple_files=False
)
set_target(uploaded_file)

st.sidebar.write("Select a model to add:")
sky_toggle = st.sidebar.toggle("add sky level", value=False, on_change=change_sky_toggle)
sersic_button = st.sidebar.button("add Sersic", on_click=add_sersic)
star_button = st.sidebar.button("add Moffat star", on_click=add_star)
reset_button = st.sidebar.button("reset", on_click=reset_models)

st.sidebar.write("Optimize the models")
optimize_button = st.sidebar.button("Optimize", on_click=optimize_model)

st.sidebar.write(f"fit result: {st.session_state.fitmessage}")

state = st.session_state.group.get_state()
st.sidebar.download_button(
    "Download model params",
    yaml.dump(state, indent=2),
    "model_params.yaml",
)


st.title("AstroPhot fitting demo")
# Create a two-column layout
col1, col2 = st.columns([3, 5])

# Sliders for model parameters in the first column
with col1:
    st.header(r"$\textsf{\tiny Model Parameters}$", divider="blue")
    target = st.session_state.target
    if any([model.model_type == "point model" for model in st.session_state.models]):
        with st.expander("PSF model parameters"):
            psf = st.session_state.psf
            # n
            val = st.slider(
                "Moffat n",
                min_value=0.4,
                max_value=4.0,
                value=psf["n"].value.item(),
                key="PSF n",
            )
            psf["n"].value = val
            # Rd
            val = st.slider(
                "Moffat Rd",
                min_value=0.0,
                max_value=10.0 * target.pixel_length.item(),
                value=psf["Rd"].value.item(),
                key="PSF Rd",
            )
            psf["Rd"].value = val

    for model in st.session_state.models:
        with st.expander(f"{model.name}: {model.model_type} parameters"):
            for label, bounds in model_slider_configs[model.model_type].items():
                if label in ["x"]:
                    scale = target.pixel_length.item() * target.data.shape[0]
                elif label in ["y"]:
                    scale = target.pixel_length.item() * target.data.shape[1]
                elif label in ["Re"]:
                    scale = target.pixel_length.item() * target.data.shape[0] / 2
                elif label in ["Rd"]:
                    scale = target.pixel_length.item() * 10
                else:
                    scale = 1.0
                if label == "x":
                    if model["center"].value is None:
                        model["center"].value = (bounds[2] * scale, 0.5 * scale)
                    val = st.slider(
                        label,
                        min_value=bounds[0] * scale,
                        max_value=bounds[1] * scale,
                        value=model["center"].value[0].item(),
                        key=f"{model.name}_{label}",
                    )
                    model["center"].value = (val, model["center"].value[1])
                elif label == "y":
                    val = st.slider(
                        label,
                        min_value=bounds[0] * scale,
                        max_value=bounds[1] * scale,
                        value=model["center"].value[1].item(),
                        key=f"{model.name}_{label}",
                    )
                    model["center"].value = (model["center"].value[0], val)
                elif label == "F":
                    if model[label].value is None:
                        model[label].value = (
                            np.nanmedian(target.data.numpy()) / target.pixel_area.item()
                        )
                    skymin = np.quantile(target.data.numpy(), 0.01) / target.pixel_area.item()
                    skymax = np.quantile(target.data.numpy(), 0.99) / target.pixel_area.item()
                    val = st.slider(
                        label,
                        min_value=skymin,
                        max_value=skymax,
                        value=model[label].value.item(),
                        step=(skymin - skymax) / 100,
                        key=f"{model.name}_{label}",
                    )
                    model[label].value = val
                elif label in ["Ie"]:
                    if model[label].value is None:
                        model[label].value = np.log10(
                            np.nanmedian(target.data.numpy() / target.pixel_area.item())
                        )
                    intensitymin = np.log10(
                        iqr(target.data.numpy()) / 10 / target.pixel_area.item()
                    )
                    intensitymax = np.log10(np.max(target.data.numpy()) / target.pixel_area.item())
                    val = st.slider(
                        label,
                        min_value=intensitymin,
                        max_value=intensitymax,
                        value=model[label].value.item(),
                        key=f"{model.name}_{label}",
                    )
                    model[label].value = val
                elif label in ["flux"]:
                    intensitymin = np.log10(iqr(target.data.numpy()) / 10)
                    intensitymax = np.log10(np.max(target.data.numpy()) * 1e3)
                    val = st.slider(
                        label,
                        min_value=intensitymin,
                        max_value=intensitymax,
                        value=model[label].value.item(),
                        key=f"{model.name}_{label}",
                    )
                    model[label].value = val
                else:
                    if model[label].value is None:
                        model[label].value = bounds[2] * scale
                    val = st.slider(
                        label,
                        min_value=bounds[0] * scale,
                        max_value=bounds[1] * scale,
                        value=model[label].value.item(),
                        key=f"{model.name}_{label}",
                    )
                    model[label].value = val


# Display the fit
with col2:
    st.header(r"$\textsf{\tiny Visualization}$")

    fig, axarr = plt.subplots(1, 3, figsize=(10, 3))
    ap.plots.target_image(fig, axarr[0], st.session_state.target)
    axarr[0].set_title("Target Image")
    axarr[0].axis("off")
    ap.plots.model_image(fig, axarr[1], st.session_state.group, showcbar=False)
    axarr[1].set_title("Model Image")
    axarr[1].axis("off")
    ap.plots.residual_image(
        fig, axarr[2], st.session_state.group, showcbar=False, normalize_residuals=True
    )
    axarr[2].set_title("Residual Image")
    axarr[2].axis("off")
    st.pyplot(fig)
