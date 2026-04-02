import tempfile
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
import pillow_avif  # noqa: F401 — registers AVIF with Pillow
from fastai.vision.all import load_learner, PILImage

st.set_page_config(
    page_title="Imagenette Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path("/app/imagenette_classifier.pkl")

# Folder names in Imagenette are WordNet IDs; show readable names in the UI.
IMAGENETTE_SYNSET_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "english springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "french horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def human_label(cat) -> str:
    key = str(cat)
    if key in IMAGENETTE_SYNSET_TO_NAME:
        return IMAGENETTE_SYNSET_TO_NAME[key].title()
    return key


# Imagenette / fastai default CNN pipeline (squish 224 + ImageNet stats).
# Bypasses learn.predict → test_dl, which breaks on Python 3.13 + fasttransform (str/dict TypeError).
INFER_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def classify_image_path(learn, path: str):
    dev = next(learn.model.parameters()).device
    dtype = next(learn.model.parameters()).dtype
    pil = Image.open(path).convert("RGB").resize(
        (INFER_SIZE, INFER_SIZE), Image.Resampling.LANCZOS
    )
    x = TVF.pil_to_tensor(pil).float() / 255.0
    x = x.unsqueeze(0).to(dev)
    mean = torch.tensor(_IMAGENET_MEAN, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean) / std
    if dtype == torch.float16:
        x = x.half()
    learn.model.eval()
    with torch.no_grad():
        out = learn.model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    probs = torch.softmax(out.float(), dim=-1)[0].cpu()
    idx = int(probs.argmax())
    pred = learn.dls.vocab[idx]
    return pred, idx, probs


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=Fraunces:opsz,wght@9..144,600&display=swap');
      html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif !important; }
      h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.02em; }
      .block-container { padding-top: 2.75rem !important; max-width: 1100px !important; }
      div[data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 600px at 10% -10%, rgba(236, 72, 153, 0.09), transparent 55%),
                    radial-gradient(900px 500px at 100% 0%, rgba(244, 114, 182, 0.1), transparent 50%),
                    #fff5f7;
      }
      .hero-wrap {
        padding: 1.85rem 1.75rem 1.25rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(253, 242, 248, 0.9) 100%);
        border: 1px solid rgba(190, 24, 93, 0.18);
        box-shadow: 0 12px 40px rgba(131, 24, 67, 0.07);
        margin-bottom: 0.5rem;
        margin-top: 0.35rem;
      }
      .hero-wrap p.hero-kicker {
        display: inline-block;
        font-size: 0.8125rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        color: #9f1239 !important;
        -webkit-text-fill-color: #9f1239 !important;
        font-weight: 700;
        margin: 0.15rem 0 0.5rem 0;
        padding: 0.4rem 0.75rem;
        border-radius: 8px;
        background: linear-gradient(180deg, rgba(251, 207, 232, 0.95) 0%, rgba(252, 231, 243, 0.9) 100%);
        border: 1px solid rgba(190, 24, 93, 0.35);
        box-shadow: 0 1px 2px rgba(131, 24, 67, 0.06);
      }
      .hero-wrap h1.hero-title,
      div[data-testid="stMarkdownContainer"] h1.hero-title {
        font-size: 2rem !important;
        margin: 0 !important;
        line-height: 1.15 !important;
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        background: none !important;
        background-clip: unset !important;
      }
      .hero-wrap p.hero-lede {
        margin: 0.5rem 0 0;
        color: #be185d !important;
        -webkit-text-fill-color: #be185d !important;
        font-size: 1.02rem;
        max-width: 36rem;
        line-height: 1.45;
      }
      .result-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        background: rgba(236, 72, 153, 0.22) !important;
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        font-weight: 600;
        font-size: 1.05rem;
      }
      div[data-testid="stMarkdownContainer"] p span.result-pill,
      div[data-testid="stMarkdownContainer"] .result-pill {
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        background: rgba(236, 72, 153, 0.22) !important;
        background-image: none !important;
      }
      div[data-testid="stFileUploader"] section {
        padding: 1rem;
        border-radius: 12px;
        border: 1px dashed rgba(219, 39, 119, 0.45) !important;
        background: rgba(255,255,255,0.72);
      }
      div[data-testid="stMarkdownContainer"] h5 {
        color: #9d174d !important;
        -webkit-text-fill-color: #9d174d !important;
        font-weight: 600 !important;
        background: none !important;
      }
      div[data-testid="stMarkdownContainer"] h6 {
        color: #db2777 !important;
        -webkit-text-fill-color: #db2777 !important;
        font-weight: 600 !important;
        background: none !important;
        background-color: transparent !important;
      }
      div[data-testid="stMarkdownContainer"] h6 * {
        color: #db2777 !important;
        -webkit-text-fill-color: #db2777 !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
      }
      div[data-testid="stMarkdownContainer"] p:not(.hero-lede):not(.hero-kicker) {
        color: #500724 !important;
        -webkit-text-fill-color: #500724 !important;
      }
      /* File uploader: inner rows often get primary chip + white text */
      [data-testid="stFileUploader"] label,
      [data-testid="stFileUploader"] label * {
        color: #9d174d !important;
        -webkit-text-fill-color: #9d174d !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
      }
      [data-testid="stFileUploader"] section p,
      [data-testid="stFileUploader"] section p *,
      [data-testid="stFileUploader"] section span,
      [data-testid="stFileUploader"] section small,
      [data-testid="stFileUploader"] section div {
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
      }
      [data-testid="stFileUploaderDropzoneInstructions"],
      [data-testid="stFileUploaderDropzoneInstructions"] *,
      [data-testid="stFileUploaderDropzone"] p,
      [data-testid="stFileUploaderDropzone"] span,
      [data-testid="stFileUploaderDropzone"] div {
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
      }
      /* Primary button: white on pink */
      [data-testid="stFileUploader"] button[kind="primary"],
      [data-testid="stFileUploader"] button[kind="primary"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: #db2777 !important;
        background-image: none !important;
      }
      /* Secondary (common for “Browse”): dark pink text */
      [data-testid="stFileUploader"] button[kind="secondary"],
      [data-testid="stFileUploader"] button[kind="secondary"] * {
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        background-image: none !important;
      }
      /* Bordered container caption if present */
      [data-testid="stVerticalBlockBorderWrapper"] > div > div > label {
        color: #9d174d !important;
        -webkit-text-fill-color: #9d174d !important;
      }
      /* Metrics: inner wrappers use primary bg + white text — strip chip, use pink text */
      [data-testid="stMetricContainer"] {
        background: transparent !important;
      }
      [data-testid="stMetricLabel"],
      [data-testid="stMetricLabel"] * {
        color: #9d174d !important;
        -webkit-text-fill-color: #9d174d !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
      }
      [data-testid="stMetricValue"],
      [data-testid="stMetricValue"] * {
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        background: rgba(255, 255, 255, 0.75) !important;
        background-color: rgba(255, 255, 255, 0.75) !important;
        background-image: none !important;
        box-shadow: none !important;
      }
      /* Progress: label row often gets primary fill + white type */
      [data-testid="stProgress"] p,
      [data-testid="stProgress"] p *,
      [data-testid="stProgress"] span,
      [data-testid="stProgress"] label {
        color: #500724 !important;
        -webkit-text-fill-color: #500724 !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
      }
      [data-testid="stProgress"] > div > div:first-child,
      [data-testid="stProgress"] > div > div:first-child * {
        color: #500724 !important;
        -webkit-text-fill-color: #500724 !important;
        background: rgba(255, 255, 255, 0.85) !important;
        background-color: rgba(255, 255, 255, 0.85) !important;
        background-image: none !important;
      }
      /* Spinner status text */
      [data-testid="stSpinner"] p,
      [data-testid="stSpinner"] span {
        color: #9d174d !important;
        -webkit-text-fill-color: #9d174d !important;
      }
    </style>
    <div class="hero-wrap">
      <p class="hero-kicker">Imagenette · 10 classes</p>
      <h1 class="hero-title">Image classifier</h1>
      <p class="hero-lede">Upload a photo and see which Imagenette category the model thinks it is — with confidence and top alternatives.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


@st.cache_resource
def get_model():
    return load_learner(MODEL_PATH)


learn = get_model()

ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp", "avif", "gif"]
with st.container(border=True):
    uploaded_file = st.file_uploader(
        "Drop an image here",
        type=ALLOWED_TYPES,
        help="JPG, PNG, WebP, AVIF, GIF",
        label_visibility="visible",
    )

if uploaded_file is not None:
    suffix = Path(uploaded_file.name or "upload.jpg").suffix.lower()
    if suffix not in {f".{t}" for t in ALLOWED_TYPES}:
        suffix = ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    path_str = str(tmp_path)
    img = PILImage.create(path_str)

    with st.spinner("Running the model…"):
        pred, idx, probs = classify_image_path(learn, path_str)

    p = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else probs
    idx_i = int(idx) if not isinstance(idx, int) else idx
    conf = float(p[idx_i])
    top5 = sorted(
        [(learn.dls.vocab[i], float(p[i])) for i in range(len(learn.dls.vocab))],
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    max_s = max((s for _, s in top5), default=1.0) or 1.0

    col_img, col_out = st.columns([1, 1.05], gap="large")
    with col_img:
        st.markdown("##### Your image")
        st.image(img, use_container_width=True)

    with col_out:
        st.markdown("##### Best guess")
        st.markdown(
            f'<p><span class="result-pill">{human_label(pred)}</span></p>',
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{conf:.1%}")
        st.markdown("###### How sure on other labels")
        for label, score in top5:
            st.progress(
                min(score / max_s, 1.0),
                text=f"{human_label(label)} · {score:.1%}",
            )
