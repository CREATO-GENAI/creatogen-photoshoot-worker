import base64
import io
import os
import time
from typing import Optional, List

import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel, Field

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

# depth preprocessor
from controlnet_aux.midas import MidasDetector

# -------- Settings --------
HF_TOKEN = os.getenv("HF_TOKEN", None)

MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_VAE = "madebyollin/sdxl-vae-fp16-fix"           # good VAE for SDXL
CONTROLNET_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"  # SDXL depth ControlNet

# device / dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

app = FastAPI(title="Creatogen Photoshoot Worker", version="0.1.0")

# Global objects
_pipe = None
_depth = None


def _b64_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _letterbox(im: Image.Image, out_w: int, out_h: int, bg=(255, 255, 255)) -> Image.Image:
    """Keep aspect, pad to exactly (out_w, out_h)."""
    w, h = im.size
    scale = min(out_w / w, out_h / h)
    nw, nh = int(w * scale), int(h * scale)
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (out_w, out_h), bg)
    ox, oy = (out_w - nw) // 2, (out_h - nh) // 2
    canvas.paste(im2, (ox, oy))
    return canvas


def load_pipeline():
    global _pipe, _depth

    if _pipe is not None:
        return _pipe

    print("[load] loading SDXL base + VAE + ControlNet (depth)…")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_DEPTH, torch_dtype=DTYPE, use_safetensors=True, token=HF_TOKEN
    )
    vae = AutoencoderKL.from_pretrained(MODEL_VAE, torch_dtype=DTYPE, use_safetensors=True, token=HF_TOKEN)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        MODEL_BASE,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=DTYPE,
        use_safetensors=True,
        token=HF_TOKEN,
    )

    # speed-ups
    if DEVICE == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # depth preprocessor
    _depth = MidasDetector.from_pretrained("lllyasviel/Annotators")

    _pipe = pipe
    print("[load] pipeline ready.")
    return _pipe


# ---------- Schemas ----------
class Overlay(BaseModel):
    headline: Optional[str] = ""
    sub: Optional[str] = ""
    brand: Optional[str] = ""
    layout: Optional[str] = "bottom"        # "top" | "bottom"
    banner_pct: float = 0.18
    bg: str = "#000000CC"
    fg: str = "#FFFFFF"


class ImageIn(BaseModel):
    b64: Optional[str] = None
    url: Optional[str] = None  # not used in this first version


class GenerateIn(BaseModel):
    prompt: str = Field(..., description="Positive prompt for the scene/background.")
    negative_prompt: Optional[str] = Field(default="low quality, blurry, artifacts")
    images: List[ImageIn]

    width: int = 1024
    height: int = 1024
    num_infer_steps: int = 28
    guidance_scale: float = 5.0
    seed: Optional[int] = None

    # Control strength
    controlnet_strength: float = 0.9

    # Optional overlay (drawn client-side usually; we can ignore here)
    overlay: Optional[Overlay] = None


class VariantOut(BaseModel):
    image_b64: str
    format: str = "png"
    seed: int
    width: int
    height: int


class GenerateOut(BaseModel):
    variants: List[VariantOut]
    meta: dict


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE}


@app.on_event("startup")
def _startup():
    load_pipeline()


@app.post("/generate", response_model=GenerateOut)
def generate(req: GenerateIn):
    """
    1) Auto-pick the sharpest of up to N uploaded product photos (simple variance-of-Laplacian).
    2) Compute a depth map for that image.
    3) Run SDXL + ControlNet(Depth) using the user's prompt to synthesize background, lighting, shadows.
    """
    pipe = load_pipeline()

    if not req.images:
        return {"variants": [], "meta": {"error": "No images"}}

    # Decode to PIL and keep a normalized list
    decoded: list[Image.Image] = []
    for item in req.images:
        if item.b64:
            decoded.append(_b64_to_pil(item.b64))
        # (URL fetch omitted for simplicity)

    # Auto-pick: use a quick sharpness metric
    def sharpness(im: Image.Image) -> float:
        import cv2
        arr = np.array(im.convert("L"))
        return cv2.Laplacian(arr, cv2.CV_64F).var()

    picked = max(decoded, key=sharpness)

    # Fit to requested canvas (keeps aspect)
    product_rgb = _letterbox(picked, req.width, req.height, bg=(255, 255, 255))

    # Prepare depth map via MiDaS
    depth_image = _depth(product_rgb)

    # Seed
    generator = torch.Generator(device=DEVICE)
    seed = req.seed if isinstance(req.seed, int) else int(time.time()) % 10_000_000
    generator.manual_seed(seed)

    # Run pipeline
    images = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        image=product_rgb,          # conditioning image for SDXL (used as initial content)
        controlnet_conditioning_image=depth_image,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_infer_steps,
        guidance_scale=req.guidance_scale,
        controlnet_conditioning_scale=req.controlnet_strength,
        generator=generator,
    ).images

    # Return 2 variants (run twice w/ same depth, different minor noise)
    variants = []
    for i, im in enumerate(images[:2]):
        b64 = _pil_to_b64(im, fmt="PNG")
        variants.append(
            VariantOut(
                image_b64=b64,
                format="png",
                seed=seed + i,
                width=req.width,
                height=req.height,
            )
        )

    meta = {
        "model_base": MODEL_BASE,
        "controlnet": CONTROLNET_DEPTH,
        "width": req.width,
        "height": req.height,
        "seed": seed,
    }
    return GenerateOut(variants=variants, meta=meta)
