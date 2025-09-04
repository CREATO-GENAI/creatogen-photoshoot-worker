import base64, io, os, math, asyncio
from typing import Optional, Tuple
import runpod

from PIL import Image, ImageOps
import requests
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
# depth preprocessor (MiDaS) for ControlNet
from controlnet_aux import MidasDetector

# ------------------------
# Lazy global pipeline
# ------------------------
PIPE = None
DEPTH = None
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SDXL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
CN_DEPTH = os.getenv("CN_DEPTH", "diffusers/controlnet-depth-sdxl-1.0")

def _to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _from_b64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _from_url(url: str) -> Image.Image:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def _fit_letterbox(img: Image.Image, W: int, H: int, bg=(255,255,255)) -> Image.Image:
    iw, ih = img.size
    scale = min(W/iw, H/ih)
    nw, nh = max(1,int(iw*scale)), max(1,int(ih*scale))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), bg)
    ox, oy = (W - nw)//2, (H - nh)//2
    canvas.paste(img2, (ox, oy))
    return canvas

def _resolve_size(aspect_ratio: Optional[str], width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    if width and height:
        return width, height
    # default SDXL square
    if not aspect_ratio:
        return 1024, 1024
    ar = {
        "1:1": (1,1), "4:5": (4,5), "3:4": (3,4), "2:3": (2,3),
        "16:9": (16,9), "9:16": (9,16)
    }.get(aspect_ratio, (1,1))
    base_long = 1024
    w, h = ar
    scale = (base_long / w) if w >= h else (base_long / h)
    W = int(math.floor(w * scale / 64) * 64)
    H = int(math.floor(h * scale / 64) * 64)
    return max(512, W), max(512, H)

def _load_pipeline():
    global PIPE, DEPTH
    if PIPE is not None:
        return
    controlnet = ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=DTYPE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE, controlnet=controlnet,
        torch_dtype=DTYPE, variant="fp16", use_safetensors=True
    )
    pipe.to(DEVICE)
    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    PIPE = pipe
    DEPTH = MidasDetector.from_pretrained("lllyasviel/Annotators")

# ------------------------
# Handler
# ------------------------
def handler(event):
    """
    Input schema:
    {
      "prompt": str,
      "negative_prompt": str (opt),
      "image_b64": str (opt) OR "image_url": str (opt)  <-- provide one,
      "width": int (opt), "height": int (opt), "aspect_ratio": "1:1" | ... (opt),
      "guidance_scale": float (opt, default 5.5),
      "denoise": float (opt, default 0.45),           # lower => more product preservation
      "seed": int (opt),
      "strength": float (alias of denoise)
    }
    """
    try:
        _load_pipeline()

        inp = event.get("input", {}) if isinstance(event, dict) else {}
        prompt = inp.get("prompt") or "studio product photo, premium lighting"
        negative = inp.get("negative_prompt", "low quality, watermark, logo, text")
        W, H = _resolve_size(inp.get("aspect_ratio"), inp.get("width"), inp.get("height"))

        # denoise/strength (img2img)
        denoise = float(inp.get("denoise", inp.get("strength", 0.45)))
        guidance = float(inp.get("guidance_scale", 5.5))
        seed = int(inp.get("seed")) if inp.get("seed") is not None else None
        generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed else None

        # input image
        img: Optional[Image.Image] = None
        if inp.get("image_b64"):
            img = _from_b64(inp["image_b64"])
        elif inp.get("image_url"):
            img = _from_url(inp["image_url"])
        else:
            return {"error": "Provide image_b64 or image_url"}

        # enforce size (letterbox) to requested canvas
        init_img = _fit_letterbox(img, W, H, bg=(255,255,255))

        # depth map for controlnet
        depth = DEPTH(init_img)  # returns PIL single-channel map sized to init_img

        # run pipeline (img2img with controlnet)
        with torch.inference_mode():
            out = PIPE(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                control_image=depth,
                guidance_scale=guidance,
                strength=denoise,
                generator=generator,
                num_inference_steps=28,   # keep reasonably fast
            )

        result: Image.Image = out.images[0]
        b64_png = _to_png_b64(result)

        return {
            "width": W, "height": H,
            "guidance_scale": guidance, "denoise": denoise,
            "format": "png",
            "image_b64": b64_png
        }

    except Exception as e:
        return {"error": str(e)}

# Start serverless
runpod.serverless.start({"handler": handler})
