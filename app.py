import base64, io, os, math, requests
from typing import Optional, Tuple
import runpod

from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from controlnet_aux import MidasDetector

PIPE = None
DEPTH = None
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SDXL_BASE = os.getenv("SDXL_BASE", "stabilityai/stable-diffusion-xl-base-1.0")
CN_DEPTH  = os.getenv("CN_DEPTH",  "diffusers/controlnet-depth-sdxl-1.0")
IP_ADAPTER_LOADED = False

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

def _resolve_size(ar: Optional[str], width: Optional[int], height: Optional[int]) -> Tuple[int,int]:
    if width and height:
        return width, height
    if not ar:
        return 1024, 1024
    w, h = {
        "1:1": (1,1), "4:5": (4,5), "3:4": (3,4), "2:3": (2,3),
        "16:9": (16,9), "9:16": (9,16)
    }.get(ar, (1,1))
    base_long = 1024
    scale = (base_long / w) if w >= h else (base_long / h)
    W = int(math.floor(w * scale / 64) * 64)
    H = int(math.floor(h * scale / 64) * 64)
    return max(512, W), max(512, H)

def _load_pipeline():
    global PIPE, DEPTH, IP_ADAPTER_LOADED
    if PIPE is not None:
        return

    controlnet = ControlNetModel.from_pretrained(CN_DEPTH, torch_dtype=DTYPE)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SDXL_BASE,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)

    pipe.enable_vae_tiling()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # --- NEW: load IP-Adapter SDXL weights + image encoder
    try:
        # common layout in h94/IP-Adapter for SDXL
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl.safetensors"
        )
        IP_ADAPTER_LOADED = True
    except Exception as e:
        # fallbacks for older repo layouts
        try:
            pipe.load_ip_adapter("h94/IP-Adapter", weight_name="ip-adapter-plus_sdxl.safetensors")
            IP_ADAPTER_LOADED = True
        except Exception as _:
            print(f"[warn] IP-Adapter not loaded: {e}")

    PIPE = pipe
    DEPTH = MidasDetector.from_pretrained("lllyasviel/Annotators")


def handler(event):
    try:
        _load_pipeline()
        inp = event.get("input", {}) if isinstance(event, dict) else {}

        prompt   = inp.get("prompt") or "studio product photo, premium lighting"
        negative = inp.get("negative_prompt", "low quality, watermark, logo, text")
        W, H     = _resolve_size(inp.get("aspect_ratio"), inp.get("width"), inp.get("height"))

        denoise  = float(inp.get("denoise", inp.get("strength", 0.45)))
        denoise  = max(0.10, min(0.80, denoise))
        guidance = float(inp.get("guidance_scale", 5.5))
        scale_cn = float(inp.get("controlnet_scale", 0.8))
        ip_scale = float(inp.get("ip_scale", 0.6))  # NEW: IP-Adapter strength
        seed     = inp.get("seed")
        gen      = torch.Generator(device=DEVICE).manual_seed(int(seed)) if seed is not None else None

        # image input
        if inp.get("image_b64"):
            img = _from_b64(inp["image_b64"])
        elif inp.get("image_url"):
            img = _from_url(inp["image_url"])
        else:
            return {"error": "Provide image_b64 or image_url"}

        init_img = _fit_letterbox(img, W, H, bg=(255, 255, 255))
        depth    = DEPTH(init_img)  # PIL single-channel

        # --- NEW: set IP-Adapter scale (if it loaded)
        if IP_ADAPTER_LOADED:
            try:
                PIPE.set_ip_adapter_scale(ip_scale)
            except Exception:
                pass  # method exists in diffusers ≥0.29/0.30

        # run
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=DTYPE if DEVICE == "cuda" else torch.float32
        ):
            out = PIPE(
                prompt=prompt,
                negative_prompt=negative,
                image=init_img,
                control_image=depth,
                controlnet_conditioning_scale=scale_cn,
                # --- NEW: pass product image to IP-Adapter
                ip_adapter_image=init_img if IP_ADAPTER_LOADED else None,
                guidance_scale=guidance,
                strength=denoise,
                generator=gen,
                num_inference_steps=28,
            )

        result = out.images[0]
        b64_png = _to_png_b64(result)
        return {
            "variants": [{"image_b64": b64_png, "format": "png", "seed": int(seed) if seed else 0}],
            "meta": {
                "width": W, "height": H,
                "guidance_scale": guidance,
                "denoise": denoise,
                "controlnet_scale": scale_cn,
                "ip_scale": ip_scale,
                "ip_adapter_loaded": IP_ADAPTER_LOADED,
                "model": SDXL_BASE,
                "controlnet": CN_DEPTH
            }
        }

    except Exception as e:
        return {"error": str(e)}
    
runpod.serverless.start({"handler": handler})
