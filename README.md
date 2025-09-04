# Creatogen Photoshoot Worker (SDXL + ControlNet Depth)

HTTP API server (FastAPI) that generates **photorealistic product shots**
from user photos by preserving geometry (depth) and synthesizing background,
shadows and lighting via SDXL + ControlNet(Depth).

## Endpoints

- `GET /health`
- `POST /generate`

### Request (JSON)

```json
{
  "prompt": "lifestyle flatlay on marble, soft daylight, realistic shadows",
  "negative_prompt": "low quality, blurry, artifacts",
  "images": [{"b64": "<base64 image 1>"}, {"b64": "<base64 image 2>"}],
  "width": 1024,
  "height": 1024,
  "num_infer_steps": 28,
  "guidance_scale": 5.0,
  "seed": 12345,
  "controlnet_strength": 0.9
}


RESPONSE -

{
  "variants": [
    {"image_b64": "<...>", "format": "png", "seed": 12345, "width": 1024, "height": 1024},
    {"image_b64": "<...>", "format": "png", "seed": 12346, "width": 1024, "height": 1024}
  ],
  "meta": { "model_base": "...", "controlnet": "...", "width": 1024, "height": 1024, "seed": 12345 }
}
