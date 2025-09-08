# file: rp_handler.py
import runpod

def handler(event):
    """
    event['input'] holds the JSON you POST to the endpoint.
    Return any JSON-serializable object.
    """
    prompt = event["input"].get("prompt", "a photo")
    # TODO: call your existing generation function here
    # result = generate_image(prompt, ...)
    return {"ok": True, "prompt": prompt}

runpod.serverless.start({"handler": handler})
