# -*- coding: utf-8 -*-
"""
Hunyuan3D-2.1 -- HuggingFace Space Frontend Client
====================================================
Lightweight local UI that proxies inference to any running
HuggingFace Space hosting Hunyuan3D-2.1.

No local GPU needed -- all heavy work runs on HF's servers.

Install:
    pip install gradio gradio_client

Run:
    python hf_frontend.py
    python hf_frontend.py --port 8080 --share
"""

import os
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

from pathlib import Path
from typing import Optional

import gradio as gr
from gradio_client import Client, handle_file

# ---------------------------------------------------------------------------
# Known spaces  (checked 2026-05-27)
# ---------------------------------------------------------------------------

KNOWN_SPACES = {
    "tencent/Hunyuan3D-2       (v2.0 official -- LIVE)": "tencent/Hunyuan3D-2",
    "dylanebert/Hunyuan3D-2.1  (v2.1 community -- ZeroGPU)": "dylanebert/Hunyuan3D-2.1",
    "tencent/Hunyuan3D-2.1     (v2.1 official -- may be PAUSED)": "tencent/Hunyuan3D-2.1",
}

DEFAULT_SPACE = "tencent/Hunyuan3D-2"

# ---------------------------------------------------------------------------
# Examples / constants
# ---------------------------------------------------------------------------

EXAMPLES_DIR = Path(__file__).parent / "assets" / "example_images"
EXAMPLE_IMAGES = sorted(EXAMPLES_DIR.glob("*.png"))[:12]
MAX_SEED = int(1e7)

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
#title-block { text-align: center; padding: 20px 0 6px 0; }
#title-block h1 { font-size: 2rem; font-weight: 700; margin: 0; }
#title-block p  { color: #888; margin: 4px 0 0 0; }
"""

TITLE_HTML = """
<div id="title-block">
  <h1>Hunyuan3D-2.1</h1>
  <p>Image to 3D &nbsp;&middot;&nbsp;
     <a href="https://huggingface.co/spaces/tencent/Hunyuan3D-2.1" target="_blank">HuggingFace Spaces</a>
     &nbsp;&middot;&nbsp; No local GPU needed
  </p>
</div>
"""

PAUSED_HINT = (
    "The selected Space is **PAUSED** or unreachable. "
    "Please pick a different one from the Space dropdown above."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_space(label: str) -> str:
    """Turn a dropdown label into the raw HF space ID."""
    return KNOWN_SPACES.get(label, label).strip()


def _extract_file(item) -> Optional[str]:
    """
    Normalise a gradio_client file result to a local filesystem path.

    gradio_client may return any of:
      - a plain string path             -> return as-is
      - {"path": "...", ...}            -> Gradio 5 FileData
      - {"value": "...", "__type__": "update"}  -> gr.update() wrapper (Gradio 4)
      - {"url": "..."}                  -> remote URL
    """
    if item is None:
        return None
    if isinstance(item, str):
        return item if os.path.exists(item) else None
    if isinstance(item, dict):
        # Try keys in priority order
        for key in ("path", "value", "url"):
            val = item.get(key)
            if val and isinstance(val, str):
                return val
    return None


def _make_client(space_id: str) -> Client:
    return Client(space_id, verbose=False)


def _get_param_names(client: Client, endpoint: str) -> list[str]:
    """Return the parameter names for a given named endpoint."""
    try:
        info = client.view_api(return_format="dict")
        params = info.get("named_endpoints", {}).get(endpoint, {}).get("parameters", [])
        return [p["parameter_name"] for p in params]
    except Exception:
        return []


def _build_kwargs(param_names: list[str], image, steps, guidance_scale,
                  seed, octree_resolution, check_box_rembg, num_chunks,
                  randomize_seed) -> dict:
    """
    Build a keyword-argument dict that matches what the target Space expects.
    Handles both the original tencent API (has 'caption') and stripped forks
    (no 'caption', no mv_image_* keyword names).
    We always pass by position for unknown/unnamed spaces.
    """
    base = {
        "image": handle_file(image),
        "mv_image_front": None,
        "mv_image_back": None,
        "mv_image_left": None,
        "mv_image_right": None,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "octree_resolution": octree_resolution,
        "check_box_rembg": check_box_rembg,
        "num_chunks": num_chunks,
        "randomize_seed": randomize_seed,
    }
    # Add caption only if the space declares it
    if "caption" in param_names:
        base = {"caption": None, **base}

    # If the Space has no named params (empty list), fall back to positional
    if not param_names:
        # Positional order from the original gradio_app.py
        # (caption, image, mv_front, mv_back, mv_left, mv_right,
        #  steps, guidance_scale, seed, octree_res, rembg, chunks, rand_seed)
        return list(base.values())   # caller must use *args, not **kwargs

    # Only pass params the Space actually declares (avoids "invalid keyword" errors)
    return {k: v for k, v in base.items() if k in param_names}


# ---------------------------------------------------------------------------
# API discovery (shown in the API Info tab)
# ---------------------------------------------------------------------------

def discover_api(space_label: str) -> str:
    space_id = _resolve_space(space_label)
    try:
        client = _make_client(space_id)
        info = client.view_api(return_format="dict")
        named = info.get("named_endpoints", {})
        lines = [f"Connected to **{space_id}**\n",
                 f"Found **{len(named)}** named endpoint(s):\n"]
        for name, details in named.items():
            params = [p["parameter_name"] for p in details.get("parameters", [])]
            lines.append(f"- `{name}` -- params: {params}")
        return "\n".join(lines)
    except Exception as exc:
        err = str(exc)
        if "PAUSED" in err.upper():
            return PAUSED_HINT
        return (
            f"Could not reach **{space_id}**.\n\nError: `{err}`\n\n"
            "Try a different Space from the dropdown."
        )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _connect_or_raise(space_id: str):
    try:
        return _make_client(space_id)
    except Exception as exc:
        err = str(exc)
        if "PAUSED" in err.upper():
            raise gr.Error(
                f"The Space '{space_id}' is currently **PAUSED** by its owner.\n"
                "Please choose a different Space from the dropdown."
            )
        raise gr.Error(f"Cannot connect to '{space_id}'.\n\n{err}")


def generate_shape(
    space_label: str,
    image,
    steps: int,
    guidance_scale: float,
    seed: int,
    randomize_seed: bool,
    remove_bg: bool,
    octree_resolution: int,
    num_chunks: int,
    progress=gr.Progress(),
):
    if image is None:
        raise gr.Error("Please upload an image first.")

    space_id = _resolve_space(space_label)
    progress(0.05, desc=f"Connecting to {space_id} ...")
    client = _connect_or_raise(space_id)

    # Discover what params this space actually accepts
    param_names = _get_param_names(client, "/shape_generation")

    progress(0.15, desc="Uploading image ...")
    kwargs = _build_kwargs(param_names, image, steps, guidance_scale,
                           seed, octree_resolution, remove_bg,
                           num_chunks, randomize_seed)
    try:
        if isinstance(kwargs, list):
            result = client.predict(*kwargs, api_name="/shape_generation")
        else:
            result = client.predict(**kwargs, api_name="/shape_generation")
    except Exception as exc:
        err = str(exc)
        if "PAUSED" in err.upper():
            raise gr.Error(
                f"The Space '{space_id}' is PAUSED. "
                "Please choose a different Space from the dropdown."
            )
        raise gr.Error(f"Shape generation failed on '{space_id}'.\n\n{err}")

    progress(0.95, desc="Downloading mesh ...")
    glb_path = _extract_file(result[0])
    stats    = result[2] if len(result) > 2 else {}
    new_seed = int(result[3]) if len(result) > 3 else int(seed)

    if glb_path is None:
        raise gr.Error("Generation succeeded but the mesh file could not be retrieved.")

    progress(1.0, desc="Done!")
    return glb_path, stats, new_seed


def generate_textured(
    space_label: str,
    image,
    steps: int,
    guidance_scale: float,
    seed: int,
    randomize_seed: bool,
    remove_bg: bool,
    octree_resolution: int,
    num_chunks: int,
    progress=gr.Progress(),
):
    if image is None:
        raise gr.Error("Please upload an image first.")

    space_id = _resolve_space(space_label)
    progress(0.05, desc=f"Connecting to {space_id} ...")
    client = _connect_or_raise(space_id)

    param_names = _get_param_names(client, "/generation_all")

    progress(0.15, desc="Uploading image ...")
    kwargs = _build_kwargs(param_names, image, steps, guidance_scale,
                           seed, octree_resolution, remove_bg,
                           num_chunks, randomize_seed)
    try:
        if isinstance(kwargs, list):
            result = client.predict(*kwargs, api_name="/generation_all")
        else:
            result = client.predict(**kwargs, api_name="/generation_all")
    except Exception as exc:
        err = str(exc)
        if "PAUSED" in err.upper():
            raise gr.Error(
                f"The Space '{space_id}' is PAUSED. "
                "Please choose a different Space from the dropdown."
            )
        raise gr.Error(f"Textured generation failed on '{space_id}'.\n\n{err}")

    progress(0.95, desc="Downloading mesh ...")
    white_path    = _extract_file(result[0])
    textured_path = _extract_file(result[1]) if len(result) > 1 else None
    stats         = result[3] if len(result) > 3 else {}
    new_seed      = int(result[4]) if len(result) > 4 else int(seed)

    best = textured_path or white_path
    if best is None:
        raise gr.Error("Generation succeeded but the mesh file could not be retrieved.")

    progress(1.0, desc="Done!")
    return best, white_path, stats, new_seed


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    space_choices = list(KNOWN_SPACES.keys())
    default_label = space_choices[0]

    with gr.Blocks(title="Hunyuan3D-2.1 Client", analytics_enabled=False) as demo:

        gr.HTML(TITLE_HTML)

        # -- Space selector --------------------------------------------------
        with gr.Row():
            space_dd  = gr.Dropdown(
                choices=space_choices, value=default_label,
                label="HuggingFace Space",
                info="Switch here if the current Space is PAUSED or slow",
                scale=4, allow_custom_value=True,
            )
            check_btn = gr.Button("Check connection", scale=1, size="sm")

        with gr.Row(equal_height=False):

            # -- Left: inputs ------------------------------------------------
            with gr.Column(scale=2, min_width=320):

                image_in = gr.Image(
                    label="Input Image", type="filepath",
                    image_mode="RGBA", height=280,
                    sources=["upload", "clipboard"],
                )

                with gr.Row():
                    btn_shape = gr.Button("Generate Shape",   variant="primary", scale=1)
                    btn_tex   = gr.Button("+ Texture (PBR)", variant="primary", scale=1)

                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        steps = gr.Slider(1, 100, value=30, step=1, label="Inference Steps")
                        cfg   = gr.Slider(1.0, 15.0, value=5.0, step=0.5, label="Guidance Scale")
                    with gr.Row():
                        seed      = gr.Slider(0, MAX_SEED, value=1234, step=1, label="Seed")
                        rand_seed = gr.Checkbox(value=True, label="Randomise seed")
                    with gr.Row():
                        rembg  = gr.Checkbox(value=True, label="Remove background")
                        octree = gr.Slider(16, 512, value=256, step=16, label="Octree Resolution")
                    chunks = gr.Slider(
                        1000, 200000, value=8000, step=1000,
                        label="Num Chunks (memory vs speed)",
                    )

                with gr.Accordion("Example Images", open=True):
                    if EXAMPLE_IMAGES:
                        gr.Examples(
                            examples=[[str(p)] for p in EXAMPLE_IMAGES],
                            inputs=[image_in], label=None, examples_per_page=6,
                        )
                    else:
                        gr.Markdown("_No examples found._")

            # -- Right: outputs ----------------------------------------------
            with gr.Column(scale=3, min_width=480):
                with gr.Tabs() as out_tabs:

                    with gr.Tab("3D Viewer"):
                        model3d = gr.Model3D(
                            label="Generated Mesh", interactive=False, height=520,
                        )
                        with gr.Row():
                            dl_shape    = gr.DownloadButton(
                                "Download Shape (.glb)",   visible=False, variant="secondary")
                            dl_textured = gr.DownloadButton(
                                "Download Textured (.glb)", visible=False, variant="secondary")

                    with gr.Tab("Mesh Stats"):
                        stats_out = gr.JSON(label="Statistics", value={})

                    with gr.Tab("API Info"):
                        api_md = gr.Markdown(
                            "_Click **Check connection** above to test the Space._"
                        )

        # -- State -----------------------------------------------------------
        seed_state       = gr.State(1234)
        white_path_state = gr.State(None)

        INPUTS = [space_dd, image_in, steps, cfg, seed, rand_seed, rembg, octree, chunks]

        # -- Check connection ------------------------------------------------
        check_btn.click(fn=discover_api, inputs=[space_dd], outputs=[api_md])

        # -- Shape generation ------------------------------------------------
        btn_shape.click(
            fn=generate_shape,
            inputs=INPUTS,
            outputs=[model3d, stats_out, seed_state],
        ).success(
            fn=lambda glb, _s, ns: (
                gr.update(value=glb, visible=bool(glb)),
                gr.update(visible=False),
                ns,
            ),
            inputs=[model3d, stats_out, seed_state],
            outputs=[dl_shape, dl_textured, seed],
        )

        # -- Textured generation --------------------------------------------
        btn_tex.click(
            fn=generate_textured,
            inputs=INPUTS,
            outputs=[model3d, white_path_state, stats_out, seed_state],
        ).success(
            fn=lambda tex, white, _s, ns: (
                gr.update(value=tex, visible=bool(tex)),
                gr.update(value=white, visible=bool(white)),
                ns,
            ),
            inputs=[model3d, white_path_state, stats_out, seed_state],
            outputs=[dl_textured, dl_shape, seed],
        )

        # -- Auto-check on load ----------------------------------------------
        demo.load(fn=discover_api, inputs=[space_dd], outputs=[api_md])

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 HF Space frontend")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--space", type=str, default=DEFAULT_SPACE,
                        help="Default HuggingFace Space ID")
    args = parser.parse_args()

    DEFAULT_SPACE = args.space

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="violet"),
        css=CSS,
    )
