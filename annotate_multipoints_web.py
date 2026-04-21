#!/usr/bin/env python3
"""
DragBench multi-point annotation tool (Gradio web UI).

Saves a sidecar pickle (default: meta_data_multi.pkl) next to each sample's
original meta_data.pkl. Sidecar schema is identical to the original:
    {"prompt": str, "points": [[sx0,sy0],[tx0,ty0],...], "mask": np.ndarray}

Switch at inference time: pass --annotation_variant meta_data_multi.pkl to test.py.

Usage:
    python annotate_multipoints_web.py \
        --root /mnt/disk1/datasets/DragBench \
        --bench_type both \
        --sidecar_name meta_data_multi.pkl \
        [--start_index 0] [--only_missing] [--categories human_head,animals] \
        [--port 7878] [--host 127.0.0.1]

Open the printed URL in your browser (VSCode Remote will auto-forward the port).
"""
import argparse
import math
import os
import pickle
import sys

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


# ------------------------ filesystem + schema helpers ------------------------
def list_samples(root, bench_type):
    bench_type = set(bench_type)
    if "both" in bench_type:
        bench_type = {"dr", "sr"}
    samples = []
    if "dr" in bench_type:
        dr_root = os.path.join(root, "dragbench-dr")
        if os.path.isdir(dr_root):
            for cat in sorted(os.listdir(dr_root)):
                cat_dir = os.path.join(dr_root, cat)
                if not os.path.isdir(cat_dir):
                    continue
                for s in sorted(os.listdir(cat_dir)):
                    sd = os.path.join(cat_dir, s)
                    if os.path.isdir(sd):
                        samples.append(sd)
    if "sr" in bench_type:
        sr_root = os.path.join(root, "dragbench-sr")
        if os.path.isdir(sr_root):
            for s in sorted(os.listdir(sr_root)):
                sd = os.path.join(sr_root, s)
                if os.path.isdir(sd):
                    samples.append(sd)
    return samples


def load_pickle(p):
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, p):
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, p)


def points_to_pairs(points):
    src, tgt = [], []
    for i in range(0, len(points) - 1, 2):
        src.append([float(points[i][0]), float(points[i][1])])
        tgt.append([float(points[i + 1][0]), float(points[i + 1][1])])
    return src, tgt


def pairs_to_points(src, tgt):
    out = []
    for s, t in zip(src, tgt):
        out.append([float(s[0]), float(s[1])])
        out.append([float(t[0]), float(t[1])])
    return out


# ----------------------------- global config --------------------------------
CONFIG = {
    "samples": [],
    "sidecar": "meta_data_multi.pkl",
    "root": "",
}


def sample_paths(idx):
    sd = CONFIG["samples"][idx]
    return {
        "sample_dir": sd,
        "image": os.path.join(sd, "original_image.png"),
        "orig_meta": os.path.join(sd, "meta_data.pkl"),
        "sidecar": os.path.join(sd, CONFIG["sidecar"]),
    }


def load_sample(idx):
    p = sample_paths(idx)
    orig_meta = load_pickle(p["orig_meta"]) or {}
    side_meta = load_pickle(p["sidecar"])
    orig_points = list(orig_meta.get("points", []))
    if side_meta is not None and side_meta.get("points"):
        src, tgt = points_to_pairs(side_meta["points"])
    else:
        src, tgt = [], []
    img = np.asarray(Image.open(p["image"]).convert("RGB"))
    return {
        "idx": int(idx),
        "img": img,
        "orig_meta": orig_meta,
        "orig_points": orig_points,
        "src_pts": list(src),
        "tgt_pts": list(tgt),
        "pending_src": None,
        "dirty": False,
    }


# -------------------------------- rendering ---------------------------------
def _draw_arrow(d, s, t, color, width):
    d.line([tuple(s), tuple(t)], fill=color, width=width)
    dx, dy = t[0] - s[0], t[1] - s[1]
    L = max(1e-6, math.hypot(dx, dy))
    ux, uy = dx / L, dy / L
    size = 10
    lx, ly = -uy, ux
    p1 = (t[0] - ux * size + lx * size * 0.5, t[1] - uy * size + ly * size * 0.5)
    p2 = (t[0] - ux * size - lx * size * 0.5, t[1] - uy * size - ly * size * 0.5)
    d.polygon([tuple(t), p1, p2], fill=color)


def render(state):
    img = Image.fromarray(state["img"]).convert("RGBA").copy()
    d = ImageDraw.Draw(img, "RGBA")

    # reference: original meta points, hollow
    for i in range(0, len(state["orig_points"]) - 1, 2):
        sx, sy = state["orig_points"][i]
        tx, ty = state["orig_points"][i + 1]
        d.ellipse([sx - 9, sy - 9, sx + 9, sy + 9],
                  outline=(255, 90, 90, 180), width=2)
        d.ellipse([tx - 9, ty - 9, tx + 9, ty + 9],
                  outline=(90, 255, 90, 180), width=2)
        d.line([(sx, sy), (tx, ty)], fill=(255, 255, 80, 130), width=1)

    # new pairs, filled
    for i, (s, t) in enumerate(zip(state["src_pts"], state["tgt_pts"])):
        _draw_arrow(d, s, t, color=(255, 230, 40, 235), width=2)
        d.ellipse([s[0] - 7, s[1] - 7, s[0] + 7, s[1] + 7],
                  fill=(255, 0, 0, 255), outline=(255, 255, 255, 255), width=1)
        d.ellipse([t[0] - 7, t[1] - 7, t[0] + 7, t[1] + 7],
                  fill=(0, 255, 0, 255), outline=(255, 255, 255, 255), width=1)
        d.text((s[0] + 8, s[1] - 12), str(i), fill=(255, 255, 255, 255),
               stroke_fill=(0, 0, 0, 255), stroke_width=2)

    if state["pending_src"] is not None:
        sx, sy = state["pending_src"]
        d.line([(sx - 9, sy - 9), (sx + 9, sy + 9)], fill=(255, 0, 0, 255), width=3)
        d.line([(sx - 9, sy + 9), (sx + 9, sy - 9)], fill=(255, 0, 0, 255), width=3)

    return np.asarray(img.convert("RGB"))


def info_text(state):
    p = sample_paths(state["idx"])
    rel = os.path.relpath(p["sample_dir"], CONFIG["root"])
    has = "YES" if os.path.exists(p["sidecar"]) else "NO"
    dirt = " *unsaved*" if state["dirty"] else ""
    nxt = "SRC (red)" if state["pending_src"] is None else "TGT (green)"
    prompt = state["orig_meta"].get("prompt", "")
    return (f"**[{state['idx']+1}/{len(CONFIG['samples'])}]** `{rel}`  \n"
            f"pairs: **{len(state['src_pts'])}**   next click: **{nxt}**   "
            f"sidecar: **{has}**{dirt}  \n"
            f"prompt: _{prompt}_")


# -------------------------------- persistence -------------------------------
def save_if_dirty(state):
    if not state["dirty"]:
        return state
    p = sample_paths(state["idx"])
    if len(state["src_pts"]) != len(state["tgt_pts"]) or len(state["src_pts"]) == 0:
        if os.path.exists(p["sidecar"]):
            os.remove(p["sidecar"])
            print(f"[removed] {p['sidecar']} (no complete pairs)")
        state["dirty"] = False
        return state
    meta = dict(state["orig_meta"]) if state["orig_meta"] else {}
    meta["points"] = pairs_to_points(state["src_pts"], state["tgt_pts"])
    save_pickle(meta, p["sidecar"])
    print(f"[saved] {p['sidecar']} ({len(state['src_pts'])} pairs)")
    state["dirty"] = False
    return state


# -------------------------------- handlers ----------------------------------
def on_click(state, evt: gr.SelectData):
    if evt is None or evt.index is None:
        return state, gr.update(), info_text(state)
    x, y = evt.index[0], evt.index[1]
    if state["pending_src"] is None:
        state["pending_src"] = [float(x), float(y)]
    else:
        state["src_pts"].append(state["pending_src"])
        state["tgt_pts"].append([float(x), float(y)])
        state["pending_src"] = None
        state["dirty"] = True
    return state, render(state), info_text(state)


def on_undo(state):
    if state["pending_src"] is not None:
        state["pending_src"] = None
    elif state["src_pts"]:
        state["src_pts"].pop()
        state["tgt_pts"].pop()
        state["dirty"] = True
    return state, render(state), info_text(state)


def on_clear(state):
    state["src_pts"] = []
    state["tgt_pts"] = []
    state["pending_src"] = None
    state["dirty"] = True
    return state, render(state), info_text(state)


def on_reset(state):
    s, t = points_to_pairs(state["orig_points"])
    state["src_pts"] = list(s)
    state["tgt_pts"] = list(t)
    state["pending_src"] = None
    state["dirty"] = True
    return state, render(state), info_text(state)


def on_save(state):
    state = save_if_dirty(state)
    return state, info_text(state)


def _navigate(state, new_idx, save=True):
    if save:
        state = save_if_dirty(state)
    new_idx = max(0, min(len(CONFIG["samples"]) - 1, int(new_idx)))
    new = load_sample(new_idx)
    return new, render(new), info_text(new)


def on_next(state):
    return _navigate(state, state["idx"] + 1, save=True)


def on_prev(state):
    return _navigate(state, state["idx"] - 1, save=True)


def on_skip(state):
    return _navigate(state, state["idx"] + 1, save=False)


def on_goto(state, idx_val):
    try:
        new_idx = int(idx_val) - 1
    except Exception:
        return state, render(state), info_text(state)
    return _navigate(state, new_idx, save=True)


# ------------------------------- UI builder ---------------------------------
def build_ui(initial):
    with gr.Blocks(title="DragBench Multi-Point Annotator", fill_height=True) as demo:
        state = gr.State(initial)
        info = gr.Markdown(info_text(initial))
        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(
                    value=render(initial),
                    type="numpy",
                    interactive=False,
                    show_label=False,
                    show_download_button=False,
                    height=560,
                )
            with gr.Column(scale=1, min_width=260):
                with gr.Row():
                    btn_prev = gr.Button("◀ Prev (save)", size="sm")
                    btn_next = gr.Button("Next (save) ▶", size="sm", variant="primary")
                btn_save = gr.Button("💾 Save")
                btn_skip = gr.Button("Skip (no save)")
                gr.Markdown("---")
                btn_undo = gr.Button("Undo last")
                btn_clear = gr.Button("Clear new points")
                btn_reset = gr.Button("Reset → copy original points")
                gr.Markdown("---")
                with gr.Row():
                    goto_inp = gr.Number(value=initial["idx"] + 1, label="Go to #",
                                         precision=0, minimum=1,
                                         maximum=len(CONFIG["samples"]))
                    btn_goto = gr.Button("Go")
                gr.Markdown(
                    "**How to annotate**  \n"
                    "• Click on the image: first click = **SRC** (red), next click = **TGT** (green); they auto-pair.  \n"
                    "• Hollow circles = original `meta_data.pkl` points (reference only, never saved).  \n"
                    "• **Next/Prev** auto-saves if there are unsaved changes.  \n"
                    "• **Save with 0 pairs** removes the sidecar for this sample.  \n"
                )

        img.select(on_click, inputs=[state], outputs=[state, img, info])
        btn_undo.click(on_undo,   inputs=[state], outputs=[state, img, info])
        btn_clear.click(on_clear, inputs=[state], outputs=[state, img, info])
        btn_reset.click(on_reset, inputs=[state], outputs=[state, img, info])
        btn_save.click(on_save,   inputs=[state], outputs=[state, info])
        btn_next.click(on_next,   inputs=[state], outputs=[state, img, info])
        btn_prev.click(on_prev,   inputs=[state], outputs=[state, img, info])
        btn_skip.click(on_skip,   inputs=[state], outputs=[state, img, info])
        btn_goto.click(on_goto,   inputs=[state, goto_inp], outputs=[state, img, info])
    return demo


# --------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/mnt/disk1/datasets/DragBench")
    ap.add_argument("--bench_type", type=str, default="both", choices=["dr", "sr", "both"])
    ap.add_argument("--sidecar_name", type=str, default="meta_data_multi.pkl")
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--only_missing", action="store_true",
                    help="skip samples that already have a sidecar")
    ap.add_argument("--categories", type=str, default=None,
                    help="comma-separated substrings; keep a sample if its path contains any")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7878)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    CONFIG["root"] = args.root
    CONFIG["sidecar"] = args.sidecar_name

    samples = list_samples(args.root, [args.bench_type])
    if args.categories:
        keys = [k.strip() for k in args.categories.split(",") if k.strip()]
        samples = [s for s in samples if any(k in s for k in keys)]
    if args.only_missing:
        samples = [s for s in samples
                   if not os.path.exists(os.path.join(s, args.sidecar_name))]
    if not samples:
        print("No samples match filters.")
        sys.exit(0)
    CONFIG["samples"] = samples

    start = max(0, min(args.start_index, len(samples) - 1))
    print(f"Loaded {len(samples)} samples. Starting at {start}.")
    print(f"Sidecar name: {args.sidecar_name}")
    print(f"Launching at http://{args.host}:{args.port}")

    initial = load_sample(start)
    demo = build_ui(initial)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, inbrowser=False)


if __name__ == "__main__":
    main()
