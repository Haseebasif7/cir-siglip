"""
Phase 33: Modal GPU training app for the tuning grid (step 3) and
ensembling (step 4) -- the two steps whose cost profile actually benefits
from Modal's parallelism (.map() across up to 5 concurrent containers),
unlike steps 1-2 (a handful of isolated measurement runs, done locally for
free -- see checkpoint_selection_check.md, text_input_integration.md).

Reuses this phase's own local train_core.py/model.py UNCHANGED (mounted via
add_local_dir, not reimplemented inline the way phase 31/32's much smaller
model.py was -- this phase's vectorized train_core.py is too large to
duplicate safely). Own app ("phase33-csanet-parity"), own volume upload of
the two CSA-Net-specific data files phase 31/32's shared
"phase27-text-category-data" volume doesn't have yet
(week4/phase13_csa_net_baseline/data/training_data.json -- already there
from phase 31 -- and negative_candidates.json, CSA-Net's own mined-negative
file, NOT yet uploaded since phase 31 used random-mode negatives for
OutfitTransformer and never needed it).
"""
from pathlib import Path

import modal

SCRIPTS_DIR = Path(__file__).resolve().parent

app = modal.App("phase33-csanet-parity")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    .add_local_dir(str(SCRIPTS_DIR), remote_path="/root/scripts")
)
volume = modal.Volume.from_name("phase27-text-category-data", create_if_missing=False)

DATA_DIR = "/data"

DEFAULT_CONFIG = {
    "name": "default",
    "input_mode": "image_text",   # step 2 confirmed text helps; tuning/ensembling build on that winner
    "selection_metric": "recall10",  # step 1's adopted fix
    "lr": 5e-5, "batch_size": 96, "patience": 5,   # phase 13b's originals; patience unchanged, see step 1
    "max_epochs": 40, "min_delta": 0.0005,
    "num_negatives": 10, "uniformity_weight": 1.0, "margin": 0.3,
    "seed": 42,
    "save_checkpoint": False,
}


@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=7200, max_containers=5)
def train_one(config: dict) -> dict:
    import json
    import sys
    import time
    from pathlib import Path as P

    import torch

    sys.path.insert(0, "/root/scripts")
    import train_core  # noqa: E402 -- must come after sys.path insert

    cfg = {**DEFAULT_CONFIG, **config}
    assert cfg["name"].startswith("ot33_"), "every phase33 config name must be prefixed ot33_"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = P(DATA_DIR)

    # --- Credit-safety, unchanged pattern from phase 27-32 ---
    status_path = data_dir / f"status_{cfg['name']}.json"
    ckpt_path = data_dir / f"checkpoint_{cfg['name']}.pt"
    if status_path.exists():
        with open(status_path) as f:
            prior_status = json.load(f)
        if prior_status.get("status") == "done":
            print(f"{cfg['name']}: already done (val_recall10={prior_status.get('best_recall10')}), skipping.")
            return prior_status["result"]

    text_npz = data_dir / "text_embeddings.npz" if cfg["input_mode"] == "image_text" else None
    catalog = train_core.load_catalog(
        data_dir / "siglip_base.npz", data_dir / "training_data.json", data_dir / "negative_candidates.json",
        text_embeddings_npz=text_npz, device=device,
    )
    with open(data_dir / "cir_val_benchmark.json") as f:
        bench = json.load(f)
    val_benchmark = (bench["pools"], bench["queries"])

    t0 = time.time()
    result = train_core.run_training(
        catalog, device, max_epochs=cfg["max_epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        patience=cfg["patience"], margin=cfg["margin"], uniformity_weight=cfg["uniformity_weight"],
        num_negatives=cfg["num_negatives"], seed=cfg["seed"], min_delta=cfg["min_delta"],
        selection_metric=cfg["selection_metric"], val_benchmark=val_benchmark, log_every=1000,
    )

    out = {
        "config": cfg, "curve": result["curve"], "best_epoch": result["best_epoch"],
        "best_recall10": result["best_recall10"], "n_epochs_run": result["n_epochs_run"],
        "wall_time_sec": time.time() - t0, "n_params": result["n_params"], "device": device,
    }

    if cfg.get("save_checkpoint") and result["best_state"] is not None:
        torch.save(result["best_state"], ckpt_path)
        out["checkpoint_path"] = str(ckpt_path)
        with open(status_path, "w") as f:
            json.dump({"status": "done", "best_epoch": out["best_epoch"],
                        "best_recall10": out["best_recall10"], "result": out}, f)
        volume.commit()

    return out


@app.local_entrypoint()
def smoke_test():
    """`modal run scripts/modal_app.py::smoke_test` -- confirms the mounted
    train_core.py imports and runs correctly on Modal before any real sweep."""
    cfg = {"name": "ot33_smoke", "max_epochs": 2, "patience": 5, "save_checkpoint": False}
    result = train_one.remote(cfg)
    import json
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, indent=2))
    print("curve:")
    for row in result["curve"]:
        print(" ", row)
