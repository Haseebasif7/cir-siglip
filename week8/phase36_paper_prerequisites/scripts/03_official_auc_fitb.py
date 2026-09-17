"""
Phase 36, step 3: the OFFICIAL, released Polyvore Outfits (nondisjoint) test
protocols -- compatibility AUC (compatibility_test.txt, 20,000 lines) and
fill-in-the-blank accuracy (fill_in_blank_test.json, 10,000 questions) -- for
the matched-condition systems Claim 1 rests on, which had never been scored on
them. Same 10,000 test outfits the CIR benchmark was built from (verified during
planning), so this is a second protocol over the same released split, not new
data exposure.

Two protocols, pre-declared in phase36_notes.md:
  PRIMARY  "CIR-native leave-one-out": each model scored through its own
           complementary-retrieval function s(context, candidate).
           AUC score(outfit) = mean_i s(outfit \ {i}, item_i);  FITB = argmax_c s(question, c).
  SECONDARY phase 9's exact protocol (mean pairwise cosine of unconditioned item
           embeddings; FITB = mean cosine to the question items). Only defined
           for models with an unconditioned per-item embedding (ours, OT, raw).

Guards run FIRST: phase 9 Model A must reproduce 0.9469 / 0.7031 and raw
image-only SigLIP 0.7172 / 0.4843 under the secondary protocol (these validate the
re-typed protocol code). Ties in FITB are counted and reported.
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

GUARDS = {"raw_siglip_image_only": (0.7172, 0.4843), "phase9_model_a": (0.9469, 0.7031)}
PUBLISHED = [
    ("Vasileva et al. 2018, type-aware embeddings (ResNet-18 fine-tuned)", 0.88, 0.576,
     "ECCV'18; nondisjoint; already cited in week3/phase9 results_table.md"),
    ("CSA-Net, Lin et al. 2020 (ResNet-18 fine-tuned end-to-end, no text)", 0.91, 0.6373,
     "arXiv 1912.08967, Table 1, Polyvore Outfits (nondisjoint)"),
    ("OutfitTransformer, Sarkar et al. 2023 (ResNet-18 fine-tuned + SentenceBERT fc, text)", 0.93, 0.6710,
     "arXiv 2204.04812, Tables 1-2, nondisjoint; AUC 0.92 without text"),
]
SYSTEMS = [
    ("ours_solo", "ours", [C.PHASE27_MODELS / "text_only.pt"]),
    ("ours_ens", "ours", [C.PHASE28_MODELS / f"text_ensemble_seed{s}.pt" for s in [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    ("ot_solo", "ot", [C.PHASE32_DIR / "models" / "ot32_seed42.pt"]),
    ("ot_ens", "ot", [C.PHASE32_DIR / "models" / f"ot32_seed{s}.pt" for s in [42, 1, 2]]),
    ("csa_ens", "csa", [C.PHASE34_DIR / "models" / f"csanet34_seed{s}.pt" for s in [42, 1, 2]]),
]


def main():
    t_start = time.perf_counter()
    item_ids, idx, image_emb, base_repr = C.load_item_data()
    resolver = C.build_setid_index_resolver("test")
    compat_rows, sk_c = C.load_compatibility_lines(resolver, idx)
    fitb_rows, sk_f = C.load_fitb_questions(resolver, idx)
    labels = np.array([lab for lab, _ in compat_rows])
    print(f"compat lines {len(compat_rows)} (skipped {sk_c}; pos {int(labels.sum())}); FITB {len(fitb_rows)} (skipped {sk_f})")
    assert sk_c == 0 and sk_f == 0
    cats, cat_to_idx, item_cat = C.load_category_data()

    res = {"device": C.DEVICE, "n_compat": len(compat_rows), "n_compat_pos": int(labels.sum()), "n_fitb": len(fitb_rows),
           "skipped": {"compat": sk_c, "fitb": sk_f}, "guards": {}, "secondary": {}, "primary": {},
           "published_nondisjoint": [{"label": a, "auc": b, "fitb": c, "source": d} for a, b, c, d in PUBLISHED]}

    # ---------------- GUARDS (secondary protocol, single embedding matrix)
    def secondary(emb_list):
        auc, n = C.evaluate_compatibility_pairwise(emb_list, idx, compat_rows)
        acc, n2, ties = C.evaluate_fitb_pairwise(emb_list, idx, fitb_rows)
        return {"auc": float(auc), "fitb": float(acc), "fitb_ties": int(ties)}

    g1 = secondary([image_emb])
    m9 = C.ProjectionHeadPhase9().to(C.DEVICE).eval()
    m9.load_state_dict(torch.load(C.PHASE9_MODEL_A, map_location=C.DEVICE))
    with torch.no_grad():
        p9 = m9(torch.tensor(image_emb, device=C.DEVICE)).cpu().numpy()
    p9 = p9 / np.linalg.norm(p9, axis=1, keepdims=True)
    g2 = secondary([p9])
    del m9
    ok1 = (round(g1["auc"], 4), round(g1["fitb"], 4)) == GUARDS["raw_siglip_image_only"]
    ok2 = (round(g2["auc"], 4), round(g2["fitb"], 4)) == GUARDS["phase9_model_a"]
    res["guards"] = {"raw_siglip_image_only": {**g1, "expected": GUARDS["raw_siglip_image_only"], "ok": ok1},
                     "phase9_model_a": {**g2, "expected": GUARDS["phase9_model_a"], "ok": ok2}}
    print(f"GUARD raw image-only SigLIP: AUC={g1['auc']:.4f} FITB={g1['fitb']:.4f} expected {GUARDS['raw_siglip_image_only']} ok={ok1}")
    print(f"GUARD phase 9 Model A      : AUC={g2['auc']:.4f} FITB={g2['fitb']:.4f} expected {GUARDS['phase9_model_a']} ok={ok2}")
    if not (ok1 and ok2):
        with open(C.DATA_DIR / "official_benchmark.json", "w") as f:
            json.dump(res, f, indent=2)
        print("!!! GUARD FAILURE -- protocol copy is not faithful; stopping before any new number is written")
        sys.exit(2)

    # ---------------- build the primary-protocol task lists once
    loo_row, loo_ctx, loo_ctx_cat, loo_cand, loo_cand_cat = [], [], [], [], []
    for r, (_, items) in enumerate(compat_rows):
        g = [idx[i] for i in items]
        c = [cat_to_idx[item_cat[i]] for i in items]
        for i in range(len(items)):
            ctx = g[:i] + g[i + 1:]
            if not ctx:
                continue
            loo_row.append(r); loo_ctx.append(ctx); loo_ctx_cat.append(c[:i] + c[i + 1:])
            loo_cand.append([g[i]]); loo_cand_cat.append([c[i]])
    loo_row = np.array(loo_row)
    fitb_ctx = [[idx[i] for i in q] for q, _ in fitb_rows]
    fitb_ctx_cat = [[cat_to_idx[item_cat[i]] for i in q] for q, _ in fitb_rows]
    fitb_cand = [[idx[a] for a in ans] for _, ans in fitb_rows]
    fitb_cand_cat = [[cat_to_idx[item_cat[a]] for a in ans] for _, ans in fitb_rows]
    same_cat_answers = int(sum(len(set(cc)) == 1 for cc in fitb_cand_cat))
    res["n_loo_tasks"] = int(len(loo_row))
    res["fitb_questions_with_all_same_category_answers"] = same_cat_answers
    print(f"LOO tasks {len(loo_row)}; FITB questions whose 4 answers share one category: {same_cat_answers}/{len(fitb_rows)}")

    def primary(system):
        t0 = time.perf_counter()
        s = system.score_sets(loo_ctx, loo_cand, ctx_cat_lists=loo_ctx_cat, cand_cat_lists=loo_cand_cat, base_repr=base_repr)[:, 0]
        row_sum = np.zeros(len(compat_rows)); row_cnt = np.zeros(len(compat_rows))
        np.add.at(row_sum, loo_row, s); np.add.at(row_cnt, loo_row, 1.0)
        auc = C.auc_rank_sum(row_sum / row_cnt, labels)
        sf = system.score_sets(fitb_ctx, fitb_cand, ctx_cat_lists=fitb_ctx_cat, cand_cat_lists=fitb_cand_cat, base_repr=base_repr)
        pred = sf.argmax(axis=1)  # first max = lowest index among ties, same rule as phase 9's strict `>`
        ties = int(((sf == sf.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())
        # per-line / per-question arrays saved for the paired bootstrap in 03b
        np.savez_compressed(C.DATA_DIR / f"official_primary_{system.name}.npz",
                            compat_scores=(row_sum / row_cnt).astype(np.float64), labels=labels.astype(np.int8),
                            fitb_pred=pred.astype(np.int8), fitb_correct=(pred == 0))
        return {"auc": float(auc), "fitb": float((pred == 0).mean()), "fitb_ties": ties, "seconds": time.perf_counter() - t0}

    # ---------------- untrained references
    raw_img = C.OursSystem("raw_siglip_image_only", []); raw_img.proj_list = [image_emb]
    raw_it = C.OursSystem("raw_siglip_image_text", []); raw_it.proj_list = [base_repr.cpu().numpy()]
    res["secondary"]["raw_siglip_image_only"] = g1
    res["secondary"]["raw_siglip_image_text"] = secondary(raw_it.proj_list)
    res["secondary"]["phase9_model_a"] = g2
    res["primary"]["raw_siglip_image_only"] = primary(raw_img)
    res["primary"]["raw_siglip_image_text"] = primary(raw_it)
    p9sys = C.OursSystem("phase9_model_a", []); p9sys.proj_list = [p9]
    res["primary"]["phase9_model_a"] = primary(p9sys)
    for k in ["raw_siglip_image_only", "raw_siglip_image_text", "phase9_model_a"]:
        print(f"{k}: secondary AUC={res['secondary'][k]['auc']:.4f} FITB={res['secondary'][k]['fitb']:.4f} | "
              f"primary AUC={res['primary'][k]['auc']:.4f} FITB={res['primary'][k]['fitb']:.4f}")
    del raw_img, raw_it, p9sys, p9

    # ---------------- the five matched-condition systems
    for name, kind, ckpts in SYSTEMS:
        print(f"\n=== {name} ===")
        if kind == "ours":
            system = C.OursSystem(name, ckpts)
        elif kind == "ot":
            system = C.OTSystem(name, ckpts)
        else:
            system = C.CSASystem(name, ckpts, cat_to_idx, item_cat)
        system.precompute(base_repr)
        if kind in ("ours", "ot"):
            res["secondary"][name] = secondary(system.item_embeddings())
            print(f"  secondary: AUC={res['secondary'][name]['auc']:.4f} FITB={res['secondary'][name]['fitb']:.4f} ties={res['secondary'][name]['fitb_ties']}")
        else:
            res["secondary"][name] = None
        res["primary"][name] = primary(system)
        print(f"  primary  : AUC={res['primary'][name]['auc']:.4f} FITB={res['primary'][name]['fitb']:.4f} ties={res['primary'][name]['fitb_ties']}  ({res['primary'][name]['seconds']:.0f}s)")
        del system
        gc.collect()
        if C.DEVICE == "mps":
            torch.mps.empty_cache()

    res["wall_time_s"] = time.perf_counter() - t_start
    with open(C.DATA_DIR / "official_benchmark.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---------------- markdown
    def row(label, e):
        return f"| {label} | {e['auc']:.4f} | {100*e['fitb']:.2f}% | {e['fitb_ties']} |"
    L = ["# Phase 36: Official Polyvore Outfits Test Split -- Compatibility AUC and FITB", "",
         f"Released files: `nondisjoint/compatibility_test.txt` ({res['n_compat']} lines, {res['n_compat_pos']} positive) and "
         f"`nondisjoint/fill_in_blank_test.json` ({res['n_fitb']} questions; `answers[0]` is the correct candidate). "
         f"0 lines/questions skipped. These reference the same 10,000 `test.json` outfits the CIR benchmark is built from.", "",
         "## Guards (secondary protocol must reproduce phase 9's recorded numbers before anything new is trusted)", "",
         "| Guard | AUC | FITB | expected (AUC, FITB) | ok |", "|---|---|---|---|---|"]
    for k, e in res["guards"].items():
        L.append(f"| {k} | {e['auc']:.4f} | {e['fitb']:.4f} | {e['expected']} | {'yes' if e['ok'] else 'NO'} |")
    L += ["", "## Primary protocol: CIR-native leave-one-out (each model scored through its own retrieval function)", "",
          f"AUC score of an outfit = mean over its items of s(outfit minus item, item); FITB = argmax over the 4 candidates of "
          f"s(question, candidate). {res['n_loo_tasks']} leave-one-out tasks. For CSA-Net the target category is the candidate's own "
          f"category ({same_cat_answers}/{res['n_fitb']} FITB questions have all four answers in one category). Ensembles average "
          f"member scores, never embeddings. Ties = FITB questions whose top score is shared (resolved to the lower index, as in phase 9).", "",
          "| System | AUC | FITB acc. | FITB ties |", "|---|---|---|---|",
          row("Raw SigLIP, image only (untrained)", res["primary"]["raw_siglip_image_only"]),
          row("Raw SigLIP, image + text (untrained, 1536-d)", res["primary"]["raw_siglip_image_text"]),
          row("Phase 9 Model A (image-only projection, original)", res["primary"]["phase9_model_a"])]
    for name, _, _ in SYSTEMS:
        L.append(row(f"`{name}` -- {C.SYSTEM_LABELS[name]}", res["primary"][name]))
    L += ["", "**Published, same released nondisjoint split, different (fine-tuned) backbone -- not matched-condition:**", "",
          "| Published system | AUC | FITB acc. | Source |", "|---|---|---|---|"]
    for a, b, c, d in PUBLISHED:
        L.append(f"| {a} | {b:.2f} | {100*c:.2f}% | {d} |")
    L += ["", "## Secondary protocol: phase 9's exact protocol (mean pairwise cosine of unconditioned item embeddings)", "",
          "Not defined for CSA-Net (every CSA-Net item embedding is conditioned on a category pair). Reported to link to the "
          "numbers this project has cited since phase 9 and to show sensitivity to the aggregation choice.", "",
          "| System | AUC | FITB acc. | FITB ties |", "|---|---|---|---|",
          row("Raw SigLIP, image only (guard)", res["secondary"]["raw_siglip_image_only"]),
          row("Raw SigLIP, image + text (untrained, 1536-d)", res["secondary"]["raw_siglip_image_text"]),
          row("Phase 9 Model A (guard)", res["secondary"]["phase9_model_a"])]
    for name, kind, _ in SYSTEMS:
        if kind in ("ours", "ot"):
            L.append(row(f"`{name}`", res["secondary"][name]))
    L.append("| `csa_ens` | n/a | n/a | -- |")
    L += ["", "Interpretation is written in `phase36_notes.md` after reading these tables.", ""]
    (C.PHASE_DIR / "official_benchmark.md").write_text("\n".join(L))
    print(f"\nwall {res['wall_time_s']:.0f}s; saved official_benchmark.md and data/official_benchmark.json")


if __name__ == "__main__":
    main()
