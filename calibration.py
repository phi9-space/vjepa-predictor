"""
v2/calibration.py — Pre-Training Signal Calibration (CNN compressor.md §3).

Two-stage calibration sequence that isolates true thermodynamic movement
from the inherent reconstruction jitter of the V-JEPA backbone.

Stage 1: Sensor Noise Floor (ε)
  1. Stream Ego10k tubelets
  2. Compute optical flow magnitude (RAFT) to find static scenes
  3. Keep the 1,000 most-static tubelets (min-heap by M_flow)
  4. Re-extract winners, run V-JEPA to get latents
  5. ε = P95 of L1 latent differences across consecutive frames

Stage 2: Adaptive Kinetic Gate (τ_kinetic)
  1. Sample 5,000 random tubelets globally
  2. Encode through V-JEPA
  3. Compute Ψ = Σ max(0, ||Z_t - Z_{t-1}||₁ - ε) per tubelet
  4. τ_kinetic = P20 of Ψ distribution

Both stages use the SAME VJEPAEncoder.encode() → guaranteed consistent transform.

Output: calibration.json containing ε, τ_kinetic, and diagnostic statistics.
"""

from __future__ import annotations

import gc
import heapq
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from . import config as cfg
from .vjepa_encoder import VJEPAEncoder

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Ψ Computation (shared between calibration and training)
# ═══════════════════════════════════════════════════════════════════════════

def compute_psi(z: np.ndarray, epsilon: float) -> float:
    """
    Compute continuous signal intensity Ψ for a latent tubelet.

    CNN compressor.md §3 Stage 2:
      Ψ = Σ_{t=1}^{T-1} max(0, mean(|Z_t - Z_{t-1}|) - ε)

    The L1 diff uses mean() reduction (per-element average), not sum().
    This makes ε a per-element-per-frame noise floor comparable across
    tensor shapes. Both ε and Ψ use the same reduction for consistency.

    Args:
        z: [T, 24, 24, 768] float32 numpy array (T = 32 for full tubelet)
        epsilon: sensor noise floor from Stage 1

    Returns:
        Ψ as a float (always ≥ 0)
    """
    # z[1:] - z[:-1] → [T-1, 24, 24, 768]
    diffs = np.abs(z[1:] - z[:-1])

    # Mean L1 over spatial + channel dims per frame pair → [T-1]
    per_frame = diffs.mean(axis=(1, 2, 3))

    # Subtract noise floor, clamp to 0, sum
    psi = float(np.sum(np.maximum(0.0, per_frame - epsilon)))
    return psi


def compute_psi_gpu(z: torch.Tensor, epsilon: float) -> float:
    """
    GPU-vectorized version of compute_psi.

    Numerically equivalent to compute_psi() within FP32 precision (~1e-6).

    Args:
        z: [T, 24, 24, 768] float32 GPU tensor
        epsilon: sensor noise floor

    Returns:
        Ψ as a float
    """
    # z[1:] - z[:-1] → [T-1, 24, 24, 768]
    diffs = torch.abs(z[1:] - z[:-1]).mean(dim=(1, 2, 3))  # [T-1]
    psi = torch.clamp(diffs - epsilon, min=0.0).sum()
    return float(psi.item())


# ═══════════════════════════════════════════════════════════════════════════
# Optical Flow (RAFT) — for finding static tubelets
# ═══════════════════════════════════════════════════════════════════════════

def load_raft(device: str = "cuda:0") -> torch.nn.Module:
    """
    Load RAFT-Small optical flow model for static tubelet detection.

    RAFT-Small is ~1M params, fits in 16GB VRAM alongside V-JEPA.

    Returns:
        RAFT model in eval mode on `device`.
    """
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights)
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    logger.info(
        f"RAFT-Small loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
        f"device={device}"
    )
    return model


def compute_frame_diff_score(frames: np.ndarray) -> float:
    """
    Cheap CPU pre-filter: mean absolute pixel difference across all frame pairs.

    NOTE: This runs entirely on the CPU using numpy. The frames arrive from the
    decoder as numpy arrays in CPU RAM, so NO GPU transfer is involved here.
    It takes ~1ms and eliminates ~80% of tubelets before they ever reach the GPU.

    If this exceeds PRE_FILTER_DIFF_THRESHOLD, the tubelet definitely has motion
    and we skip the expensive RAFT computation.

    Args:
        frames: [T, H, W, 3] uint8 numpy

    Returns:
        Mean absolute pixel difference (0-255 scale)
    """
    # Use a stride-2 sample for speed (every other pair)
    diffs = np.abs(
        frames[2::2].astype(np.float32) - frames[:-2:2].astype(np.float32)
    )
    return float(diffs.mean())


def compute_flow_magnitude(
    raft_model: torch.nn.Module,
    frames: np.ndarray,
    device: str = "cuda:0",
) -> float:
    """
    Compute mean optical flow magnitude M_flow for a tubelet using RAFT.

    CNN compressor.md §3 Stage 1 Step 2:
      M_flow = (1 / (T × H × W)) Σ_t Σ_{x,y} sqrt(u² + v²)

    NOTE on RAFT_RESIZE (320p): This is a practical VRAM limitation, not a RAFT default.
    Computing dense optical flow on 1080p pairs uses massive VRAM. We only need to
    rank tubelets by relative motion ("is this static?"), not get precise vectors.

    NOTE on Transfers: This involves exactly ONE CPU -> GPU transfer of the resized
    frame pair (~1MB). The flow magnitude is computed on GPU, and a single float scalar
    is returned to CPU via .item().

    Processes frame pairs sequentially to control GPU memory.
    Frames are resized to RAFT_RESIZE (320p) before flow computation.

    Args:
        raft_model: pretrained RAFT model in eval mode
        frames: [T, H, W, 3] uint8 numpy RGB
        device: torch device

    Returns:
        M_flow as a float (lower = more static)
    """
    T = frames.shape[0]
    if T < 2:
        return 0.0

    raft_h = raft_w = cfg.RAFT_RESIZE
    total_magnitude = 0.0
    n_pairs = 0

    for t in range(T - 1):
        # Resize frame pair to RAFT resolution
        f1 = cv2.resize(frames[t], (raft_w, raft_h), interpolation=cv2.INTER_LINEAR)
        f2 = cv2.resize(frames[t + 1], (raft_w, raft_h), interpolation=cv2.INTER_LINEAR)

        # [H,W,3] uint8 → [1,3,H,W] float32 normalized to [0,1]
        t1 = torch.from_numpy(f1).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        t2 = torch.from_numpy(f2).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.inference_mode():
            # RAFT returns list of flow predictions; last one is the finest
            flow_list = raft_model(t1, t2)
            flow = flow_list[-1]  # [1, 2, H, W]

        # Flow magnitude: sqrt(u² + v²), clipped to RAFT_FLOW_CLIP
        u = flow[0, 0]  # [H, W]
        v = flow[0, 1]  # [H, W]
        magnitude = torch.sqrt(u ** 2 + v ** 2).clamp(max=cfg.RAFT_FLOW_CLIP)

        total_magnitude += magnitude.mean().item()
        n_pairs += 1

        del t1, t2, flow_list, flow

    return total_magnitude / max(n_pairs, 1)


# Need cv2 for flow computation
import cv2


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Sensor Noise Floor (ε)
# ═══════════════════════════════════════════════════════════════════════════

def _heap_entry(
    m_flow: float,
    uid: int,
    factory_id: str,
    worker_id: str,
    video_index: int,
    tubelet_start: int,
) -> tuple:
    """
    Create a max-heap entry (negated for Python's min-heap).

    We want to keep the LOWEST m_flow tubelets. Python's heapq is a min-heap,
    so we negate m_flow: the entry with the LARGEST m_flow (worst = most motion)
    sits at the top and gets popped when the heap exceeds capacity.
    """
    return (-m_flow, uid, factory_id, worker_id, video_index, tubelet_start)


def run_stage1(
    encoder: VJEPAEncoder,
    tubelet_stream,
    raft_model: Optional[torch.nn.Module] = None,
    target_count: int = cfg.STAGE1_TARGET_STATIC_COUNT,
    device: str = "cuda:0",
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[float, dict]:
    """
    Stage 1: Find the most-static tubelets and compute ε.

    Pass 1 — RAFT scan:
      Stream all tubelets, score each by optical flow magnitude M_flow.
      Keep a max-heap of size `target_count` with the lowest M_flow tubelets.

    Pass 2 — V-JEPA encode:
      Re-extract the winning tubelets, encode through V-JEPA, compute
      ε = P95 of L1 latent differences.

    Args:
        encoder:         VJEPAEncoder instance (for Pass 2). NOTE: This is from
                         the standalone V2 vjepa_encoder.py module, entirely rewritten
                         from scratch without any V1 dependencies.
        tubelet_stream:  generator yielding (frames, meta, start_idx) tuples.
                         Must be restartable (called twice: Pass 1 and Pass 2).
                         `frames` is [64, H, W, 3] uint8 numpy at native resolution.
        raft_model:      pre-loaded RAFT model (auto-loads if None)
        target_count:    number of static tubelets to collect (default 1000)
        device:          torch device
        checkpoint_dir:  where to save heap checkpoint (optional)

    Returns:
        (epsilon, stats_dict)
    """
    logger.info("=" * 60)
    logger.info("  STAGE 1: SENSOR NOISE FLOOR (ε)")
    logger.info("=" * 60)
    logger.info(f"  Target: {target_count} most-static tubelets")
    logger.info(f"  Pre-filter threshold: {cfg.PRE_FILTER_DIFF_THRESHOLD}")

    # ── Pass 1: RAFT scan ──
    should_unload_raft = raft_model is None
    if raft_model is None:
        raft_model = load_raft(device=device)

    heap: List[tuple] = []
    heap_set: Set[tuple] = set()  # dedup by (factory, worker, vid_idx, start)
    uid_counter = 0
    total_scanned = 0
    prefilter_passed = 0
    raft_computed = 0
    t_start = time.time()

    logger.info("\n── Pass 1: RAFT Optical Flow Scan ──")

    for tubelet_frames, meta, start_idx in tubelet_stream():
        total_scanned += 1
        factory_id = meta.get("factory_id", "unknown")
        worker_id = meta.get("worker_id", "unknown")
        video_index = meta.get("video_index", -1)

        # Dedup key
        key = (factory_id, worker_id, video_index, start_idx)
        if key in heap_set:
            continue

        # Stage 1a: Cheap pre-filter (frame differencing on CPU)
        diff_score = compute_frame_diff_score(tubelet_frames)
        if diff_score > cfg.PRE_FILTER_DIFF_THRESHOLD:
            if total_scanned % 100 == 0:
                logger.debug(
                    f"  [{total_scanned}] Skipped (diff={diff_score:.1f} > "
                    f"{cfg.PRE_FILTER_DIFF_THRESHOLD})"
                )
            continue
        prefilter_passed += 1

        # Stage 1b: RAFT optical flow (GPU)
        m_flow = compute_flow_magnitude(
            raft_model, tubelet_frames, device=device,
        )
        raft_computed += 1

        # Add to max-heap (negated m_flow)
        entry = _heap_entry(
            m_flow, uid_counter, factory_id, worker_id, video_index, start_idx,
        )
        uid_counter += 1

        if len(heap) < target_count:
            heapq.heappush(heap, entry)
            heap_set.add(key)
        else:
            popped = heapq.heappushpop(heap, entry)
            if popped is not entry:
                popped_key = (popped[2], popped[3], popped[4], popped[5])
                heap_set.discard(popped_key)
                heap_set.add(key)

        # Progress logging
        if raft_computed % 50 == 0:
            elapsed = time.time() - t_start
            worst_mflow = -heap[0][0] if heap else float("inf")
            logger.info(
                f"  [{elapsed:.0f}s] Scanned: {total_scanned} | "
                f"Pre-filter: {prefilter_passed} | RAFT: {raft_computed} | "
                f"Heap: {len(heap)}/{target_count} (worst M_flow={worst_mflow:.4f})"
            )

    elapsed_pass1 = time.time() - t_start
    logger.info(f"\n  Pass 1 complete in {elapsed_pass1:.0f}s:")
    logger.info(f"    Total scanned: {total_scanned}")
    logger.info(f"    Pre-filter passed: {prefilter_passed} ({100*prefilter_passed/max(1,total_scanned):.1f}%)")
    logger.info(f"    RAFT computed: {raft_computed}")
    logger.info(f"    Heap size: {len(heap)}")

    # Unload RAFT before loading V-JEPA (save VRAM)
    if should_unload_raft:
        logger.info("  Unloading RAFT...")
        raft_model.cpu()
        del raft_model
        gc.collect()

    # ── Extract sorted winning tubelets ──
    winners = []
    for entry in sorted(heap, key=lambda e: -e[0]):
        m_flow, uid, fid, wid, vidx, start = entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]
        winners.append({
            "m_flow": -m_flow,  # un-negate
            "factory_id": fid,
            "worker_id": wid,
            "video_index": vidx,
            "tubelet_start": start,
        })

    if len(winners) < cfg.MIN_TUBELETS_FOR_STABLE_EPSILON:
        raise RuntimeError(
            f"Only {len(winners)} static tubelets found (need ≥{cfg.MIN_TUBELETS_FOR_STABLE_EPSILON}). "
            f"Increase data scope or lower PRE_FILTER_DIFF_THRESHOLD."
        )

    if winners:
        logger.info(f"    M_flow range: [{winners[0]['m_flow']:.6f}, {winners[-1]['m_flow']:.6f}]")

    # Save heap checkpoint
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        heap_path = checkpoint_dir / "stage1_heap.json"
        with open(heap_path, "w") as f:
            json.dump({"winners": winners, "count": len(winners)}, f, indent=2)
        logger.info(f"  Heap checkpoint saved: {heap_path}")

    # ── Pass 2: V-JEPA encode winning tubelets ──
    logger.info("\n── Pass 2: V-JEPA Latent Extraction ──")
    logger.info(f"  Re-extracting {len(winners)} winning tubelets...")

    # Build lookup set for fast matching
    target_set: Dict[tuple, int] = {}
    for i, w in enumerate(winners):
        key = (w["factory_id"], w["worker_id"], w["video_index"], w["tubelet_start"])
        target_set[key] = i

    latents_list: List[Optional[np.ndarray]] = [None] * len(winners)
    found_count = 0
    t_start2 = time.time()

    for tubelet_frames, meta, start_idx in tubelet_stream():
        factory_id = meta.get("factory_id", "unknown")
        worker_id = meta.get("worker_id", "unknown")
        video_index = meta.get("video_index", -1)

        key = (factory_id, worker_id, video_index, start_idx)
        if key not in target_set:
            continue

        idx = target_set[key]
        if latents_list[idx] is not None:
            continue

        # Encode through V-JEPA (THE canonical transform is inside encode())
        z = encoder.encode(tubelet_frames, return_on_gpu=False)
        # z: [32, 24, 24, 768] float32 on CPU
        latents_list[idx] = z.numpy()
        found_count += 1

        if found_count % 50 == 0:
            elapsed = time.time() - t_start2
            logger.info(f"  [{elapsed:.0f}s] Encoded {found_count}/{len(winners)} tubelets")

        if found_count >= len(winners):
            break

    elapsed_pass2 = time.time() - t_start2
    logger.info(f"  Encoded {found_count}/{len(winners)} tubelets in {elapsed_pass2:.0f}s")

    missing = sum(1 for l in latents_list if l is None)
    if missing > 0:
        logger.warning(f"  {missing} tubelets could not be re-extracted!")

    # Filter out missing
    valid_latents = [l for l in latents_list if l is not None]

    if len(valid_latents) < cfg.MIN_TUBELETS_FOR_STABLE_EPSILON:
        raise RuntimeError(
            f"Only {len(valid_latents)} tubelets encoded (need ≥{cfg.MIN_TUBELETS_FOR_STABLE_EPSILON})"
        )

    # ── Compute ε ──
    logger.info("\n── Computing ε ──")
    epsilon, stats = _compute_epsilon(valid_latents)

    # Add pass stats to output
    stats["pass1_total_scanned"] = total_scanned
    stats["pass1_prefilter_passed"] = prefilter_passed
    stats["pass1_raft_computed"] = raft_computed
    stats["pass1_elapsed_s"] = elapsed_pass1
    stats["pass2_found"] = found_count
    stats["pass2_missing"] = missing
    stats["pass2_elapsed_s"] = elapsed_pass2

    return epsilon, stats


def _compute_epsilon(
    latents_list: List[np.ndarray],
    percentile: float = cfg.STAGE1_PERCENTILE,
) -> Tuple[float, dict]:
    """
    Compute ε = P95 of L1 latent differences across static tubelets.

    CNN compressor.md §3 Stage 1 Step 5:
      ε = P95({||Z_t - Z_{t-1}||₁}_{inert_anchor})

    NOTE on computation: This happens entirely on the CPU. During Pass 2, latents
    are extracted by V-JEPA on the GPU, and then copied back to CPU RAM.
    Since we only do this for the 1,000 most-static tubelets, it's a small memory
    footprint and allows us to compute the global P95 over all 1,000 tubelets
    at once without blowing up VRAM.

    Uses mean() reduction (per-element average) for consistency with Ψ.

    Args:
        latents_list: list of [32, 24, 24, 768] float32 numpy arrays
        percentile: default 95.0

    Returns:
        (epsilon, stats_dict)
    """
    all_diffs = []

    for latents in latents_list:
        # latents: [32, 24, 24, 768]
        # Consecutive frame diffs: [31, 24, 24, 768]
        diffs = np.abs(latents[1:] - latents[:-1])

        # Mean L1 over spatial + channel dims per frame pair → [31]
        frame_diffs = diffs.mean(axis=(1, 2, 3))
        all_diffs.extend(frame_diffs.tolist())

    all_diffs = np.array(all_diffs)
    epsilon = float(np.percentile(all_diffs, percentile))

    stats = {
        "n_tubelets": len(latents_list),
        "n_frame_pairs": len(all_diffs),
        "mean_diff": float(np.mean(all_diffs)),
        "median_diff": float(np.median(all_diffs)),
        "std_diff": float(np.std(all_diffs)),
        "min_diff": float(np.min(all_diffs)),
        "max_diff": float(np.max(all_diffs)),
        "p50": float(np.percentile(all_diffs, 50)),
        "p90": float(np.percentile(all_diffs, 90)),
        "p95": epsilon,
        "p99": float(np.percentile(all_diffs, 99)),
    }

    logger.info(f"  Sensor noise floor ε computed:")
    logger.info(f"    Tubelets: {stats['n_tubelets']}, Frame pairs: {stats['n_frame_pairs']}")
    logger.info(f"    L1 diff distribution:")
    logger.info(f"      Mean={stats['mean_diff']:.6f}  Median={stats['median_diff']:.6f}  Std={stats['std_diff']:.6f}")
    logger.info(f"      P50={stats['p50']:.6f}  P90={stats['p90']:.6f}  P95={epsilon:.6f}  P99={stats['p99']:.6f}")
    logger.info(f"      Min={stats['min_diff']:.6f}  Max={stats['max_diff']:.6f}")
    logger.info(f"    → ε = {epsilon:.6f}")

    return epsilon, stats


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Adaptive Kinetic Gate (τ_kinetic)
# ═══════════════════════════════════════════════════════════════════════════

def run_stage2(
    encoder: VJEPAEncoder,
    tubelet_stream,
    epsilon: float,
    target_count: int = cfg.STAGE2_TARGET_SAMPLE_COUNT,
    device: str = "cuda:0",
) -> Tuple[float, dict]:
    """
    Stage 2: Compute τ_kinetic from the Ψ distribution.

    Sample `target_count` unconstrained tubelets, encode through V-JEPA,
    compute Ψ for each, and set τ_kinetic = P20.

    This drops the lowest-energy 20th percentile while preserving subtle
    continuous physics (CNN compressor.md §3 Stage 2).

    Args:
        encoder:        VJEPAEncoder instance
        tubelet_stream: generator factory yielding (frames, meta, start_idx)
        epsilon:        sensor noise floor from Stage 1
        target_count:   number of tubelets to sample (default 5000)
        device:         torch device

    Returns:
        (tau_kinetic, stats_dict)
    """
    logger.info("=" * 60)
    logger.info("  STAGE 2: ADAPTIVE KINETIC GATE (τ_kinetic)")
    logger.info("=" * 60)
    logger.info(f"  Target: {target_count} random tubelets")
    logger.info(f"  ε = {epsilon:.6f}")

    psi_values = []
    n_encoded = 0
    t_start = time.time()

    for tubelet_frames, meta, start_idx in tubelet_stream():
        if n_encoded >= target_count:
            break

        # Encode through V-JEPA
        z = encoder.encode(tubelet_frames, return_on_gpu=True)
        # z: [32, 24, 24, 768] float32 on GPU

        # Compute Ψ on GPU (avoids GPU→CPU transfer)
        psi = compute_psi_gpu(z, epsilon)
        psi_values.append(psi)
        n_encoded += 1

        del z

        # Verbose logging
        if n_encoded % 100 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"  [{elapsed:.0f}s] Encoded {n_encoded}/{target_count} | "
                f"Ψ range: [{min(psi_values):.4f}, {max(psi_values):.4f}] | "
                f"Ψ mean: {np.mean(psi_values):.4f}"
            )

    elapsed = time.time() - t_start

    if n_encoded < 10:
        raise RuntimeError(
            f"Only {n_encoded} tubelets encoded for Stage 2 (need ≥10). "
            f"Check data source."
        )

    psi_arr = np.array(psi_values)
    tau_kinetic = float(np.percentile(psi_arr, cfg.STAGE2_PERCENTILE))

    stats = {
        "n_tubelets": n_encoded,
        "psi_mean": float(np.mean(psi_arr)),
        "psi_std": float(np.std(psi_arr)),
        "psi_min": float(np.min(psi_arr)),
        "psi_max": float(np.max(psi_arr)),
        "psi_p10": float(np.percentile(psi_arr, 10)),
        "psi_p20": tau_kinetic,
        "psi_p50": float(np.percentile(psi_arr, 50)),
        "psi_p80": float(np.percentile(psi_arr, 80)),
        "psi_p90": float(np.percentile(psi_arr, 90)),
        "n_below_tau": int(np.sum(psi_arr < tau_kinetic)),
        "n_above_tau": int(np.sum(psi_arr >= tau_kinetic)),
        "elapsed_s": elapsed,
    }

    logger.info(f"\n  τ_kinetic computed:")
    logger.info(f"    Tubelets sampled: {n_encoded}")
    logger.info(f"    Ψ distribution:")
    logger.info(f"      Mean={stats['psi_mean']:.6f}  Std={stats['psi_std']:.6f}")
    logger.info(f"      P10={stats['psi_p10']:.6f}  P20={tau_kinetic:.6f}  P50={stats['psi_p50']:.6f}")
    logger.info(f"      P80={stats['psi_p80']:.6f}  P90={stats['psi_p90']:.6f}")
    logger.info(f"      Min={stats['psi_min']:.6f}  Max={stats['psi_max']:.6f}")
    logger.info(f"    Below τ: {stats['n_below_tau']} ({100*stats['n_below_tau']/n_encoded:.1f}%)")
    logger.info(f"    Above τ: {stats['n_above_tau']} ({100*stats['n_above_tau']/n_encoded:.1f}%)")
    logger.info(f"    → τ_kinetic = {tau_kinetic:.6f}")

    return tau_kinetic, stats


# ═══════════════════════════════════════════════════════════════════════════
# Full Calibration Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_calibration(
    tubelet_stream_factory,
    device: str = "cuda:0",
    stage1_target: int = cfg.STAGE1_TARGET_STATIC_COUNT,
    stage2_target: int = cfg.STAGE2_TARGET_SAMPLE_COUNT,
    output_dir: Optional[Path] = None,
) -> Tuple[float, float, dict]:
    """
    Run full calibration pipeline: Stage 1 (ε) → Stage 2 (τ_kinetic).

    Args:
        tubelet_stream_factory: A callable (e.g., a lambda or function) that returns
                                a fresh generator/iterator yielding (frames, meta, start_idx).
                                Why a factory? Because Stage 1 needs to stream the entire
                                dataset TWICE (Pass 1 for RAFT, Pass 2 for V-JEPA).
                                Python generators get exhausted after one pass, so we need
                                a factory to create a fresh stream for Pass 2.
                                `frames` is [64, H, W, 3] uint8 at native/scaled resolution.
        device:                 torch device
        stage1_target:          number of static tubelets for ε
        stage2_target:          number of random tubelets for τ_kinetic
        output_dir:             where to save calibration.json (default: config)

    Returns:
        (epsilon, tau_kinetic, full_stats_dict)
    """
    if output_dir is None:
        output_dir = cfg.CALIBRATION_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 60)
    logger.info("  CALIBRATION PIPELINE (V2)")
    logger.info("=" * 60)
    logger.info(f"  Device: {device}")
    logger.info(f"  Stage 1 target: {stage1_target} static tubelets")
    logger.info(f"  Stage 2 target: {stage2_target} random tubelets")
    logger.info(f"  Output: {output_dir}")

    t_total_start = time.time()

    # Load V-JEPA encoder (FP32 for training-phase calibration)
    encoder = VJEPAEncoder(device=device, fp32=True)

    # ── Stage 1 ──
    epsilon, stage1_stats = run_stage1(
        encoder=encoder,
        tubelet_stream=tubelet_stream_factory,
        target_count=stage1_target,
        device=device,
        checkpoint_dir=output_dir / "checkpoints",
    )

    # ── Stage 2 ──
    tau_kinetic, stage2_stats = run_stage2(
        encoder=encoder,
        tubelet_stream=tubelet_stream_factory,
        epsilon=epsilon,
        target_count=stage2_target,
        device=device,
    )

    # ── Save results ──
    elapsed_total = time.time() - t_total_start

    result = {
        "epsilon": epsilon,
        "tau_kinetic": tau_kinetic,
        "stage1": stage1_stats,
        "stage2": stage2_stats,
        "total_elapsed_s": elapsed_total,
        "config": {
            "stage1_target": stage1_target,
            "stage1_percentile": cfg.STAGE1_PERCENTILE,
            "stage2_target": stage2_target,
            "stage2_percentile": cfg.STAGE2_PERCENTILE,
            "pre_filter_threshold": cfg.PRE_FILTER_DIFF_THRESHOLD,
            "raft_resize": cfg.RAFT_RESIZE,
            "image_size": cfg.IMAGE_SIZE,
            "shorter_side_size": cfg.SHORTER_SIDE_SIZE,
            "target_fps": cfg.TARGET_FPS,
        },
    }

    output_path = output_dir / "calibration.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("  CALIBRATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  ε = {epsilon:.6f}")
    logger.info(f"  τ_kinetic = {tau_kinetic:.6f}")
    logger.info(f"  Total time: {elapsed_total:.0f}s")
    logger.info(f"  Saved to: {output_path}")
    logger.info("=" * 60)

    return epsilon, tau_kinetic, result
