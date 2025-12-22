# -*- coding: utf-8 -*-
"""
Django views.py — OptoGPT Inference API
--------------------------------------
- 端点:  POST /api/optogpt/infer/
- 功能:  接收 (R/T, λ, 值, 权重) 的行编辑数据, 构造目标谱与权重, 
         调用已加载的 Transformer 逆设计模型, 进行贪心 + Top-kp 采样, 
         用 TFCalc 评分挑选最优膜系, 并返回列表和最佳结果。

注意: 本文件将模型与辅助函数放在模块顶层, 只在 Django 进程启动时加载一次。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt

# ---------------------------------------------------------------------
# 1) Python 模块路径修正: 让 `import core....` 能成功
# 仓库结构假设:  .../optogpt/optogpt/optogpt_backend/optogpt_api/views.py
# 我们把  .../optogpt/optogpt  加到 sys.path
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parent.parent.parent  # .../optogpt/optogpt
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# ---------------------------------------------------------------------
# 2) 导入你工程里的模块
# ---------------------------------------------------------------------
from core.datasets.sim import inc_tmm  # noqa: E402
from core.models.transformer import make_model_I, subsequent_mask  # noqa: E402
from core.trains.train import LabelSmoothing, count_params  # noqa: E402
from core.datasets.datasets import PrepareData  # noqa: E402

# ---------------------------------------------------------------------
# 3) 常量 / 设备
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- 折射率表 & 波长网格 ----------------------------
def make_wavelengths() -> np.ndarray:
    return np.arange(380.0, 750.0 + 1e-9, 5.0)  # nm

lam_nm: np.ndarray = make_wavelengths()              # nm
wavelengths_um: np.ndarray = lam_nm / 1e3            # µm, 仅用于界面一致

# TiO2 / SiO2 / MgF2 / Glass — 简化表 (与你脚本一致)
lam_tab_tio2 = np.array([380.0,425.0,450.0,475.0,500.0,525.0,550.0,575.0,600.0,625.0,650.0,675.0,750.0,775.0,800.0,825.0,850.0,900.0,1000.0,1060.0])
n_tab_tio2   = np.array([2.55,2.49,2.469,2.444,2.422,2.402,2.385,2.37,2.351,2.343,2.337,2.331,2.322,2.317,2.313,2.311,2.309,2.305,2.300,2.299])

def n_tio2(lam: np.ndarray) -> np.ndarray:  # nm
    return np.interp(lam, lam_tab_tio2, n_tab_tio2)

lam_tab_sio2 = np.array([300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,900.0,1000.0])
n_tab_sio2   = np.array([1.478 ,1.472 ,1.467 ,1.463 ,1.459 ,1.455 ,1.452 ,1.450 ,1.446 ,1.437 ,1.434])

def n_sio2(lam: np.ndarray) -> np.ndarray:  # nm
    return np.interp(lam, lam_tab_sio2, n_tab_sio2)

lam_tab_mgf2 = np.array([248.0, 550.0, 1550.0])
n_tab_mgf2   = np.array([1.40 , 1.38 , 1.36  ])

def n_mgf2(lam: np.ndarray) -> np.ndarray:  # nm
    return np.interp(lam, lam_tab_mgf2, n_tab_mgf2)

glass_n_const = 1.5163

def n_glass(lam: np.ndarray) -> np.ndarray:
    return np.full_like(lam, glass_n_const, dtype=float)

nk_dict: Dict[str, np.ndarray] = {
    'TiO2': n_tio2(lam_nm).astype(np.complex128),
    'SiO2': n_sio2(lam_nm).astype(np.complex128),
    'MgF2': n_mgf2(lam_nm).astype(np.complex128),
    'Glass_Substrate': n_glass(lam_nm).astype(np.complex128),
}

# ------------------------ TMM 包装 -------------------------------
def spectrum(materials: List[str], thickness: List[float], *, pol: str = 's', theta: float = 0,
             wavelengths: np.ndarray, nk_dict: Dict[str, np.ndarray],
             substrate: str = 'Glass_Substrate', substrate_thick: float = 500000.0) -> List[float]:
    """返回 [R..., T...] (与脚本一致)"""
    assert len(materials) == len(thickness)
    assert pol in ('s', 'p', 'u')

    theta_rad = theta * (math.pi / 180.0)
    wavess = (1e3 * np.asarray(wavelengths)).astype(int)  # µm -> nm int

    thickness_full = [np.inf] + list(thickness) + [substrate_thick, np.inf]
    inc_list = ['i'] + ['c'] * len(materials) + ['i', 'i']

    R, T = [], []

    def _rt(pol_char, n_list, d_list, inc_list, theta_in, lambda_vac):
        res = inc_tmm(pol_char, n_list, d_list, inc_list, theta_in, lambda_vac)
        return res['R'], res['T']

    for i, lambda_vac in enumerate(wavess):
        n_list = [1.0] + [nk_dict[m][i] for m in materials] + [nk_dict[substrate][i], 1.0]
        if pol == 'u':
            rs, ts = _rt('s', n_list, thickness_full, inc_list, theta_rad, lambda_vac)
            rp, tp = _rt('p', n_list, thickness_full, inc_list, theta_rad, lambda_vac)
            r, t = 0.5 * (rs + rp), 0.5 * (ts + tp)
        else:
            r, t = _rt(pol, n_list, thickness_full, inc_list, theta_rad, lambda_vac)
        R.append(float(r)); T.append(float(t))
    return R + T

# ------------------------ 工具函数 -------------------------------
SPECIAL = {"BOS", "EOS", "PAD", "UNK", None, ""}

def clean_tokens(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        if tok in SPECIAL:
            if tok == "EOS":
                break
            continue
        out.append(tok)
    return out


def return_mat_thick(tokens: List[str], max_layers: int = 20) -> Tuple[List[str], List[float]]:
    tokens = clean_tokens(tokens)[:max_layers]
    mats, thks = [], []
    for tok in tokens:
        s = str(tok)
        if "_" not in s:
            if mats and s.replace(".", "", 1).isdigit():
                mats.append(mats[-1]); thks.append(float(s))
            continue
        mat, thk = s.split("_", 1)
        num = "".join(ch for ch in thk if ch.isdigit() or ch == ".")
        if not num:
            continue
        thks.append(float(num)); mats.append(mat)
    return mats, thks


def _is_vector(x: Any, n: int | None = None) -> bool:
    try:
        arr = np.asarray(x).squeeze()
    except Exception:
        return False
    if arr.ndim != 1:
        return False
    return (n is None) or (arr.size == n)


def extract_RT(spec_obj: Any, wavelengths_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(spec_obj, (list, tuple)):
        if len(spec_obj) >= 2 and all(_is_vector(v) for v in spec_obj[:2]):
            return np.asarray(spec_obj[0]).squeeze(), np.asarray(spec_obj[1]).squeeze()
    if isinstance(spec_obj, dict):
        keys = {str(k).lower(): k for k in spec_obj.keys()}
        if "r" in keys and "t" in keys:
            return np.asarray(spec_obj[keys["r"]]).squeeze(), np.asarray(spec_obj[keys["t"]]).squeeze()
        r_key = next((k for k in spec_obj if str(k).lower().startswith("r")), None)
        t_key = next((k for k in spec_obj if str(k).lower().startswith("t")), None)
        if r_key is not None and t_key is not None:
            return np.asarray(spec_obj[r_key]).squeeze(), np.asarray(spec_obj[t_key]).squeeze()
    for attr_r in ("R", "r"):
        for attr_t in ("T", "t"):
            if hasattr(spec_obj, attr_r) and hasattr(spec_obj, attr_t):
                return np.asarray(getattr(spec_obj, attr_r)).squeeze(), np.asarray(getattr(spec_obj, attr_t)).squeeze()
    arr = np.asarray(spec_obj).squeeze()
    wl = wavelengths_len
    if arr.ndim == 1 and arr.size == 2 * wl:
        first, second = arr[:wl], arr[wl:]
        if np.all((first >= 0) & (first <= 1)) and np.all((second >= 0) & (second <= 1)):
            return first, second
        return second, first
    raise ValueError("extract_RT: 无法解析 (R,T)。")


def assemble_spec_vector(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32).reshape(-1)
    T = np.asarray(T, dtype=np.float32).reshape(-1)
    if R.shape != T.shape:
        raise ValueError(f"R/T 长度不一致: {R.shape} vs {T.shape}")
    return np.concatenate([R, T], axis=0)


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None = None) -> float:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if w is None:
        return float(np.mean(np.abs(a - b)))
    w = np.asarray(w, dtype=np.float64)
    return float(np.sum(np.abs(a - b) * w) / (np.sum(w) + 1e-12))


def parse_user_targets(directives_text: str, wavelengths_nm: np.ndarray,
                       default_R: float = 0.0, default_T: float = 0.0,
                       clip01: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    wl = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    N = wl.size
    pts: Dict[str, List[Tuple[float, float]]] = {"R": [], "T": []}
    lines = [ln.strip() for ln in str(directives_text).strip().splitlines() if ln.strip()]
    for raw in lines:
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) < 3:
            raise ValueError(f"格式错误: {raw}，应为 通道, 波长, 值[, 权重]")
        ch = parts[0].upper()
        if ch not in ("R", "T"):
            raise ValueError(f"未知通道: {ch}（只支持 R/T）")
        lam = float(parts[1])
        val = float(parts[2])
        if val > 1.0:
            val = val / 100.0
        pts[ch].append((lam, val))

    def build_channel(target_points: List[Tuple[float, float]], default_val: float) -> np.ndarray:
        if len(target_points) == 0:
            arr = np.full(N, float(default_val), dtype=np.float32)
        elif len(target_points) == 1:
            v = float(target_points[0][1])
            arr = np.full(N, v, dtype=np.float32)
        else:
            target_points = sorted(target_points, key=lambda x: x[0])
            xs = np.array([p[0] for p in target_points], dtype=float)
            ys = np.array([p[1] for p in target_points], dtype=float)
            arr = np.interp(wl, xs, ys, left=ys[0], right=ys[-1]).astype(np.float32)
        if clip01:
            np.clip(arr, 0.0, 1.0, out=arr)
        return arr

    R = build_channel(pts["R"], default_R)
    T = build_channel(pts["T"], default_T)
    return R, T


def build_spec_weights_from_points(wavelengths_nm: np.ndarray, directives_text: str, base: float = 1.0) -> np.ndarray:
    wl = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    N = wl.size
    wR = np.full(N, base, dtype=np.float32)
    wT = np.full(N, base, dtype=np.float32)
    lines = [ln.strip() for ln in directives_text.strip().splitlines() if ln.strip()]
    for raw in lines:
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) < 4:
            raise ValueError(f"格式错误: {raw}，应为 通道, 波长, 值, 权重")
        ch, lam, _val, alpha = parts[0].upper(), float(parts[1]), float(parts[2]), float(parts[3])
        idx = int(np.argmin(np.abs(wl - lam)))
        if ch == "R":
            wR[idx] = float(alpha)
        elif ch == "T":
            wT[idx] = float(alpha)
        else:
            raise ValueError(f"未知通道 {ch}")
    return np.concatenate([wR, wT], axis=0)


def build_spec_weights_from_points_gaussian(wavelengths_nm: np.ndarray, directives_text: str, sigma_nm: float = 15.0, base: float = 0.0) -> np.ndarray:
    wl = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    N = wl.size
    wR = np.full(N, base, dtype=np.float32)
    wT = np.full(N, base, dtype=np.float32)
    lines = [ln.strip() for ln in directives_text.strip().splitlines() if ln.strip()]
    for raw in lines:
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) < 4:
            raise ValueError(f"格式错误: {raw}")
        ch, lam, _val, alpha = parts[0].upper(), float(parts[1]), float(parts[2]), float(parts[3])
        g = np.exp(-0.5 * ((wl - lam) / sigma_nm) ** 2)
        g = g / (g.max() + 1e-12) * float(alpha)
        if ch == "R":
            wR = np.maximum(wR, g.astype(np.float32))
        elif ch == "T":
            wT = np.maximum(wT, g.astype(np.float32))
        else:
            raise ValueError(f"未知通道 {ch}")
    return np.concatenate([wR, wT], axis=0)


# ------------------------ TFCalc 评分 -------------------------------
def tfcalc_merit(C_vec, T_vec, N_vec=None, Tol_vec=None, I=1.0, D=1.0, k: int = 2, eps: float = 1e-12) -> float:
    C = np.asarray(C_vec, dtype=np.float64).reshape(-1)
    T = np.asarray(T_vec, dtype=np.float64).reshape(-1)
    assert C.shape == T.shape, "C/T 维度不一致"
    m = C.size

    def _to_arr(x):
        if x is None:
            return np.ones_like(C, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        return x if x.shape == C.shape else np.full_like(C, float(x), dtype=np.float64)

    N = _to_arr(N_vec)
    Tol = _to_arr(Tol_vec)
    Iv = _to_arr(I)
    Dv = _to_arr(D)

    diff_k = np.power(np.abs(C - T) + eps, float(k))
    term = (Iv * Dv * diff_k) / (np.maximum(N * Tol, eps))
    F_powk = np.sum(term) / max(1, m)
    F = np.power(F_powk, 1.0 / float(k))
    return float(F)


def build_tfcalc_params(wavelengths_nm: np.ndarray, directives_text: str | None = None, N: float = 1.0, Tol: float = 0.05, I: float = 1.0, D: float = 1.0, use_weight_to_tol: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wl = np.asarray(wavelengths_nm, dtype=float).reshape(-1)
    Nw = np.full(wl.size, float(N), dtype=np.float32)
    Tolw = np.full(wl.size, float(Tol), dtype=np.float32)
    Iw = np.full(wl.size, float(I), dtype=np.float32)
    Dw = np.full(wl.size, float(D), dtype=np.float32)

    if directives_text:
        lines = [ln.strip() for ln in directives_text.strip().splitlines() if ln.strip()]
        for raw in lines:
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 2:
                continue
            ch, lam = parts[0].upper(), float(parts[1])
            alpha = float(parts[3]) if len(parts) >= 4 else 1.0
            idx = int(np.argmin(np.abs(wl - lam)))
            if use_weight_to_tol:
                Tolw[idx] = float(Tol) / max(alpha, 1e-6)
            else:
                Tolw[idx] = float(Tol)
            Iw[idx] = float(I)
            Dw[idx] = float(D)

    return (np.concatenate([Nw, Nw], axis=0),
            np.concatenate([Tolw, Tolw], axis=0),
            np.concatenate([Iw, Iw], axis=0),
            np.concatenate([Dw, Dw], axis=0))


# ------------------------ 采样/解码器 -------------------------------

def mask_invalid_next(probs: torch.Tensor, word_dict: Dict[str, int], step_layers: int, max_layers: int = 20) -> torch.Tensor:
    if probs.dim() == 2:
        probs = probs[0]
    probs = probs.clone()
    eos_id = word_dict['EOS']
    bos_id = word_dict['BOS']
    if step_layers >= max_layers:
        keep = torch.zeros_like(probs); keep[eos_id] = 1.0
        probs = probs * keep
    else:
        probs[bos_id] = 0.0
    s = probs.sum()
    if s.item() <= 0:
        probs[:] = 0.0
        probs[eos_id] = 1.0
    else:
        probs = probs / s
    return probs.unsqueeze(0)


def apply_top_k_top_p(logits: torch.Tensor, top_k: int = 10, top_p: float = 0.8) -> torch.Tensor:
    top_k = max(1, int(top_k))
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    if top_k < logits.size(-1):
        thresh = sorted_logits[..., top_k - 1:top_k]
        remove = sorted_logits < thresh
        sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
    probs_sorted = torch.softmax(sorted_logits, dim=-1)
    cum = probs_sorted.cumsum(dim=-1)
    remove = cum > top_p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
    out = torch.full_like(logits, float('-inf'))
    out.scatter_(1, sorted_idx, sorted_logits)
    return out


def greedy_decode_w(model, struc_word_dict, R_target, T_target, max_len, start_symbol="BOS", spec_weights=None, device=None, eos_bias_after=15, eos_bias_logit=1.5, max_layers=20):
    DEVICE_ = device or next(model.parameters()).device
    id2tok = {v: k for k, v in struc_word_dict.items()}
    BOS_id = struc_word_dict.get(start_symbol, 2)
    EOS_id = struc_word_dict['EOS']
    PAD_id = 0
    spec_vec = assemble_spec_vector(R_target, T_target)
    src = torch.tensor([spec_vec], dtype=torch.float32, device=DEVICE_)
    ys = torch.tensor([[BOS_id]], dtype=torch.long, device=DEVICE_)
    step_layers = 0
    probs_hist = []
    for _ in range(max_len - ys.size(1)):
        tgt_mask = (ys != PAD_id).unsqueeze(-2) & (subsequent_mask(ys.size(-1)).to(ys.device))
        out = model(src, ys, src_mask=None, tgt_mask=tgt_mask)
        dec_last = out[:, -1, :]
        logp = model.generator(dec_last)
        if step_layers >= eos_bias_after:
            logp[:, EOS_id] = logp[:, EOS_id] + eos_bias_logit
        probs = torch.exp(logp)
        probs = mask_invalid_next(probs, struc_word_dict, step_layers, max_layers=max_layers)
        next_id = int(torch.argmax(probs, dim=-1).item())
        probs_hist.append(float(probs[0, next_id].item()))
        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE_)], dim=1)
        if next_id == EOS_id:
            break
        step_layers += 1
    if step_layers >= max_layers and ys[0, -1].item() != EOS_id:
        ys = torch.cat([ys, torch.tensor([[EOS_id]], dtype=torch.long, device=DEVICE_)], dim=1)
    raw_tokens = [id2tok.get(tid, None) for tid in ys[0].tolist()]
    cleaned_tokens = []
    for tok in raw_tokens:
        if tok in {"BOS", "PAD", "UNK", None, ""}:
            continue
        if tok == "EOS":
            break
        cleaned_tokens.append(tok)
    return raw_tokens, cleaned_tokens, probs_hist


def top_k_n_w(k, top_p, model, struc_word_dict, R_target, T_target, max_len, start_symbol="BOS", spec_weights=None, device=None, temperature=1.0, start_mat=None, eos_bias_after=15, eos_bias_logit=1.5, max_layers=20):
    DEVICE_ = device or next(model.parameters()).device
    id2tok = {v: k for k, v in struc_word_dict.items()}
    BOS_id = struc_word_dict.get(start_symbol, 2)
    EOS_id = struc_word_dict['EOS']
    PAD_id = 0
    spec_vec = assemble_spec_vector(R_target, T_target)
    src = torch.tensor([spec_vec], dtype=torch.float32, device=DEVICE_)
    if start_mat:
        ys = torch.tensor([[BOS_id, struc_word_dict[start_mat]]], dtype=torch.long, device=DEVICE_)
        step_layers = 1
    else:
        ys = torch.tensor([[BOS_id]], dtype=torch.long, device=DEVICE_)
        step_layers = 0
    probs_hist = []
    for _ in range(max_len - ys.size(1)):
        tgt_mask = (ys != PAD_id).unsqueeze(-2) & (subsequent_mask(ys.size(-1)).to(ys.device))
        out = model(src, ys, src_mask=None, tgt_mask=tgt_mask)
        dec_last = out[:, -1, :]
        logp = model.generator(dec_last) / max(1e-6, float(temperature))
        if step_layers >= eos_bias_after:
            logp[:, EOS_id] = logp[:, EOS_id] + eos_bias_logit
        logits_kept = apply_top_k_top_p(logp, top_k=k, top_p=top_p)
        probs = torch.softmax(logits_kept, dim=-1)
        probs = mask_invalid_next(probs, struc_word_dict, step_layers, max_layers=max_layers)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        probs_hist.append(float(probs[0, next_id].item()))
        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE_)], dim=1)
        if next_id == EOS_id:
            break
        step_layers += 1
    if step_layers >= max_layers and ys[0, -1].item() != EOS_id:
        ys = torch.cat([ys, torch.tensor([[EOS_id]], dtype=torch.long, device=DEVICE_)], dim=1)
    tokens = []
    for tid in ys[0].tolist():
        tok = id2tok.get(tid, None)
        if tok in (None, 'BOS', 'PAD', 'UNK'):
            continue
        if tok == 'EOS':
            break
        tokens.append(tok)
    return tokens, probs_hist


# ------------------------ 评估 -------------------------------
TF_K = 2  # 默认, 可通过请求覆盖


def eval_structure(tokens_list: List[str]) -> Tuple[float, float, float, Tuple[np.ndarray, np.ndarray]]:
    if not tokens_list:
        zR, zT = np.zeros_like(lam_nm, dtype=float), np.zeros_like(lam_nm, dtype=float)
        return float("inf"), float("inf"), float("inf"), (zR, zT)
    try:
        mats, thks = return_mat_thick(tokens_list)
        spec_obj = spectrum(mats, thks, wavelengths=wavelengths_um, nk_dict=nk_dict, substrate='Glass_Substrate', substrate_thick=500000)
        R_sim, T_sim = extract_RT(spec_obj, wavelengths_len=len(wavelengths_um))
        vec_sim = assemble_spec_vector(R_sim, T_sim)

        # 目标/权重 由外层注入 (见 infer 函数局部变量)
        mae_plain = float(np.mean(np.abs(vec_sim - CURRENT_SPEC_TARGET_VEC)))
        mae_weight = weighted_mae(CURRENT_SPEC_TARGET_VEC, vec_sim, w=CURRENT_SPEC_WEIGHTS)
        tf_merit = tfcalc_merit(C_vec=vec_sim, T_vec=CURRENT_SPEC_TARGET_VEC, N_vec=CURRENT_N_VEC, Tol_vec=CURRENT_TOL_VEC, I=CURRENT_I_VEC, D=CURRENT_D_VEC, k=TF_K)
        return mae_plain, mae_weight, tf_merit, (R_sim, T_sim)
    except Exception as e:
        print("[WARN] eval_structure failed:", e)
        zR, zT = np.zeros_like(lam_nm, dtype=float), np.zeros_like(lam_nm, dtype=float)
        return float("inf"), float("inf"), float("inf"), (zR, zT)

# 这些“CURRENT_*”在处理一条请求时被赋值，用于 eval_structure()
CURRENT_SPEC_TARGET_VEC: np.ndarray = np.zeros(len(lam_nm) * 2, dtype=np.float32)
CURRENT_SPEC_WEIGHTS: np.ndarray = np.ones(len(lam_nm) * 2, dtype=np.float32)
CURRENT_N_VEC: np.ndarray = np.ones(len(lam_nm) * 2, dtype=np.float32)
CURRENT_TOL_VEC: np.ndarray = np.ones(len(lam_nm) * 2, dtype=np.float32) * 0.05
CURRENT_I_VEC: np.ndarray = np.ones(len(lam_nm) * 2, dtype=np.float32)
CURRENT_D_VEC: np.ndarray = np.ones(len(lam_nm) * 2, dtype=np.float32)


# ------------------------ 模型加载 (模块级, 只加载一次) -------------------------------
MODEL_OK = False
MODEL_INFO: Dict[str, Any] = {}

try:
   
    DEFAULT_MODEL_PATH = (
        "/data/PXY/optogpt/optogpt/saved_models/optogpt/test/"
        "model_inverse_R_T_S_R_B_LR_WU_L_H_D_F_[2, 0.05, 1024, 0.0004, 2000, 6, 8, 512, 2048]_best07.pt"
    )
    a = torch.load(DEFAULT_MODEL_PATH, map_location=DEVICE)
    args = a['configs']
    torch.manual_seed(args.seeds); np.random.seed(args.seeds)

    model = make_model_I(args.spec_dim, args.struc_dim, args.layers, args.d_model, args.d_ff, args.head_num, args.dropout).to(DEVICE)
    model.load_state_dict(a['model_state_dict'])
    model.eval()


    # ✅ 用绝对路径（按你的目录）
    DATASET_DIR = "/data/PXY/optogpt/optogpt/dataset"
    TRAIN_FILE = f"{DATASET_DIR}/Structure_train.pkl"
    TRAIN_SPEC_FILE = f"{DATASET_DIR}/Spectrum_train.pkl"
    DEV_FILE = f"{DATASET_DIR}/Structure_dev.pkl"
    DEV_SPEC_FILE = f"{DATASET_DIR}/Spectrum_dev.pkl"

    data = PrepareData(TRAIN_FILE, TRAIN_SPEC_FILE, args.ratios, DEV_FILE, DEV_SPEC_FILE,
                       args.batch_size, 'R_T', 'Inverse')
    data.struc_word_dict, data.struc_index_dict = a['configs'].struc_word_dict, a['configs'].struc_index_dict

    MODEL_OK = True
    MODEL_INFO = {
        'device': str(DEVICE),
        'vocab_size': len(data.struc_word_dict),
        'spec_dim': int(args.spec_dim),
        'model_path': DEFAULT_MODEL_PATH,
        'dataset_dir': DATASET_DIR,
    }
except Exception as e:
    MODEL_OK = False
    MODEL_INFO = {'error': f'Model load failed: {e}'}
    print('[ERROR] model load failed:', e)

# ------------------------ JSON 安全数值 -------------------------------
def safe_float(x, default=0.0):
    """将数值转换为 JSON 安全的 float；非有限数回退到 default，再不行则 0.0。"""
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    try:
        v_def = float(default)
        if np.isfinite(v_def):
            return v_def
    except Exception:
        pass
    return 0.0


def merge_same_material_layers(tokens: list, max_thick: float = 300.0):
    """
    将相邻相同材料的层合并厚度；若任一合并后厚度超限则判定无效。
    返回 (merged_tokens, valid_bool)。
    """
    merged = []
    last_mat = None
    last_thk = 0.0
    for tok in tokens:
        if not tok or "_" not in str(tok):
            continue
        mat, thk = str(tok).split("_", 1)
        try:
            thk_val = float(thk)
        except Exception:
            continue
        if mat == last_mat:
            last_thk += thk_val
            if last_thk > max_thick:
                return [], False
            merged[-1] = f"{mat}_{int(round(last_thk))}"
        else:
            if thk_val > max_thick:
                return [], False
            merged.append(f"{mat}_{int(round(thk_val))}")
            last_mat = mat
            last_thk = thk_val
    return merged, True

# ------------------------ 视图函数 -------------------------------
@csrf_exempt
def infer(request: HttpRequest):
    """POST /api/optogpt/infer/"""
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': 'Only POST allowed'}, status=405)

    if not MODEL_OK:
        return JsonResponse({'ok': False, 'error': f"Model not ready: {MODEL_INFO.get('error','unknown')}"}, status=503)

    try:
        raw_body = request.body.decode('utf-8') if request.body else ''
        try:
            body = json.loads(raw_body) if raw_body else {}
        except Exception as e:
            return JsonResponse({'ok': False, 'error': f'Invalid JSON: {e}', 'raw': raw_body[:500]}, status=400)

        # -------- 兼容前端多余字段：lam_nm / R_target / T_target / spec_weights / tf_params / weight_strategy --------
        rows_in = body.get('rows') or []
        directives_text = (body.get('directives_text') or '').strip()

        # 前端有可能传 tf_params（而不是 tf）
        tf_cfg = body.get('tf') or body.get('tf_params') or {}
        # Tol 是容差值（tolerance），默认 0.05（5%容差）
        # 注意：根据新公式，当偏差 = Tol 时，TF 分数 = 1（基准分数）
        # 所以 Tol=0.05 时，基准分数自动为 1；Tol 不是基准分数本身
        tol = float(tf_cfg.get('tol', tf_cfg.get('Tol', 0.05)))
        try:
            k_exp = int(tf_cfg.get('k', 2))
        except Exception:
            k_exp = 2
        weight_mode = (tf_cfg.get('weightMode') or tf_cfg.get('weight_mode') or 'fullband').lower()

        # 如果没直接传 directives_text，就从 rows 组装；rows 支持 list/dict 混合
        def parse_row(r):
            # list/tuple: ['R', 550, 0.2, 1]
            if isinstance(r, (list, tuple)):
                ch = str(r[0]).upper() if len(r) > 0 else 'R'
                lam = float(r[1]) if len(r) > 1 else 550
                v = float(r[2]) if len(r) > 2 else 0.0
                w = float(r[3]) if len(r) > 3 else 1.0
                return ch, lam, v, w
            # dict: {ch/lam/v/w 或 val}
            if isinstance(r, dict):
                ch_raw = r.get('ch', 'R')
                if isinstance(ch_raw, dict):
                    ch = str(ch_raw.get('value', ch_raw.get('text', 'R'))).upper()
                else:
                    ch = str(ch_raw).upper()
                lam_raw = r.get('lam', 550)
                if isinstance(lam_raw, dict):
                    lam = float(lam_raw.get('value', lam_raw.get('text', 550)))
                else:
                    lam = float(lam_raw)
                v_raw = r.get('v', r.get('val', 0.0))
                try:
                    v = float(v_raw)
                except Exception:
                    v = 0.0
                w_raw = r.get('w', 1.0)
                try:
                    w = float(w_raw)
                except Exception:
                    w = 1.0
                return ch, lam, v, w
            # fallback
            return 'R', 550.0, 0.0, 1.0

        if not directives_text:
            lines = []
            for r in rows_in:
                ch, lam, v, w = parse_row(r)
                lines.append(f"{ch},{lam},{v},{w}")
            directives_text = "\n".join(lines)

        # 没有任何行 → 给一个最简单的默认，避免后续解析报错
        if not directives_text.strip():
            directives_text = "R,550,0.0,1"

        # -------- 构造 R/T 目标 + 权重 + TFCalc 参数 --------
        R_target, T_target = parse_user_targets(directives_text, lam_nm)
        if weight_mode == 'gauss':
            spec_weights = build_spec_weights_from_points_gaussian(lam_nm, directives_text, sigma_nm=float(tf_cfg.get('sigma_nm', 15.0)), base=float(tf_cfg.get('base', 0.0)))
        elif weight_mode == 'anchors':
            spec_weights = build_spec_weights_from_points(lam_nm, directives_text, base=0.0)
        else:
            spec_weights = build_spec_weights_from_points(lam_nm, directives_text, base=1.0)

        N_vec, Tol_vec, I_vec, D_vec = build_tfcalc_params(
            lam_nm, directives_text=directives_text, N=1.0, Tol=tol, I=1.0, D=1.0, use_weight_to_tol=True
        )

        # 注入评分上下文
        global CURRENT_SPEC_TARGET_VEC, CURRENT_SPEC_WEIGHTS, CURRENT_N_VEC, CURRENT_TOL_VEC, CURRENT_I_VEC, CURRENT_D_VEC, TF_K
        CURRENT_SPEC_TARGET_VEC = assemble_spec_vector(R_target, T_target)
        CURRENT_SPEC_WEIGHTS = np.asarray(spec_weights, dtype=np.float32)
        CURRENT_N_VEC, CURRENT_TOL_VEC, CURRENT_I_VEC, CURRENT_D_VEC = N_vec, Tol_vec, I_vec, D_vec
        TF_K = k_exp

        # -------- 解码：贪心 + Top-kp --------
        model.eval()
        results = []
        best_score = float('inf')
        best_struct = []
        best_tag = 'Greedy'

        raw_tokens, cleaned_tokens, _ = greedy_decode_w(
            model, data.struc_word_dict, R_target, T_target,
            max_len=getattr(args, 'max_len', 22),
            start_symbol='BOS', spec_weights=None, device=DEVICE
        )
        # 合并相邻同材层；若无效则丢弃
        merged_tokens, valid = merge_same_material_layers(cleaned_tokens)
        if not valid or not merged_tokens:
            mae_plain = mae_weight = tf_merit = float('inf')
        else:
            mae_plain, mae_weight, tf_merit, _ = eval_structure(merged_tokens)
        tf_val = safe_float(tf_merit, default=mae_weight)
        score_g = tf_val if np.isfinite(tf_val) else safe_float(mae_weight, default=0.0)
        # 只有非空结构才采纳
        if valid and merged_tokens:
            results.append({'idx': -1, 'tag': 'Greedy', 'tf': tf_val, 'structure': merged_tokens})
            best_score, best_struct, best_tag = score_g, merged_tokens, 'Greedy'
        else:
            best_score, best_struct, best_tag = float('inf'), [], 'None'

        # 读取 Top-kp 数量：优先从 top_kp 读取，如果没有则从 samples 读取（向后兼容），默认 20
        # 限制范围在 5-50 之间
        try:
            top_kp_val = body.get('top_kp') or body.get('topKp') or body.get('samples') or 20
            SAMPLES = int(top_kp_val)
            SAMPLES = max(5, min(50, SAMPLES))  # 限制在 5-50 范围内
        except Exception:
            SAMPLES = 20

        # 需要收集有效（非空）样本；为空则跳过继续取，最多尝试 5 倍
        collected = 0
        attempt = 0
        max_attempt = SAMPLES * 5
        while collected < SAMPLES and attempt < max_attempt:
            attempt += 1
            struc_k, _ = top_k_n_w(
                k=10, top_p=0.8, model=model, struc_word_dict=data.struc_word_dict,
                R_target=R_target, T_target=T_target,
                max_len=getattr(args, 'max_len', 22),
                start_symbol='BOS', spec_weights=None, device=DEVICE
            )
            # 合并相邻同材层；若无效或为空则跳过
            merged_tokens, valid = merge_same_material_layers(struc_k)
            if not valid or not merged_tokens:
                continue
            mae_p, mae_w, tf_m, _ = eval_structure(merged_tokens)
            tf_val = safe_float(tf_m, default=mae_w)
            sc = tf_val if np.isfinite(tf_val) else safe_float(mae_w, default=0.0)
            results.append({'idx': collected, 'tag': f'Sample#{collected:02d}', 'tf': tf_val, 'structure': merged_tokens})
            collected += 1
            if sc < best_score:
                best_score, best_struct, best_tag = sc, merged_tokens, f'Sample#{collected-1:02d}'

        # 统一在返回前做一次 JSON 安全清洗
        best_score_safe = safe_float(best_score, default=0.0)
        for item in results:
            tfv = safe_float(item.get('tf', 0.0), default=0.0)
            item['tf'] = tfv

        return JsonResponse({
            'ok': True,
            'model': MODEL_INFO,
            'best': {'source': best_tag, 'score': best_score_safe, 'structure': best_struct},
            'samples': results,
            'structure': best_struct,
            'tf_score': best_score_safe,
        }, json_dumps_params={'ensure_ascii': False, 'allow_nan': False})

    except Exception as e:
        # 把异常信息直接回给前端，便于定位
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


@csrf_exempt
def calculate_spectrum(request: HttpRequest):
    """POST /api/optogpt/calculate-spectrum/ - 计算膜系的R/T光谱"""
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': 'Only POST allowed'}, status=405)

    try:
        raw_body = request.body.decode('utf-8') if request.body else ''
        try:
            body = json.loads(raw_body) if raw_body else {}
        except Exception as e:
            return JsonResponse({'ok': False, 'error': f'Invalid JSON: {e}', 'raw': raw_body[:500]}, status=400)

        structure = body.get('structure', [])
        if not structure or not isinstance(structure, list):
            return JsonResponse({'ok': False, 'error': 'structure must be a non-empty list'}, status=400)

        # 解析结构：从 ['TiO2_50', 'SiO2_100'] 格式提取材料和厚度
        materials, thicknesses = return_mat_thick(structure, max_layers=20)
        
        if not materials or not thicknesses:
            return JsonResponse({'ok': False, 'error': 'Failed to parse structure'}, status=400)

        # 使用TMM计算R/T
        wavelengths_um = lam_nm / 1e3  # nm -> µm
        result = spectrum(
            materials=materials,
            thickness=thicknesses,
            pol='u',  # 非偏振光
            theta=0,  # 垂直入射
            wavelengths=wavelengths_um,
            nk_dict=nk_dict,
            substrate='Glass_Substrate',
            substrate_thick=500000.0
        )

        # result 是 [R..., T...] 格式
        L = len(lam_nm)
        R_values = result[:L]
        T_values = result[L:]

        return JsonResponse({
            'ok': True,
            'R': R_values,
            'T': T_values,
            'wavelengths': lam_nm.tolist(),
            'materials': materials,
            'thicknesses': thicknesses
        }, json_dumps_params={'ensure_ascii': False})

    except Exception as e:
        return JsonResponse({'ok': False, 'error': str(e)}, status=500)


@csrf_exempt
def optimize_structure(request: HttpRequest):
    """POST /api/optogpt/optimize-structure/ - 使用BFGS优化膜系结构厚度"""
    if request.method != 'POST':
        return JsonResponse({'ok': False, 'error': 'Only POST allowed'}, status=405)

    try:
        from scipy.optimize import minimize
        
        raw_body = request.body.decode('utf-8') if request.body else ''
        try:
            body = json.loads(raw_body) if raw_body else {}
        except Exception as e:
            return JsonResponse({'ok': False, 'error': f'Invalid JSON: {e}', 'raw': raw_body[:500]}, status=400)

        structure = body.get('structure', [])
        target_R = body.get('target_R', [])
        target_T = body.get('target_T', [])
        wavelengths_nm = body.get('wavelengths', [])
        
        if not structure or not isinstance(structure, list):
            return JsonResponse({'ok': False, 'error': 'structure must be a non-empty list'}, status=400)
        
        if not wavelengths_nm:
            wavelengths_nm = lam_nm.tolist()
        else:
            wavelengths_nm = np.array(wavelengths_nm, dtype=float)

        # 解析结构：从 ['TiO2_50', 'SiO2_100'] 格式提取材料和厚度
        materials, initial_thicknesses = return_mat_thick(structure, max_layers=20)
        
        if not materials or not initial_thicknesses:
            return JsonResponse({'ok': False, 'error': 'Failed to parse structure'}, status=400)

        # 构建目标反射率（如果提供了target_R，使用它；否则使用target_T的补集）
        if target_R and len(target_R) == len(wavelengths_nm):
            target_R_array = np.array(target_R, dtype=float)
        elif target_T and len(target_T) == len(wavelengths_nm):
            # 如果只提供了T，假设R = 1 - T（忽略吸收）
            target_R_array = 1.0 - np.array(target_T, dtype=float)
        else:
            # 默认目标：最小反射率
            target_R_array = np.zeros_like(wavelengths_nm, dtype=float)
        # 构建目标透射率；若未提供，则假设 T = 1 - R（忽略吸收）
        if target_T and len(target_T) == len(wavelengths_nm):
            target_T_array = np.array(target_T, dtype=float)
        else:
            target_T_array = 1.0 - target_R_array

        # 使用BFGS优化厚度
        def tmm_reflectance_single(material_list, d_list, wavelength):
            """计算单波长下的反射率（使用现有的spectrum函数）"""
            wavelengths_um = np.array([wavelength / 1e3])
            result = spectrum(
                materials=material_list,
                thickness=d_list,
                pol='u',
                theta=0,
                wavelengths=wavelengths_um,
                nk_dict=nk_dict,
                substrate='Glass_Substrate',
                substrate_thick=500000.0
            )
            # result是[R..., T...]格式，返回R
            return result[0]

        def objective_function(d_list):
            """目标函数：最小化反射率与目标的均方误差"""
            # 检查厚度范围（20-300nm）
            for d in d_list:
                if d < 20.0 or d > 300.0:
                    return 1e20
            R = np.array([tmm_reflectance_single(materials, d_list, wl) for wl in wavelengths_nm])
            return float(np.sum((R - target_R_array) ** 2))

        # 设置边界条件
        thickness_bounds = [(20.0, 300.0) for _ in range(len(initial_thicknesses))]
        
        # 执行BFGS优化
        # 根据层数调整迭代次数：大幅减少迭代次数以加快速度
        num_layers = len(initial_thicknesses)
        # 对于层数少的（<=5层），使用较少迭代；层数多的，大幅减少迭代次数
        if num_layers <= 5:
            max_iter = 20
        elif num_layers <= 10:
            max_iter = 30
        else:
            max_iter = 40  # 层数多的最多40次，避免超时
        
        try:
            result = minimize(
                objective_function, 
                initial_thicknesses, 
                method='L-BFGS-B', 
                bounds=thickness_bounds, 
                options={'ftol': 1e-5, 'gtol': 1e-4, 'maxiter': max_iter}  # 放宽容差，加快收敛
            )
            
            # 即使优化未完全成功，也使用当前结果（可能已经改善）
            optimized_thicknesses = result.x.tolist()
        except Exception as e:
            # 如果优化出错，使用初始值
            print(f"[optimize_structure] 优化异常: {e}, 使用初始值")
            optimized_thicknesses = initial_thicknesses
        
        # 构建优化后的结构（保持材料顺序，更新厚度），并合并相邻同材层
        raw_structure = [f"{mat}_{int(round(thk))}" for mat, thk in zip(materials, optimized_thicknesses)]
        optimized_structure, structure_valid = merge_same_material_layers(raw_structure)
        if not structure_valid or not optimized_structure:
            return JsonResponse({
                'ok': False,
                'error': 'Optimized structure invalid after merging adjacent same-material layers (>300nm)',
            }, status=400, json_dumps_params={'ensure_ascii': False, 'allow_nan': False})
        
        # 计算优化后的R/T光谱
        wavelengths_um = wavelengths_nm / 1e3
        opt_result = spectrum(
            materials=materials,
            thickness=optimized_thicknesses,
            pol='u',
            theta=0,
            wavelengths=wavelengths_um,
            nk_dict=nk_dict,
            substrate='Glass_Substrate',
            substrate_thick=500000.0
        )
        
        L = len(wavelengths_nm)
        R_values = opt_result[:L]
        T_values = opt_result[L:]

        # 计算 TFCalc 评分（默认 k=2, Tol=0.05）
        target_vec = assemble_spec_vector(target_R_array, target_T_array)
        sim_vec = assemble_spec_vector(R_values, T_values)
        N_vec = np.ones_like(target_vec, dtype=np.float64)
        Tol_vec = np.ones_like(target_vec, dtype=np.float64) * 0.05
        I_vec = np.ones_like(target_vec, dtype=np.float64)
        D_vec = np.ones_like(target_vec, dtype=np.float64)
        tf_score = tfcalc_merit(C_vec=sim_vec, T_vec=target_vec, N_vec=N_vec, Tol_vec=Tol_vec, I=I_vec, D=D_vec, k=2)
        tf_score_safe = safe_float(tf_score, default=0.0)

        return JsonResponse({
            'ok': True,
            'R': R_values,
            'T': T_values,
            'wavelengths': wavelengths_nm.tolist(),
            'materials': materials,
            'initial_thicknesses': initial_thicknesses,
            'optimized_thicknesses': [float(t) for t in optimized_thicknesses],
            'optimized_structure': optimized_structure,
            'tf_score': tf_score_safe,
            'optimization_success': result.success,
            'optimization_iterations': int(result.nit),
            'final_objective': float(result.fun)
        }, json_dumps_params={'ensure_ascii': False})

    except ImportError:
        return JsonResponse({'ok': False, 'error': 'scipy.optimize not available'}, status=500)
    except Exception as e:
        import traceback
        return JsonResponse({'ok': False, 'error': str(e), 'traceback': traceback.format_exc()}, status=500)
