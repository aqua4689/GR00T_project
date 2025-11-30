from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import torch

# ------------------------------------------------------------------
# 1) repo root를 sys.path에 넣어주기
# ------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
TUTORIAL_DIR = THIS_FILE.parent                    # .../Isaac-Sim-with-Gr00t-Tutorial
REPO_ROOT = TUTORIAL_DIR.parent                    # .../Isaac-GR00T
sys.path.insert(0, str(REPO_ROOT))

print("[server] sys.path[:5] =", sys.path[:5])

import gr00t
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy as SyncGr00tPolicy

# AsyncGr00tPolicy가 있는 버전도 있고 없는 버전도 있어서 try
try:
    from gr00t.model.policy import AsyncGr00tPolicy
    HAVE_ASYNC = True
    print("[server] AsyncGr00tPolicy available")
except Exception:
    AsyncGr00tPolicy = None
    HAVE_ASYNC = False
    print("[server] AsyncGr00tPolicy NOT available")

import inspect

# ------------------------------------------------------------------
# 2) 모델 / BID 설정
# ------------------------------------------------------------------
# MODEL_PATH = "nvidia/GR00T-N1.5-3B"
MODEL_PATH = "./finetuned/gr1_arms_only.Nut_pouring_batch32_nodiffusion/"
EMBODIMENT_TAG = "gr1"
EMBODIMENT_CONFIG = "fourier_gr1_arms_only"

data_config = DATA_CONFIG_MAP[EMBODIMENT_CONFIG]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

USE_ASYNC = True  # True면 AsyncGr00tPolicy 사용 (있을 때)

# ---- BID hyperparameters ----
USE_BID = True                # 전역으로 BID 허용 여부
BID_NUM_SAMPLES = 2          # strong (그리고 weak 있으면 weak) 샘플 개수 (전체, 배치 크기)
BID_TOP_K = 2                 # backward loss 기준으로 고르는 mode size K
BID_RHO = 0.5                 # backward coherence 시간 가중치 (ρ)
BID_LAMBDA_B = 1.0            # LB 가중치
BID_LAMBDA_F = 1.0            # LF 가중치

# 약한 정책(weak policy) 체크포인트 경로.
# 따로 준비한 early / underfitting 체크포인트가 있으면 여기에 넣으면 됨.
# None이면 forward contrast의 negative term은 사용하지 않음.
WEAK_MODEL_PATH: Optional[str] = "./finetuned/gr1_arms_only.Nut_pouring_batch32_nodiffusion/checkpoint-10000"

# ---- GPU 설정 (단일 GPU에만 로드) ----
if torch.cuda.is_available():
    NUM_AVAILABLE_GPUS = torch.cuda.device_count()
else:
    NUM_AVAILABLE_GPUS = 0

# BID_DEVICE_ID 환경변수로 사용할 GPU 선택 가능 (기본 0)
DEVICE_ID = int(os.getenv("BID_DEVICE_ID", "0"))
if NUM_AVAILABLE_GPUS > 0:
    DEVICE_ID = max(0, min(DEVICE_ID, NUM_AVAILABLE_GPUS - 1))
    DEVICE_STR = f"cuda:{DEVICE_ID}"
else:
    DEVICE_STR = "cpu"

print(
    f"[server] torch.cuda.is_available()={torch.cuda.is_available()}, "
    f"NUM_AVAILABLE_GPUS={NUM_AVAILABLE_GPUS}, "
    f"USING_DEVICE={DEVICE_STR}"
)

POLICIES_ARE_ASYNC = USE_ASYNC and HAVE_ASYNC

# ---- strong policy: 단일 디바이스에 로드 ----
if POLICIES_ARE_ASYNC:
    strong_policy = AsyncGr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=DEVICE_STR,
    )
else:
    strong_policy = SyncGr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=DEVICE_STR,
    )

POLICY_IS_CORO = POLICIES_ARE_ASYNC and inspect.iscoroutinefunction(
    strong_policy.get_action
)

print(
    f"[server] using {'ASYNC' if POLICIES_ARE_ASYNC else 'SYNC'} strong policy "
    f"on device {DEVICE_STR} (coroutine={POLICY_IS_CORO})"
)

# ---- weak policy: 단일 디바이스에 로드 (선택사항) ----
if USE_BID and WEAK_MODEL_PATH is not None:
    try:
        if POLICIES_ARE_ASYNC:
            weak_policy = AsyncGr00tPolicy(
                model_path=WEAK_MODEL_PATH,
                embodiment_tag=EMBODIMENT_TAG,
                modality_config=modality_config,
                modality_transform=modality_transform,
                device=DEVICE_STR,
            )
        else:
            weak_policy = SyncGr00tPolicy(
                model_path=WEAK_MODEL_PATH,
                embodiment_tag=EMBODIMENT_TAG,
                modality_config=modality_config,
                modality_transform=modality_transform,
                device=DEVICE_STR,
            )
        print(
            f"[server] weak policy loaded from {WEAK_MODEL_PATH} on device {DEVICE_STR}"
        )
        WEAK_POLICY_IS_CORO = POLICIES_ARE_ASYNC and inspect.iscoroutinefunction(
            weak_policy.get_action
        )
    except Exception as e:
        print(
            f"[server] WARNING: failed to load weak policy from {WEAK_MODEL_PATH} on device {DEVICE_STR}: {e}"
        )
        weak_policy = None
        WEAK_POLICY_IS_CORO = False
else:
    weak_policy = None
    WEAK_POLICY_IS_CORO = False
    if USE_BID and WEAK_MODEL_PATH is not None:
        print(
            "[server] WARNING: weak policy could not be loaded; BID will run without negatives."
        )

# ---- BID 상태 메모리 (이전 decision chunk 등) ----
_prev_decision_flat: Optional[np.ndarray] = None
_prev_episode_id: Optional[int] = None

# ------------------------------------------------------------------
# 3) FastAPI
# ------------------------------------------------------------------
app = FastAPI()


class InferenceRequest(BaseModel):
    task: str
    obs: list  # 256x256x3 같은거 들어온다고 가정
    state: dict
    episode: Optional[int] = None  # 에피소드 ID (BID reset용)
    use_bid: Optional[bool] = False  # per-request로 BID 사용 여부


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# -------------------- BID Helper: flatten actions --------------------
def _flatten_actions_chunk(actions_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    { "action.left_arm": [T,d1], "action.right_arm": [T,d2], ... }
    -> [T, d1+d2+...]
    (모든 파트에서 time dim T가 같다고 가정, 다를 경우는 최소값 기준으로 자름)
    """
    if not actions_dict:
        return np.zeros((0, 0), dtype=float)

    keys = sorted(k for k in actions_dict.keys() if k.startswith("action."))
    if not keys:
        return np.zeros((0, 0), dtype=float)

    arrays: List[np.ndarray] = []
    T: Optional[int] = None
    for k in keys:
        arr = np.asarray(actions_dict[k], dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if T is None:
            T = arr.shape[0]
        else:
            T = min(T, arr.shape[0])
        arrays.append(arr)

    if T is None or T <= 0:
        return np.zeros((0, 0), dtype=float)

    arrays = [a[:T] for a in arrays]
    flat = np.concatenate(arrays, axis=1)  # [T, D_total]
    return flat


def _backward_loss(
    candidate_flat: np.ndarray, prev_flat: Optional[np.ndarray], rho: float
) -> float:

    if prev_flat is None or prev_flat.size == 0 or candidate_flat.size == 0:
        return 0.0

    T_new, D = candidate_flat.shape
    T_prev, _ = prev_flat.shape

    # overlap: new[0..K-1] vs prev[1..K]
    K_overlap = min(T_new, max(T_prev - 1, 0))
    if K_overlap <= 0:
        return 0.0

    new_seg = candidate_flat[:K_overlap]       # [K_overlap, D]
    prev_seg = prev_flat[1:1 + K_overlap]      # [K_overlap, D]

    # per-timestep L2 norm
    diff = new_seg - prev_seg                  # [K_overlap, D]
    dists = np.linalg.norm(diff, axis=1)       # [K_overlap]

    taus = np.arange(K_overlap, dtype=float)   # [K_overlap]
    weights = rho ** taus                      # [K_overlap]

    return float((weights * dists).sum())


def _pairwise_sqdist(a: np.ndarray, b: np.ndarray) -> float:

    if a.size == 0 or b.size == 0:
        return 0.0
    T = min(a.shape[0], b.shape[0])
    if T <= 0:
        return 0.0
    diff = a[:T] - b[:T]            # [T, D]
    dists = np.linalg.norm(diff, axis=1)  # [T]
    return float(dists.sum())


def _forward_loss(
    candidate_flat: np.ndarray,
    pos_flats: List[np.ndarray],
    neg_flats: List[np.ndarray],
) -> float:

    if not pos_flats and not neg_flats:
        return 0.0

    pos_term = 0.0
    neg_term = 0.0

    if pos_flats:
        dists = [_pairwise_sqdist(candidate_flat, p) for p in pos_flats]
        pos_term = float(sum(dists) / len(dists))

    if neg_flats:
        dists = [_pairwise_sqdist(candidate_flat, n) for n in neg_flats]
        neg_term = float(sum(dists) / len(dists))

    if not neg_flats:
        # negative가 없으면 forward contrast를 끈다.
        return 0.0

    return pos_term - neg_term


# -------------------- Batched step_data / action helper --------------------
def _make_batched_step_data(
    step_data: Dict[str, np.ndarray],
    batch_size: int,
    task_str: str,
) -> Dict[str, np.ndarray]:

    if batch_size <= 1:
        return step_data

    batched: Dict[str, np.ndarray] = {}

    for k, v in step_data.items():
        if k.startswith("video."):
            arr = np.asarray(v)
            # 기대: (1, H, W, C)
            if arr.ndim == 4 and arr.shape[0] == 1:
                # (1, 1, H, W, C)
                arr = np.expand_dims(arr, axis=1)
                # (B, 1, H, W, C)
                arr = np.repeat(arr, batch_size, axis=0)
            batched[k] = arr

        elif k.startswith("state."):
            arr = np.asarray(v, dtype=float)
            # 기대: (1, D)
            if arr.ndim == 2 and arr.shape[0] == 1:
                # (1, 1, D)
                arr = np.expand_dims(arr, axis=1)
                # (B, 1, D)
                arr = np.repeat(arr, batch_size, axis=0)
            batched[k] = arr

        elif k == "annotation.human.action.task_description":
            batched[k] = [task_str] * batch_size

        else:
            # 기타 키는 그대로 복사
            batched[k] = v

    return batched


def _split_batched_actions(
    actions_dict: Dict[str, np.ndarray],
    num_samples: int,
) -> List[Dict[str, np.ndarray]]:

    if not actions_dict:
        return []

    any_key = next(iter(actions_dict.keys()))
    arr0 = np.asarray(actions_dict[any_key])
    if arr0.ndim == 0:
        return [actions_dict]

    B = arr0.shape[0]
    B_use = min(B, num_samples)
    candidates: List[Dict[str, np.ndarray]] = []

    for i in range(B_use):
        cand: Dict[str, np.ndarray] = {}
        for name, value in actions_dict.items():
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == B:
                cand[name] = arr[i]
            else:
                # batch dim 없는 값은 그대로 공유
                cand[name] = arr
        candidates.append(cand)

    return candidates


@app.post("/inference")
async def run_inference(req: InferenceRequest):

    global _prev_decision_flat, _prev_episode_id

    # per-request BID flag
    use_bid_flag = USE_BID and bool(req.use_bid)

    # 1) obs
    obs_arr = np.array(req.obs, dtype=np.uint8)
    # (1, 256, 256, 3) 꼴로 맞추기
    if obs_arr.ndim == 3:
        obs_arr = obs_arr.reshape((1, 256, 256, 3))

    step_data: Dict[str, np.ndarray] = {
        "video.ego_view": obs_arr,
        "annotation.human.action.task_description": [req.task],
    }

    # 2) state.* 붙이기
    for joint_part_name, joint_state in req.state.items():
        js = np.array(joint_state, dtype=float).reshape((1, -1))
        step_data[f"state.{joint_part_name}"] = js

    print(
        f"[server] task='{req.task}', obs={obs_arr.shape}, "
        f"state_keys={list(req.state.keys())}, episode={req.episode}, use_bid={use_bid_flag}"
    )

    # 에피소드 변경 시 BID 메모리 리셋
    if req.episode is not None:
        if _prev_episode_id is None or _prev_episode_id != req.episode:
            print(f"[server][BID] reset decision memory for new episode {req.episode}")
            _prev_episode_id = req.episode
            _prev_decision_flat = None

    # ---- BID 비활성화: 단일 샘플 버전 ----
    if not use_bid_flag:
        if POLICY_IS_CORO:
            predicted_action = await strong_policy.get_action(step_data)
        else:
            predicted_action = strong_policy.get_action(step_data)

        # prev decision 업데이트
        np_actions: Dict[str, np.ndarray] = {}
        for name, value in predicted_action.items():
            np_actions[name] = np.asarray(value, dtype=float)
        _prev_decision_flat = _flatten_actions_chunk(np_actions)

        return_data: Dict[str, list] = {}
        for name, value in predicted_action.items():
            return_data[name] = _to_list(value)

        # chunk_size 힌트 추가
        any_key = next(iter(predicted_action.keys()))
        T_chunk = int(np.asarray(predicted_action[any_key]).shape[0])
        return_data["chunk_size"] = T_chunk

        return return_data

    # ---- BID 활성화: strong / weak에서 단일-GPU batched 샘플 생성 ----

    total_samples = BID_NUM_SAMPLES

    # 1) strong policy batched 샘플
    strong_candidates_np: List[Dict[str, np.ndarray]] = []

    batched_step_data = _make_batched_step_data(
        step_data=step_data,
        batch_size=total_samples,
        task_str=req.task,
    )

    if POLICY_IS_CORO:
        strong_batched = await strong_policy.get_action(batched_step_data)
    else:
        strong_batched = strong_policy.get_action(batched_step_data)

    strong_batched_np: Dict[str, np.ndarray] = {}
    for name, value in strong_batched.items():
        strong_batched_np[name] = np.asarray(value, dtype=float)

    strong_candidates_np = _split_batched_actions(strong_batched_np, total_samples)
    strong_flats: List[np.ndarray] = [
        _flatten_actions_chunk(cand) for cand in strong_candidates_np
    ]

    # 2) weak policy batched 샘플 (있을 때만)
    weak_candidates_np: List[Dict[str, np.ndarray]] = []
    weak_flats: List[np.ndarray] = []
    if weak_policy is not None and WEAK_MODEL_PATH is not None:
        batched_step_data_w = _make_batched_step_data(
            step_data=step_data,
            batch_size=total_samples,
            task_str=req.task,
        )

        if WEAK_POLICY_IS_CORO:
            weak_batched = await weak_policy.get_action(batched_step_data_w)
        else:
            weak_batched = weak_policy.get_action(batched_step_data_w)

        weak_batched_np: Dict[str, np.ndarray] = {}
        for name, value in weak_batched.items():
            weak_batched_np[name] = np.asarray(value, dtype=float)
        weak_candidates_np = _split_batched_actions(weak_batched_np, total_samples)
        weak_flats = [_flatten_actions_chunk(cand) for cand in weak_candidates_np]

    # 3) backward loss 계산
    LB_strong = [_backward_loss(f, _prev_decision_flat, rho=BID_RHO) for f in strong_flats]
    LB_weak = (
        [_backward_loss(f, _prev_decision_flat, rho=BID_RHO) for f in weak_flats]
        if weak_flats
        else []
    )

    # 4) A+ / A- 선택
    if strong_flats:
        strong_order = sorted(range(len(LB_strong)), key=lambda idx: LB_strong[idx])
        A_plus_idx = strong_order[: min(BID_TOP_K, len(strong_order))]
        A_plus_flats = [strong_flats[idx] for idx in A_plus_idx]
    else:
        A_plus_idx = []
        A_plus_flats = []

    if weak_flats:
        weak_order = sorted(range(len(LB_weak)), key=lambda idx: LB_weak[idx])
        A_minus_idx = weak_order[: min(BID_TOP_K, len(weak_order))]
        A_minus_flats = [weak_flats[idx] for idx in A_minus_idx]
    else:
        A_minus_idx = []
        A_minus_flats = []

    # 5) strong 후보들에 대해 L = λ_B L_B + λ_F L_F 를 계산하고 최적 샘플 선택
    best_idx = 0
    best_total = None

    for i, cand_flat in enumerate(strong_flats):
        lb = LB_strong[i]

        # 후보 i에 대해, A+에서 자기 자신은 빼고(pos), A-는 그대로 사용
        if A_plus_flats or A_minus_flats:
            pos_flats_for_i: List[np.ndarray] = []
            for idx, f in zip(A_plus_idx, A_plus_flats):
                if idx != i:  # 자기 자신이면 제외
                    pos_flats_for_i.append(f)
            if not pos_flats_for_i and A_plus_flats:
                # 전부 자기 자신이면, 그냥 전체 A+를 사용
                pos_flats_for_i = A_plus_flats

            lf = _forward_loss(cand_flat, pos_flats_for_i, A_minus_flats)
        else:
            lf = 0.0

        total = BID_LAMBDA_B * lb + BID_LAMBDA_F * lf

        if (best_total is None) or (total < best_total):
            best_total = total
            best_idx = i

    print(
        f"[server][BID] LB_strong={LB_strong}, "
        f"best_idx={best_idx}, best_total={best_total}, "
        f"has_weak={weak_policy is not None}"
    )

    best_actions_np = strong_candidates_np[best_idx]
    _prev_decision_flat = _flatten_actions_chunk(best_actions_np)

    # Isaac Sim으로 돌려보낼 JSON 변환
    return_data: Dict[str, list] = {}
    for name, value in best_actions_np.items():
        return_data[name] = _to_list(value)

    any_key = next(iter(best_actions_np.keys()))
    T_chunk = int(np.asarray(best_actions_np[any_key]).shape[0])
    return_data["chunk_size"] = T_chunk

    return return_data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9876)
