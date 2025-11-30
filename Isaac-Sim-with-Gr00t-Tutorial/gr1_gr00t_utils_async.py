import time
import numpy as np
import requests
import gr1_config
from dataclasses import dataclass
from typing import Dict, List

# ---- Data classes ----
@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: np.ndarray  # (dof,)

# ---- Payload helpers ----
def make_gr00t_input(task: str, obs: np.ndarray, joint_positions: np.ndarray) -> dict:
    """
    obs: (H, W, 3) with W=256, H<=256, uint8 RGB
    joint_positions: (54, )
    """
    gr00t_input = {
        "task": task,
        "obs": make_square_img(obs).tolist(),
        "state": {},
    }
    for joint_part, idxs in gr1_config.gr00t_joints_index.items():
        gr00t_input["state"][joint_part] = joint_positions[idxs].tolist()
    return gr00t_input

def make_square_img(obs: np.ndarray) -> np.ndarray:
    # pad to 256x256
    H, W, _ = obs.shape
    assert W == 256, f"Expected width=256, got {W}"
    pad_top = (256 - H) // 2
    out = np.zeros((256, 256, 3), dtype=np.uint8)
    out[pad_top:pad_top + H] = obs
    return out

# ---- Networking ----
def request_gr00t_inference(payload: dict, url: str = "http://localhost:9876/inference") -> dict:
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# ---- Action parsing / aggregation ----
def parse_actions_json_to_chunk(actions_json: Dict[str, List[List[float]]]) -> Dict[str, np.ndarray]:
    """
    {"action.left_arm": [[...], ...], "action.right_arm": [[...], ...], ...}
    -> {key: np.ndarray [chunk, dof]}
    """
    out = {}
    for k, v in actions_json.items():
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        out[k] = arr
    return out

def chunk_to_joint_positions(chunk_dict: Dict[str, np.ndarray], t: int) -> np.ndarray:
    """
    t번째 스텝의 전체 54-DoF 조인트 벡터 구성.
    """
    jp = np.zeros((54,), dtype=float)
    for name, arr in chunk_dict.items():
        # "action.left_arm" -> "left_arm"
        part = name.split(".", 1)[1]
        jp[gr1_config.gr00t_joints_index[part]] = arr[t]
    return jp

def aggregate_actions(old: np.ndarray, new: np.ndarray, mode: str = "latest_only") -> np.ndarray:
    """
    액션 병합 규칙: latest_only / average / weighted_average / conservative
    """
    if mode == "latest_only":
        return new
    if mode == "average":
        return 0.5 * old + 0.5 * new
    if mode == "weighted_average":
        return 0.3 * old + 0.7 * new
    if mode == "conservative":
        return 0.7 * old + 0.3 * new
    return new  # fallback

def make_timed_chunk(chunk_dict: Dict[str, np.ndarray], start_ts: float, start_timestep: int) -> list[TimedAction]:
    """
    청크 딕셔너리와 메타정보 -> TimedAction 리스트
    (여기서는 timestamp는 모두 동일하게 start_ts로 부여)
    """
    # 임의로 "left_arm"의 길이를 청크 크기로 사용 (모든 파트 동일하다고 가정)
    any_key = next(iter(chunk_dict))
    chunk_size = chunk_dict[any_key].shape[0]
    actions = []
    for i in range(chunk_size):
        # 54-DoF 벡터로 합치기
        joint = chunk_to_joint_positions(chunk_dict, i)
        actions.append(TimedAction(timestamp=start_ts, timestep=start_timestep + i + 1, action=joint))
    return actions


