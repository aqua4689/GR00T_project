# this uses the gr00t conda environment

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import torch

# ------------------------------------------------------------------
# 1) repo root를 sys.path에 넣어주기 (네가 coderun 아래에서 돌리는 구조)
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
# 2) 모델 설정
# ------------------------------------------------------------------
#MODEL_PATH = "nvidia/GR00T-N1.5-3B"
MODEL_PATH = "./finetuned/gr1_arms_only.Nut_pouring_batch32_nodiffusion/"
EMBODIMENT_TAG = "gr1"
EMBODIMENT_CONFIG = "fourier_gr1_arms_only"

device = "cuda" if torch.cuda.is_available() else "cpu"
data_config = DATA_CONFIG_MAP[EMBODIMENT_CONFIG]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

USE_ASYNC = True  # 여기를 False로 하면 완전 sync로 동작

if USE_ASYNC and HAVE_ASYNC:
    policy = AsyncGr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    POLICY_IS_CORO = inspect.iscoroutinefunction(policy.get_action)
    print(f"[server] using ASYNC policy (coroutine={POLICY_IS_CORO})")
else:
    policy = SyncGr00tPolicy(
        model_path=MODEL_PATH,
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
    )
    POLICY_IS_CORO = False
    print("[server] using SYNC policy")

# ------------------------------------------------------------------
# 3) FastAPI
# ------------------------------------------------------------------
app = FastAPI()


class InferenceRequest(BaseModel):
    task: str
    obs: list  # 256x256x3 같은거 들어온다고 가정
    state: dict


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


@app.post("/inference")
async def run_inference(req: InferenceRequest):
    """
    Isaac Sim이 기대하는 그대로 리턴하는 버전.
    """
    # 1) obs
    obs_arr = np.array(req.obs, dtype=np.uint8)
    # sync 버전이 이렇게 했으니까 그대로 맞춤
    # (1, 256, 256, 3) 꼴로
    if obs_arr.ndim == 3:
        obs_arr = obs_arr.reshape((1, 256, 256, 3))

    step_data = {
        "video.ego_view": obs_arr,
        "annotation.human.action.task_description": [req.task],
    }

    # 2) state.* 붙이기 (sync 버전과 동일)
    for joint_part_name, joint_state in req.state.items():
        js = np.array(joint_state, dtype=float).reshape((1, -1))
        step_data[f"state.{joint_part_name}"] = js

    print(f"[server] task='{req.task}', obs={obs_arr.shape}, state_keys={list(req.state.keys())}")

    # 3) 모델 호출 (async면 await, 아니면 그냥)
    if POLICY_IS_CORO:
        predicted_action = await policy.get_action(step_data)
    else:
        predicted_action = policy.get_action(step_data)

    # 4) Isaac Sim이 sync 버전과 똑같이 받게 만든다
    return_data = {}
    for name, value in predicted_action.items():
        return_data[name] = _to_list(value)

    return return_data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9876)
