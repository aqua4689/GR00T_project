from __future__ import annotations

import os
import time
import threading
import hashlib
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from isaacsim import SimulationApp

# ---- User params ----
EPISODE_NUM = 2
EACH_EPISODE_LEN = 480
RESULT_VIDEO_FILE = "./results/async_bid.mp4"
LOAD_WORLD_FILE = "./sim_environments/gr1_NutPouring.usd"

TASK = (
    "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. "
    "Pick up the yellow bowl and place it on the metallic measuring scale."
)

CAMERA_HEIGHT = 200
CAMERA_FOCAL_LENGTH = 1.2
CAMERA_FORWARD_DIST = 0.25
CAMERA_ANGLE = 70

SERVER = "http://localhost:9876"  # server must expose POST /inference
ENV_FPS = 30.0
ENV_DT = 1.0 / ENV_FPS

# ---- Action log path (mp4 -> txt) ----
_log_base, _ = os.path.splitext(RESULT_VIDEO_FILE)
ACTION_LOG_FILE = f"{_log_base}.txt"

# ---- Async policy knob ----
LEFT_ACTION_THRESHOLD = 11

# ---- Isaac Sim init ----
simulation_app = SimulationApp(
    {
        "headless": True,
        "create_new_stage": False,
        "open_usd": LOAD_WORLD_FILE,
        "sync_loads": True,
    }
)

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.sensors.camera.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.types import ArticulationAction

import gr1_config  # gr00t_joints_index를 제공
import gr1_gr00t_utils as gutils  # make_gr00t_input


# -------------------- HTTP helper --------------------
def post_json(path: str, payload: dict, timeout: float = 10.0) -> dict:
    r = requests.post(f"{SERVER}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -------------------- Actions utilities --------------------
def _infer_chunk_size(actions_dict: Dict[str, List[List[float]]]) -> int:
    if not actions_dict:
        return 0
    any_key = next(iter(actions_dict.keys()))
    arr = np.asarray(actions_dict[any_key], dtype=float)
    return int(arr.shape[0]) if arr.ndim >= 1 else 0


def actions_dict_to_joint_positions(
    actions_dict: Dict[str, List[List[float]]],
    t: int,
    full_dof: int,
) -> np.ndarray:
    jp = np.zeros((full_dof,), dtype=float)
    for part, idxs in gr1_config.gr00t_joints_index.items():
        key = f"action.{part}"
        arr = actions_dict.get(key)
        target_len = len(idxs)
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.ndim == 1:
            arr_np = arr_np[None, :]
        if not (0 <= t < arr_np.shape[0]):
            continue
        vec = arr_np[t]
        d_part = int(vec.shape[0])
        if d_part == target_len:
            jp[idxs] = vec
        elif d_part > target_len:
            jp[idxs] = vec[:target_len]
        else:
            padded = np.zeros((target_len,), dtype=float)
            padded[:d_part] = vec
            jp[idxs] = padded
    return jp


def small_hash_for_chunk(actions_dict: Dict[str, List[List[float]]], take: int = 4) -> str:
    """Chunk identity for dedup: hash of first `take` timesteps of concatenated parts."""
    if not actions_dict:
        return "empty"
    parts = []
    for part in sorted(gr1_config.gr00t_joints_index.keys()):
        key = f"action.{part}"
        arr = np.asarray(actions_dict.get(key, []), dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        head = arr[:take].astype(np.float32).tobytes()
        parts.append(head)
    return hashlib.sha1(b"".join(parts)).hexdigest()


# -------------------- Async next-chunk fetcher --------------------
class NextChunkFetcher:
    """
    Background fetch of next chunk via /inference.
    Stores the cursor 'H_at_start' and 'global_timestep_at_start' when the request was launched.
    The caller computes s = clamp(global_now - global_at_start, 0, next_K-1) at switch time.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._resp: Optional[dict] = None
        self._hash: Optional[str] = None
        self._H_at_start: int = 0
        self._global_timestep_at_start: int = 0  # [ADDED]
        self._thread: Optional[threading.Thread] = None

    def in_flight(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, payload: dict, H_at_start: int, global_timestep_at_start: int):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready = False
            self._resp = None
            self._hash = None
            self._H_at_start = int(H_at_start)
            self._global_timestep_at_start = int(global_timestep_at_start)
            self._thread = threading.Thread(target=self._run, args=(payload,), daemon=True)
            self._thread.start()

    def _run(self, payload: dict):
        try:
            resp = post_json("/inference", payload)
            actions = resp.get("actions", resp)
            self_hash = small_hash_for_chunk(actions)
        except Exception as e:
            print("[WARN] /inference async fetch failed:", repr(e))
            return
        with self._lock:
            self._resp = resp
            self._hash = self_hash
            self._ready = True

    def take_if_ready(self) -> Tuple[Optional[dict], Optional[str], Optional[int], Optional[int]]:
        with self._lock:
            if not self._ready or self._resp is None:
                return None, None, None, None
            resp = self._resp
            h = self._hash
            Hs = self._H_at_start
            Gs = self._global_timestep_at_start
            # reset
            self._resp = None
            self._hash = None
            self._ready = False
            return resp, h, Hs, Gs


# -------------------- Main --------------------
def main():
    print("## 1. setup scene")
    world = World()
    scene: Scene = world.scene
    gr1: Robot = scene.add(Robot(prim_path="/World/gr1", name="gr1"))
    ctrl: ArticulationController = gr1.get_articulation_controller()

    camera = Camera(
        prim_path="/World/gr1/head_yaw_link/camera",
        name="camera",
        translation=np.array([CAMERA_FORWARD_DIST, 0.0, 0.07]),
        frequency=60,
        resolution=(256, CAMERA_HEIGHT),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, CAMERA_ANGLE, 0]), degrees=True),
    )
    camera.set_focal_length(CAMERA_FOCAL_LENGTH)
    camera.set_clipping_range(0.1, 2)

    print("## 2. setup post-load")
    world.reset()
    camera.initialize()
    camera.add_motion_vectors_to_frame()

    dof = len(gr1.get_joint_positions())
    ctrl.set_gains(kps=np.full((dof,), 3000.0), kds=np.full((dof,), 100.0))

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video = cv2.VideoWriter(RESULT_VIDEO_FILE, fourcc, 30, (256, CAMERA_HEIGHT), isColor=True)

    # ---- init/clear action log file ----
    with open(ACTION_LOG_FILE, "w", encoding="utf-8") as logf:
        logf.write("# action log aligned with video frames\n")
        logf.write("# format: 'ep <ep> frame <frame_idx> chunk <chunk_id> (h:<hash4>) action <t> (gt:<g_step>)'\n")

    for ep in range(EPISODE_NUM):
        print(f"Starting episode {ep}")
        world.reset()
        gr1.set_joint_positions(positions=gr1_config.default_joint_position)
        for _ in range(100):
            world.step(render=True)


        fetcher = NextChunkFetcher()  # 에피소드마다 초기화 (잔여 응답 carry-over 방지)

        # 0) 첫 청크 동기 수급
        obs_rgba = camera.get_rgba()
        img = obs_rgba[:, :, :3]
        joints = gr1.get_joint_positions()
        payload = gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints, episode=ep)
        resp = post_json("/inference", payload)

        current_actions = resp.get("actions", resp)
        current_K = int(resp.get("chunk_size", 0)) or _infer_chunk_size(current_actions)
        current_hash = small_hash_for_chunk(current_actions)

        # 현재 청크 state
        chunk_id = 0
        t_cursor = 0
        global_action_timestep = 0  # Global step counter for actions
        started_async_this_chunk = False

        # 미리 받아둔 다음 청크 (staged)
        staged_resp: Optional[dict] = None
        staged_hash: Optional[str] = None
        staged_K: int = 0
        staged_H_at_start: int = 0
        staged_global_timestep_at_start: int = 0

        frames = 0
        while frames < EACH_EPISODE_LEN:
            frame_start = time.perf_counter()

            # A) 남은 액션이 THRESHOLD 이하면 다음 청크 비동기 인퍼런스 시작
            remaining_in_chunk = current_K - t_cursor
            if (
                (remaining_in_chunk <= LEFT_ACTION_THRESHOLD)
                and (not started_async_this_chunk)
                and (not fetcher.in_flight())
                and (staged_resp is None)
            ):
                obs_rgba = camera.get_rgba()
                img = obs_rgba[:, :, :3]
                joints = gr1.get_joint_positions()
                fetcher.start(
                    gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints, episode=ep),
                    H_at_start=t_cursor,
                    global_timestep_at_start=global_action_timestep,
                )
                started_async_this_chunk = True
                print(
                    f"[PREFETCH] Triggered at t={t_cursor} (K={current_K}), "
                    f"Gs={global_action_timestep}. Remaining={remaining_in_chunk}"
                )

            # B) 프리패치 수신 시 staged 버퍼에 적재
            r, h, Hs, Gs = fetcher.take_if_ready()
            if r is not None and h is not None:
                # 현재 청크와 동일한 해시면 무시 (중복 응답 방지)
                if h != current_hash:
                    staged_resp = r
                    staged_hash = h
                    actions = r.get("actions", r)
                    staged_K = int(r.get("chunk_size", 0)) or _infer_chunk_size(actions)
                    staged_H_at_start = int(Hs)
                    staged_global_timestep_at_start = int(Gs)
                    print(
                        f"[PREFETCH] Staged chunk (h:{h[:4]}, K={staged_K}) from "
                        f"H={staged_H_at_start}, Gs={staged_global_timestep_at_start}"
                    )

            # C) 프리엠션 조건: staged 가 있고, 지금 스위치해도 되는 상황이면 즉시 스위치
            if staged_resp is not None and staged_hash is not None and staged_hash != current_hash:
                # 's' 계산: (현재 글로벌 스텝 - 요청 시점 글로벌 스텝)
                steps_passed_since_launch = global_action_timestep - staged_global_timestep_at_start
                s = max(0, min(steps_passed_since_launch, staged_K - 1))

                # 스위치 수행
                print(
                    f"[PREEMPT] Chunk {chunk_id} (h:{current_hash[:4]}) "
                    f"-> {chunk_id+1} (h:{staged_hash[:4]})..."
                )
                print(
                    f"[PREEMPT] ... Gs_at_start={staged_global_timestep_at_start}, "
                    f"Gs_now={global_action_timestep}. Offset s={s} (K_new={staged_K})"
                )

                chunk_id += 1
                current_actions = staged_resp.get("actions", staged_resp)
                current_K = staged_K
                current_hash = staged_hash
                t_cursor = s  # 새 청크의 s부터 시작
                started_async_this_chunk = False
                staged_resp = None
                staged_hash = None
                staged_K = 0
                staged_H_at_start = 0
                staged_global_timestep_at_start = 0

            # D) 현재 청크에서 정확히 한 액션 적용
            if 0 <= t_cursor < current_K:
                jp = actions_dict_to_joint_positions(current_actions, t_cursor, full_dof=dof)
                jp = jp[:dof]


                ctrl.apply_action(ArticulationAction(joint_positions=jp))

                short_h = current_hash[:4] if current_hash else "none"
                with open(ACTION_LOG_FILE, "a", encoding="utf-8") as logf:
                    logf.write(
                        f"ep {ep} frame {frames} chunk {chunk_id} (h:{short_h}) "
                        f"action {t_cursor} (gt:{global_action_timestep})\n"
                    )

                t_cursor += 1
                global_action_timestep += 1  # Increment global step only on success

            else:
                # E) 청크 소진: prefetch가 있으면 그걸로, 없으면 동기 요청
                if staged_resp is not None and staged_hash is not None and staged_hash != current_hash:
                    # 스테이지된 청크 채택
                    steps_passed_since_launch = global_action_timestep - staged_global_timestep_at_start
                    s = max(0, min(steps_passed_since_launch, staged_K - 1))

                    print(
                        f"[EXHAUST] Chunk {chunk_id} (h:{current_hash[:4]}) "
                        f"-> {chunk_id+1} (h:{staged_hash[:4]})..."
                    )
                    print(
                        f"[EXHAUST] ... Gs_at_start={staged_global_timestep_at_start}, "
                        f"Gs_now={global_action_timestep}. Offset s={s} (K_new={staged_K})"
                    )

                    chunk_id += 1
                    current_actions = staged_resp.get("actions", staged_resp)
                    current_K = staged_K
                    current_hash = staged_hash
                    t_cursor = s
                    started_async_this_chunk = False
                    staged_resp = None
                    staged_hash = None
                    staged_K = 0
                    staged_H_at_start = 0
                    staged_global_timestep_at_start = 0
                else:
                    # 동기 새 청크
                    print(f"[SYNC] Chunk {chunk_id} (h:{current_hash[:4]}) exhausted. Fetching new...")
                    obs_rgba = camera.get_rgba()
                    img = obs_rgba[:, :, :3]
                    joints = gr1.get_joint_positions()
                    next_resp = post_json(
                        "/inference",
                        gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints, episode=ep),
                    )
                    next_actions = next_resp.get("actions", next_resp)
                    next_K = int(next_resp.get("chunk_size", 0)) or _infer_chunk_size(next_actions)
                    next_hash = small_hash_for_chunk(next_actions)

                    # 동일 해시면(중복) 바로 버리고 다시 요청 (무한루프 방지: 1회만 재시도)
                    if next_hash == current_hash:
                        print(f"[WARN] Got duplicate hash {next_hash[:4]}. Re-fetching...")
                        try:
                            next_resp2 = post_json(
                                "/inference",
                                gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints, episode=ep),
                            )
                            next_actions2 = next_resp2.get("actions", next_resp2)
                            next_K2 = int(next_resp2.get("chunk_size", 0)) or _infer_chunk_size(next_actions2)
                            next_hash2 = small_hash_for_chunk(next_actions2)
                            if next_hash2 != current_hash:
                                next_actions, next_K, next_hash = next_actions2, next_K2, next_hash2
                        except Exception as e:
                            print("[WARN] sync re-fetch failed:", repr(e))

                    if next_hash != current_hash:
                        chunk_id += 1
                        current_actions = next_actions
                        current_K = next_K
                        current_hash = next_hash
                        t_cursor = 0
                        started_async_this_chunk = False
                    else:
                        # 정말 동일 청크가 반복되면(서버 정지/캐시), 안전하게 idle 1프레임 대기
                        print(f"[WARN] Duplicate hash {next_hash[:4]} persists. Idling 1 frame.")
                        time.sleep(ENV_DT)

            # F) 렌더/비디오 기록
            world.step(render=True)
            obs_rgba = camera.get_rgba()
            img = obs_rgba[:, :, :3]
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # pacing
            frames += 1
            dt = time.perf_counter() - frame_start
            time.sleep(max(0.0, ENV_DT - dt))

        print(f"Episode {ep} finished")

    video.release()
    simulation_app.close()
    print(f"[INFO] action log saved to: {ACTION_LOG_FILE}")


if __name__ == "__main__":
    main()
