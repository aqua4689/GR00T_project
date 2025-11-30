from __future__ import annotations

import os
import time
import threading
import hashlib
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

from isaacsim import SimulationApp

# ---- User params ----
EPISODE_NUM = 2
EACH_EPISODE_LEN = 480
RESULT_VIDEO_FILE = "./results/async.mp4"
LOAD_WORLD_FILE = "./sim_environments/gr1_NutPouring.usd"

TASK = (
    "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. "
    "Pick up the yellow bowl and place it on the metallic measuring scale."
)

CAMERA_HEIGHT = 200
CAMERA_FOCAL_LENGTH = 1.2
CAMERA_FORWARD_DIST = 0.25
CAMERA_ANGLE = 70

SERVER = "http://localhost:9876"
ENV_FPS = 30.0
ENV_DT = 1.0 / ENV_FPS

_log_base, _ = os.path.splitext(RESULT_VIDEO_FILE)
ACTION_LOG_FILE = f"{_log_base}.txt"
METRIC_LOG_FILE = f"{_log_base}_metrics.txt"

LEFT_ACTION_THRESHOLD = 13
CHUNK_SWITCH_OFFSET = 2 

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

import gr1_config
import gr1_gr00t_utils as gutils

# -------------------- Comprehensive Smoothness Evaluator --------------------
class SmoothnessEvaluator:
    def __init__(self, fps: float):
        self.fps = fps
        self.history: List[np.ndarray] = []

    def update(self, joint_pos: np.ndarray):
        # Record actual physical position
        self.history.append(np.array(joint_pos, dtype=float))

    def compute_and_save(self, episode_idx: int, filepath: str):
        if len(self.history) < 10:
            print(f"[METRIC] Episode {episode_idx}: Not enough data.")
            return

        positions = np.array(self.history)
        dt = 1.0 / self.fps

        # 1. Pre-processing: Gaussian smoothing to remove sim quantization noise
        smoothed_pos = gaussian_filter1d(positions, sigma=2.0, axis=0)
        
        # 2. Derivatives
        vel = np.gradient(smoothed_pos, axis=0) / dt
        acc = np.gradient(vel, axis=0) / dt
        jerk = np.gradient(acc, axis=0) / dt
        
        # 3. Speed Profile (End-effector approximation via joint norm)
        speed_profile = np.linalg.norm(vel, axis=1)

        # --- Metrics Calculation ---

        # [Metric 1] L2 Jerk (Mean L2 Norm)
        jerk_norms = np.linalg.norm(jerk, axis=1)
        mean_filtered_jerk = np.mean(jerk_norms)

        # [Metric 2] MAJ (Mean Absolute Jerk) - Simple/Classical
        # Average of absolute jerk across all joints and time
        maj = np.mean(np.abs(jerk))

        # [Metric 3] NVP (Number of Velocity Peaks) - Sub-movements
        # Counts how many times the robot accelerated and decelerated (Stop-and-go)
        # Using simple peak finding with minimal distance
        peaks, _ = find_peaks(speed_profile, distance=int(self.fps * 0.2)) # Min 0.2s between peaks
        nvp = len(peaks)

        # [Metric 4] SAL (Speed Arc Length) - Frequency Domain
        # Smoothness in frequency domain. Higher (less negative) is better.
        max_speed = np.max(speed_profile) + 1e-9
        norm_speed = speed_profile / max_speed
        dt_norm = 1.0 / len(norm_speed)
        d_speed = np.diff(norm_speed)
        arc_len = np.sum(np.sqrt(dt_norm**2 + d_speed**2))
        sparc_score = -np.log(arc_len)

        # --- Report ---
        print(f"\n{'='*60}")
        print(f" [EVALUATION] Episode {episode_idx} (Async)")
        print(f"{'='*60}")
        print(f" 1. L2 Jerk (Mean L2) : {mean_filtered_jerk:.4f}  (Lower is Better)")
        print(f" 2. MAJ (Mean Abs Jerk)     : {maj:.4f}       (Lower is Better)")
        print(f" 3. NVP (Velocity Peaks)    : {nvp}             (Lower is Better)")
        print(f" 4. SAL (Speed Arc)    : {sparc_score:.4f}      (Higher/Less Neg is Better)")
        print(f"{'='*60}\n")

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"Ep {episode_idx} | ")
            f.write(f"L2_Jerk: {mean_filtered_jerk:.4f} | ")
            f.write(f"MAJ: {maj:.4f} | ")
            f.write(f"NVP: {nvp} | ")
            f.write(f"SAL: {sparc_score:.4f}\n")

# -------------------- HTTP helper --------------------
def post_json(path: str, payload: dict, timeout: float = 10.0) -> dict:
    r = requests.post(f"{SERVER}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

# -------------------- Actions utilities --------------------
def _infer_chunk_size(actions_dict: Dict[str, List[List[float]]]) -> int:
    if not actions_dict: return 0
    any_key = next(iter(actions_dict.keys()))
    arr = np.asarray(actions_dict[any_key], dtype=float)
    return int(arr.shape[0]) if arr.ndim >= 1 else 0

def actions_dict_to_joint_positions(actions_dict: Dict[str, List[List[float]]], t: int, full_dof: int) -> np.ndarray:
    jp = np.zeros((full_dof,), dtype=float)
    for part, idxs in gr1_config.gr00t_joints_index.items():
        key = f"action.{part}"
        arr = actions_dict.get(key)
        target_len = len(idxs)
        if arr is None: continue
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.ndim == 1: arr_np = arr_np[None, :]
        if not (0 <= t < arr_np.shape[0]): continue
        vec = arr_np[t]
        d_part = int(vec.shape[0])
        if d_part == target_len: jp[idxs] = vec
        elif d_part > target_len: jp[idxs] = vec[:target_len]
        else:
            padded = np.zeros((target_len,), dtype=float)
            padded[:d_part] = vec
            jp[idxs] = padded
    return jp

def small_hash_for_chunk(actions_dict: Dict[str, List[List[float]]], take: int = 4) -> str:
    if not actions_dict: return "empty"
    parts = []
    for part in sorted(gr1_config.gr00t_joints_index.keys()):
        key = f"action.{part}"
        arr = np.asarray(actions_dict.get(key, []), dtype=float)
        if arr.ndim == 1: arr = arr[None, :]
        head = arr[:take].astype(np.float32).tobytes()
        parts.append(head)
    return hashlib.sha1(b"".join(parts)).hexdigest()

# -------------------- Async next-chunk fetcher --------------------
class NextChunkFetcher:
    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._resp: Optional[dict] = None
        self._hash: Optional[str] = None
        self._H_at_start: int = 0
        self._global_timestep_at_start: int = 0
        self._thread: Optional[threading.Thread] = None

    def in_flight(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, payload: dict, H_at_start: int, global_timestep_at_start: int):
        with self._lock:
            if self._thread is not None and self._thread.is_alive(): return
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
            if not self._ready or self._resp is None: return None, None, None, None
            resp = self._resp
            h = self._hash
            Hs = self._H_at_start
            Gs = self._global_timestep_at_start
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

    with open(ACTION_LOG_FILE, "w", encoding="utf-8") as logf:
        logf.write("# action log\n")
    with open(METRIC_LOG_FILE, "w", encoding="utf-8") as mf:
        mf.write("# Episode | F_Jerk | MAJ | LDLJ | NVP | SPARC\n")

    for ep in range(EPISODE_NUM):
        print(f"Starting episode {ep}")
        world.reset()
        gr1.set_joint_positions(positions=gr1_config.default_joint_position)
        for _ in range(100):
            world.step(render=True)

        evaluator = SmoothnessEvaluator(fps=ENV_FPS)
        fetcher = NextChunkFetcher()

        # 0) Initial Chunk
        obs_rgba = camera.get_rgba()
        img = obs_rgba[:, :, :3]
        joints = gr1.get_joint_positions()
        payload = gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints)
        resp = post_json("/inference", payload)
        current_actions = resp.get("actions", resp)
        current_K = int(resp.get("chunk_size", 0)) or _infer_chunk_size(current_actions)
        current_hash = small_hash_for_chunk(current_actions)

        chunk_id = 0
        t_cursor = 0
        global_action_timestep = 0
        started_async_this_chunk = False

        staged_resp: Optional[dict] = None
        staged_hash: Optional[str] = None
        staged_K: int = 0
        staged_H_at_start: int = 0
        staged_global_timestep_at_start: int = 0

        frames = 0
        while frames < EACH_EPISODE_LEN:
            frame_start = time.perf_counter()

            remaining_in_chunk = current_K - t_cursor
            if (remaining_in_chunk <= LEFT_ACTION_THRESHOLD) and (not started_async_this_chunk) and (not fetcher.in_flight()) and (staged_resp is None):
                obs_rgba = camera.get_rgba()
                img = obs_rgba[:, :, :3]
                joints = gr1.get_joint_positions()
                fetcher.start(
                    gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints),
                    H_at_start=t_cursor,
                    global_timestep_at_start=global_action_timestep
                )
                started_async_this_chunk = True
                print(f"[PREFETCH] Triggered at t={t_cursor}. Remaining={remaining_in_chunk}")

            r, h, Hs, Gs = fetcher.take_if_ready()
            if r is not None and h is not None:
                if h != current_hash:
                    staged_resp = r
                    staged_hash = h
                    actions = r.get("actions", r)
                    staged_K = int(r.get("chunk_size", 0)) or _infer_chunk_size(actions)
                    staged_H_at_start = int(Hs)
                    staged_global_timestep_at_start = int(Gs)

            if staged_resp is not None and staged_hash is not None and staged_hash != current_hash:
                steps_passed_since_launch = global_action_timestep - staged_global_timestep_at_start
                s = max(0, min(steps_passed_since_launch + CHUNK_SWITCH_OFFSET, staged_K - 1))
                
                print(f"[PREEMPT] Chunk {chunk_id} -> {chunk_id+1}. Offset s={s}")
                chunk_id += 1
                current_actions = staged_resp.get("actions", staged_resp)
                current_K = staged_K
                current_hash = staged_hash
                t_cursor = s
                started_async_this_chunk = False
                staged_resp = None
                staged_hash = None
                staged_K = 0

            if 0 <= t_cursor < current_K:
                jp = actions_dict_to_joint_positions(current_actions, t_cursor, full_dof=dof)
                jp = jp[:dof]
                
                ctrl.apply_action(ArticulationAction(joint_positions=jp))
                world.step(render=True)
                
                # [RECORD ACTUAL VISUAL STATE]
                evaluator.update(gr1.get_joint_positions())
                
                t_cursor += 1
                global_action_timestep += 1
            else:
                if staged_resp is not None and staged_hash is not None and staged_hash != current_hash:
                    steps_passed_since_launch = global_action_timestep - staged_global_timestep_at_start
                    s = max(0, min(steps_passed_since_launch + CHUNK_SWITCH_OFFSET, staged_K - 1))
                    print(f"[EXHAUST] Switching to staged. Offset s={s}")
                    chunk_id += 1
                    current_actions = staged_resp.get("actions", staged_resp)
                    current_K = staged_K
                    current_hash = staged_hash
                    t_cursor = s
                    started_async_this_chunk = False
                    staged_resp = None
                    staged_hash = None
                    staged_K = 0
                    
                    world.step(render=True)
                    evaluator.update(gr1.get_joint_positions())

                else:
                    print(f"[SYNC] Exhausted. Fetching new...")
                    t_start_sync = time.perf_counter()
                    
                    obs_rgba = camera.get_rgba()
                    img = obs_rgba[:, :, :3]
                    joints = gr1.get_joint_positions()
                    next_resp = post_json("/inference", gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints))
                    next_actions = next_resp.get("actions", next_resp)
                    next_K = int(next_resp.get("chunk_size", 0)) or _infer_chunk_size(next_actions)
                    next_hash = small_hash_for_chunk(next_actions)
                    
                    t_end_sync = time.perf_counter()
                    sync_dur = t_end_sync - t_start_sync
                    skipped = int(sync_dur * ENV_FPS)
                    
                    # [PADDING] Inject stationary frames to penalize sync wait
                    if skipped > 0:
                        current_real_pos = gr1.get_joint_positions()[:dof]
                        for _ in range(skipped):
                            evaluator.update(current_real_pos)

                    if next_hash == current_hash: 
                        try:
                            next_resp2 = post_json("/inference", gutils.make_gr00t_input(task=TASK, obs=img, joint_positions=joints))
                            next_actions2 = next_resp2.get("actions", next_resp2)
                            next_hash2 = small_hash_for_chunk(next_actions2)
                            if next_hash2 != current_hash:
                                next_actions, next_K, next_hash = next_actions2, int(next_resp2.get("chunk_size", 0)), next_hash2
                        except: pass

                    if next_hash != current_hash:
                        chunk_id += 1
                        current_actions = next_actions
                        current_K = next_K
                        current_hash = next_hash
                        t_cursor = 0
                        started_async_this_chunk = False
                    
                    world.step(render=True)
                    evaluator.update(gr1.get_joint_positions())

            obs_rgba = camera.get_rgba()
            img = obs_rgba[:, :, :3]
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            frames += 1
            dt = time.perf_counter() - frame_start
            time.sleep(max(0.0, ENV_DT - dt))

        evaluator.compute_and_save(ep, METRIC_LOG_FILE)
        print(f"Episode {ep} finished")

    video.release()
    simulation_app.close()
    print(f"[INFO] Metrics saved to: {METRIC_LOG_FILE}")

if __name__ == "__main__":
    main()
