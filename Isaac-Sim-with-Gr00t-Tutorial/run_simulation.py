# # this uses the issacsim conda environment

# from isaacsim import SimulationApp


# # here are the parameters
# EPISODE_NUM = 2
# EACH_EPISODE_LEN = 30
# RESULT_VIDEO_FILE = "./results/NutPouring_batch32_nodiffusion.mp4"
# LOAD_WORLD_FILE = "./sim_environments/gr1_NutPouring.usd"
# #LOAD_WORLD_FILE = "./sim_environments/gr1_exhaust_pipe.usd"

# TASK = "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. Pick up the yellow bowl and place it on the metallic measuring scale."
# #TASK = "Pickup the blue pipe and place it into the blue bin." # Exhaust Pipe task

# # setting for the camera
# CAMERA_HEIGHT = 200 # width is fixed to 256
# CAMERA_FOCAL_LENGTH = 1.2
# CAMERA_FORWARD_DIST = 0.25
# CAMERA_ANGLE = 70 

# INFERENCE_SERVER_URL = "http://localhost:9876/inference"


# simulation_app = SimulationApp({
#     "headless": True, 
#     "create_new_stage": False,
#     "open_usd" : LOAD_WORLD_FILE,
#     "sync_loads": True, # wait until asset loads
# }) 

# from isaacsim.core.api import World
# from isaacsim.core.api.scenes.scene import Scene
# from isaacsim.core.api.robots import Robot
# from isaacsim.core.api.controllers.articulation_controller import ArticulationController
# from isaacsim.sensors.camera.camera import Camera
# import numpy as np
# import isaacsim.core.utils.numpy.rotations as rot_utils
# from isaacsim.core.prims import XFormPrim, RigidPrim
# from isaacsim.core.utils.types import ArticulationAction
# import cv2
# import gr1_config, gr1_gr00t_utils





# def main():
#     ## 1. setup scene
#     print("## 1. setup scene")
#     world = World()
#     scene: Scene = world.scene
#     gr1: Robot = scene.add(Robot(
#         prim_path="/World/gr1", 
#         name="gr1",
#     ))
#     gr1_articulation_controller: ArticulationController = gr1.get_articulation_controller()
    
#     # adding camera
#     camera = Camera(
#         prim_path="/World/gr1/head_yaw_link/camera",
#         name="camera",
#         translation=np.array([CAMERA_FORWARD_DIST, 0.0, 0.07]),
#         frequency=60,
#         resolution=(256, CAMERA_HEIGHT),
#         orientation=rot_utils.euler_angles_to_quats(np.array([0, CAMERA_ANGLE, 0]), degrees=True),
#     )
#     camera.set_focal_length(CAMERA_FOCAL_LENGTH) # smaller => wider range of view
#     camera.set_clipping_range(0.1, 2)
    
    
#     ## 2. setup_post_load
#     world.reset()
#     print("## 2. setup post-load")
#     camera.initialize()
#     camera.add_motion_vectors_to_frame()
#     gr1_articulation_controller.set_gains(kps = np.array([3000.0]*54), kds = np.array([100.0]*54)) # p is the stiffness, d is the gain
    
    
#     ## 3. run simulation
#     print("## 3. run simulation")
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
#     video = cv2.VideoWriter(RESULT_VIDEO_FILE, fourcc, 30, (256, CAMERA_HEIGHT), isColor=True)
    
#     for episode_idx in range(EPISODE_NUM):
#         print(f"Starting episode {episode_idx}")
#         world.reset()
        
#         # set gr1 default position
#         gr1.set_joint_positions(positions=gr1_config.default_joint_position)
        
#         # first, just initialize the world and wait
#         print("Waiting to initialize")
#         for step in range(100):
#             world.step(render=True)
            
#         # for the actual simulation
#         print("Start episode")
#         for step in range(EACH_EPISODE_LEN):
#             world.step(render=True)
#             obs: np.ndarray = camera.get_rgba()
#             image = obs[:, :, :3]
#             video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 
#             current_joint_positions = gr1.get_joint_positions()
            
#             # inference to gr00t server
#             print(f"Episode {episode_idx} step {step} calling gr00t inference")        
#             gr00t_inference_input = gr1_gr00t_utils.make_gr00t_input(task=TASK, obs=image, joint_positions=current_joint_positions)
#             gr00t_output = gr1_gr00t_utils.request_gr00t_inference(payload=gr00t_inference_input, url=INFERENCE_SERVER_URL)
#             for timestep in range(0, 16):
#                 action_joint_position = gr1_gr00t_utils.make_joint_position_from_gr00t_output(gr00t_output, timestep=timestep)
#                 gr1_articulation_controller.apply_action(ArticulationAction(joint_positions=action_joint_position))
#                 if timestep == 15: break # at the end, do not step, as it will be done by the outer loop
#                 world.step(render=True)
#                 obs: np.ndarray = camera.get_rgba()
#                 image = obs[:, :, :3]
#                 video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            
 
        
#         print(f"Episode {episode_idx} finished")
        
#     video.release()
#     simulation_app.close()

    
# if __name__ == "__main__":
#     main()
































# this uses the issacsim conda environment

from isaacsim import SimulationApp

# ==== 시뮬 파라미터 ====
EPISODE_NUM = 1
EACH_EPISODE_LEN = 600
RESULT_VIDEO_FILE = "./results/NutPouring_batch32_600a1.mp4"
LOAD_WORLD_FILE = "./sim_environments/gr1_NutPouring.usd"

TASK = "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. Pick up the yellow bowl and place it on the metallic measuring scale."

# 카메라 설정
CAMERA_HEIGHT = 200  # width는 256 고정
CAMERA_FOCAL_LENGTH = 1.2
CAMERA_FORWARD_DIST = 0.25
CAMERA_ANGLE = 70

INFERENCE_SERVER_URL = "http://localhost:9876/inference"

# ==== 비동기 동작 파라미터 ====
CHUNK_SIZE_THRESHOLD = 0.1   # 큐 크기 / 청크 크기 <= 임계치일 때 다음 옵저베이션 전송
AGGREGATE_MODE = "latest_only"  # "latest_only" | "average" | "weighted_average" | "conservative"

simulation_app = SimulationApp({
    "headless": True,
    "create_new_stage": False,
    "open_usd": LOAD_WORLD_FILE,
    "sync_loads": True,
})

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.sensors.camera.camera import Camera
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.types import ArticulationAction
import cv2
import threading
import time
from queue import Queue, Empty
from dataclasses import dataclass

import gr1_config
import gr1_gr00t_utils as utils


# ------------------ Data structures ------------------
@dataclass
class ObsPackage:
    task: str
    image_rgb: np.ndarray        # (H, W=256, 3) uint8
    joint_positions: np.ndarray  # (54,)
    latest_timestep: int
    ts: float


# ------------------ Async Client State ------------------
class AsyncClientState:
    def __init__(self):
        self.action_queue: Queue[utils.TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_timestep = -1
        self.action_chunk_size = 0
        self.shutdown = threading.Event()
        self.must_go = threading.Event()
        self.must_go.set()  # 처음엔 반드시 보냄

        # 메인→백그라운드로 관측 전달용 (메인만 카메라/월드 접근)
        self.obs_queue: Queue[ObsPackage] = Queue(maxsize=1)

    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def queue_size(self) -> int:
        with self.action_queue_lock:
            return self.action_queue.qsize()

    def pop_action(self) -> utils.TimedAction | None:
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None
            return self.action_queue.get_nowait()

    def push_actions(self, actions: list[utils.TimedAction]):
        with self.action_queue_lock:
            # 동일 timestep이 이미 있으면 aggregate
            existing = {a.timestep: a for a in list(self.action_queue.queue)}
            new_queue = Queue()
            for a in actions:
                if a.timestep <= self.latest_timestep:
                    continue
                if a.timestep in existing:
                    merged = utils.aggregate_actions(existing[a.timestep].action, a.action, mode=AGGREGATE_MODE)
                    new_queue.put(utils.TimedAction(a.timestamp, a.timestep, merged))
                else:
                    new_queue.put(a)
            self.action_queue = new_queue


# ------------------ Receiver thread (no world/camera access here) ------------------
def action_receiver_thread_fn(state: AsyncClientState):
    """
    - 메인 쓰레드가 obs_queue에 넣어준 관측 패키지를 받아
    - HTTP inference 요청 → 액션 청크를 TimedAction 리스트로 변환하여 action_queue에 적재
    """
    while not state.shutdown.is_set():
        try:
            obs_pkg: ObsPackage = state.obs_queue.get(timeout=0.05)
        except Empty:
            continue
        try:
            payload = utils.make_gr00t_input(
                task=obs_pkg.task, obs=obs_pkg.image_rgb, joint_positions=obs_pkg.joint_positions
            )
            t0 = time.time()
            resp = utils.request_gr00t_inference(payload, url=INFERENCE_SERVER_URL)
            actions_json = resp.get("actions", {})
            chunk_size = int(resp.get("chunk_size", 0)) or 1
            state.action_chunk_size = max(state.action_chunk_size, chunk_size)

            chunk_arrays = utils.parse_actions_json_to_chunk(actions_json)
            timed_actions = utils.make_timed_chunk(
                chunk_arrays, start_ts=t0, start_timestep=obs_pkg.latest_timestep
            )
            state.push_actions(timed_actions)
            state.must_go.set()  # 큐가 비면 다음 관측은 must-go
        except Exception as e:
            print(f"[Receiver] Error: {e}")
            time.sleep(0.05)


def main():
    # 1) 씬/로봇/카메라 설정 (메인 쓰레드 전용)
    print("## 1. setup scene")
    world = World()
    scene: Scene = world.scene
    gr1: Robot = scene.add(Robot(prim_path="/World/gr1", name="gr1"))
    controller: ArticulationController = gr1.get_articulation_controller()

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

    # 2) 초기화 (메인)
    world.reset()
    print("## 2. setup post-load")
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    controller.set_gains(kps=np.array([3000.0] * 54), kds=np.array([100.0] * 54))

    # 3) 비디오 초기화 (메인)
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    video = cv2.VideoWriter(RESULT_VIDEO_FILE, fourcc, 30, (256, CAMERA_HEIGHT), isColor=True)

    state = AsyncClientState()

    # 백그라운드 스레드 시작 (월드/카메라 접근 X)
    receiver_th = threading.Thread(target=action_receiver_thread_fn, args=(state,), daemon=True)
    receiver_th.start()

    for ep in range(EPISODE_NUM):
        print(f"Starting episode {ep}")
        world.reset()
        gr1.set_joint_positions(positions=gr1_config.default_joint_position)

        # 안정화 워밍업 (메인에서만 step)
        for _ in range(60):
            world.step(render=True)

        steps = 0
        while steps < EACH_EPISODE_LEN:
            loop_start = time.perf_counter()

            # (1) 큐에서 액션 하나를 꺼내 적용 (메인)
            act = state.pop_action()
            if act is not None:
                controller.apply_action(ArticulationAction(joint_positions=act.action))
                state.latest_timestep = max(state.latest_timestep, act.timestep)

            # (2) 시뮬 스텝 + 센서 캡처 + 비디오 기록 (모두 메인)
            world.step(render=True)
            obs_rgba = camera.get_rgba()
            img = obs_rgba[:, :, :3].astype(np.uint8)
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # (3) 임계치 기반 관측 전송 트리거 (메인→백그라운드 큐)
            qsz = state.queue_size()
            chunk_size = max(1, state.action_chunk_size)
            ready_to_send = (qsz / chunk_size) <= CHUNK_SIZE_THRESHOLD
            if (ready_to_send or (state.must_go.is_set() and qsz == 0)) and state.obs_queue.empty():
                try:
                    obs_pkg = ObsPackage(
                        task=TASK,
                        image_rgb=img,  # 메인에서 캡처한 이미지 복사/전달
                        joint_positions=gr1.get_joint_positions(),
                        latest_timestep=state.latest_timestep,
                        ts=time.time(),
                    )
                    state.obs_queue.put_nowait(obs_pkg)
                    state.must_go.clear()  # 전송했으니 must_go 클리어; 큐가 비면 receiver가 다시 set
                except Exception:
                    pass  # obs_queue가 가득이면 다음 루프로

            # (4) 한 스텝 종료 처리
            steps += 1

            # busy 루프 방지 슬립
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, 0.001 - elapsed))

        print(f"Episode {ep} finished")

    # 종료 처리
    state.shutdown.set()
    receiver_th.join(timeout=2.0)
    video.release()
    simulation_app.close()


if __name__ == "__main__":
    main()

