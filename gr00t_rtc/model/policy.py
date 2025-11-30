import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    GR00T 모델 래퍼. (비동기 처리는 상위 레벨에서 큐/스레드로 처리)
    - 여기서는 액션 청크를 항상 (chunk, dim) 형태의 torch.Tensor로 보장하고
      딕셔너리 {"action.<part>": np.ndarray[chunk, dof]} 로 변환해 돌려주도록 정리.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        try:
            model_path = snapshot_download(model_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(f"Model not found on HF. Loading locally: {model_path}")

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()
        self.model_path = Path(model_path)
        self.device = device

        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        self._load_model(model_path)
        self._load_metadata(self.model_path / "experiment_cfg")
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(self.model.action_head, "num_inference_timesteps"):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"[Gr00tPolicy] Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력: (B?, T, ...) 또는 (T, ...) 형태 딕셔너리
        출력: {"action.<part>": np.ndarray [T_action, dof]}
        """
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        print()
        print("[policy] observation (+current action):", obs_copy.keys())
        normalized_input = self.apply_transforms(obs_copy)
        # print("[policy] normalized input:", normalized_input.keys())
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        print("[policy] output action (normalized):", normalized_action.shape)
        print("[policy] output action (normalized) - relevant joint:", normalized_action[0, : ,0])
        print("[policy] output action (normalized) - irrelevant joint:", normalized_action[0, : ,-1])    # Only 26 out of 32 joint outputs seem to be valid
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        print("[policy] output action:", unnormalized_action.keys())

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)
        # 기대: model_pred["action_pred"] => (B=1, chunk, dim) or (chunk, dim)
        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        # 입력 텐서를 {"action": tensor}로 포장해 unapply 후 딕셔너리 반환
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()

        expected_action_horizon = len(self._modality_config["action"].delta_indices)
        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} "
                f"(was {model.action_head.config.action_horizon})"
            )
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon
            from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead
            new_action_head = FlowmatchingActionHead(new_action_head_config)
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
            model.action_head = new_action_head
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore
        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"Check {metadata_path}",
            )
        metadata = DatasetMetadata.model_validate(metadata_dict)
        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            assert np.all(np.diff(delta_indices) == delta_indices[1] - delta_indices[0]), f"{delta_indices=}"
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


# ----------------- Helpers -----------------
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            out[k] = np.expand_dims(np.array(v), axis=0)
        elif isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = np.squeeze(v, axis=0)
        elif isinstance(v, torch.Tensor):
            out[k] = v.squeeze(0)
        else:
            out[k] = v
    return out
