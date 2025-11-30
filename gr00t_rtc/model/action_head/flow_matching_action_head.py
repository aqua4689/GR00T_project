# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer

# from tmp_debug_code_4 import acc_jacobian


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    # @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        print("[action head] action", action_input.action.shape, action_input.action[0, :, 0])
        print("[action head] action mask", action_input.action_mask[0, :, 0])

        torch.set_grad_enabled(True)

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            # t_discretized = int(t_cont * self.num_timestep_buckets)

            # # Embed noised action trajectory.
            # timesteps_tensor = torch.full(
            #     size=(batch_size,), fill_value=t_discretized, device=device
            # )
            # action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # # Maybe add position embedding.
            # if self.config.add_pos_embed:
            #     pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            #     pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            #     action_features = action_features + pos_embs

            # # Join vision, language, state and action embedding along sequence dimension.
            # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # # Run model forward.
            # model_output = self.model(
            #     hidden_states=sa_embs,
            #     encoder_hidden_states=vl_embs,
            #     timestep=timesteps_tensor,
            # )
            # pred = self.action_decoder(model_output, embodiment_id)

            # pred_velocity = pred[:, -self.action_horizon :]

            # The commented codes above are abstracted into _action_to_flow method to calcualte gradients
            v = lambda actions: self._action_to_flow(actions, vl_embs, state_features, embodiment_id, t, num_steps, batch_size, device)
            pred_velocity = v(actions)

            if not action_input.action_mask.any():    # previous action is not input
                # Update actions using euler integration.
                actions = actions + dt * pred_velocity


                # # Acceleration guidance ver.
                # valid_action_dimension = action_input.action_mask[0, 0, :].sum().item()

                # # Calculate the error term for guidance
                # onestep_prediction = actions + (1 - t_cont) * pred_velocity

                # # Calculate the guidance term including gradients
                # pred_velocity_Jacobian = torch.func.jacrev(v)(actions)
                # action_size = actions.shape[-2] * actions.shape[-1]
                # Jacobian_flatten = torch.eye(action_size, device=actions.device) + (1 - t_cont) * pred_velocity_Jacobian.reshape(action_size, action_size)    # assumes that batch size is always 1 during inference (may need modification later)

                # # New component - regularize with respect to acceleration norm
                # acc_norm = lambda traj: self._traj_to_acc_norm(traj, valid_action_dimension)
                # acc_grad = torch.func.jacrev(acc_norm)(onestep_prediction)
                # acc_grad_flatten = acc_grad.reshape(action_size)
                # acceleration_guidance = (Jacobian_flatten @ acc_grad_flatten).view(*acc_grad.shape)

                # # Update actions using euler integration.
                # actions = actions + dt * (pred_velocity - 1 * acceleration_guidance)
                # print("[action head] guidance scales:", torch.linalg.norm(pred_velocity[:, :, :26]), torch.linalg.norm(acceleration_guidance[:, :, :26]))

            else:
                # Previous actions
                prev_action = action_input.action
                prev_action_length = action_input.action_mask[0, :, 0].sum().item()
                valid_action_dimension = action_input.action_mask[0, 0, :].sum().item()
                if t == 0:
                    print("[action head] action mask II", prev_action_length)
                    print("[action head] action mask III", valid_action_dimension)

                # Define the weight matrix (subject to change - may be given as an argument later)
                guidance_weight = torch.zeros(pred_velocity.shape[1]).to(actions.device)
                horizon_full_guidance = min(prev_action_length, 4)
                horizon_partial_guidance = max(0, prev_action_length - 4)
                guidance_weight[:horizon_full_guidance] = 1
                interp_term = torch.arange(horizon_partial_guidance, 0, -1) / (horizon_partial_guidance + 1)
                guidance_weight[horizon_full_guidance : horizon_full_guidance+horizon_partial_guidance] = interp_term * (torch.exp(interp_term) - 1) / (torch.e - 1)

                # Calculate the error term for guidance
                onestep_prediction = actions + (1 - t_cont) * pred_velocity
                error = torch.diag(guidance_weight).unsqueeze(0) @ (prev_action - onestep_prediction)

                # Calculate the guidance term including gradients
                print("[action head] debug (requires_grad - actions):", actions.requires_grad)
                pred_velocity_Jacobian = torch.func.jacrev(v)(actions)
                action_size = actions.shape[-2] * actions.shape[-1]
                Jacobian_flatten = torch.eye(action_size, device=actions.device) + (1 - t_cont) * pred_velocity_Jacobian.reshape(action_size, action_size)    # assumes that batch size is always 1 during inference (may need modification later)
                error_flatten = error.reshape(action_size)
                guidance_term = (Jacobian_flatten @ error_flatten).view(*error.shape)

                # Guidance coefficient
                guidance_coeff = (1-t_cont)**2 / (t_cont**2 + (1-t_cont)**2)
                guidance_coeff = (1-t_cont) / (t_cont * guidance_coeff + 1e-8)
                if guidance_coeff > 5:    # beta=5, following the paper
                    guidance_coeff = 5

                # New component - regularize with respect to acceleration norm
                # print("[action head] debug:", onestep_prediction.shape, valid_action_dimension)
                # print("[action head] debug (requires_grad) I:", onestep_prediction.requires_grad)
                # acc_norm = lambda onestep_prediction: self._traj_to_acc_norm(onestep_prediction, valid_action_dimension)

                # onestep_prediction.requires_grad = True
                # onestep_prediction = torch.randn(1, 16, 32, requires_grad=True).to(onestep_prediction.device)
                # acc_grad = torch.func.jacrev(acc_norm)(onestep_prediction)
                # with torch.enable_grad():
                #     # x = onestep_prediction.detach().clone().requires_grad_(True)
                #     # x = onestep_prediction.clone().requires_grad_(True)
                #     # x = onestep_prediction.detach().clone().contiguous().requires_grad_(True)
                #     # x = onestep_prediction.clone().contiguous().requires_grad_(True)
                #     x = torch.randn(1, 16, 32, requires_grad=True)  # debug input
                #     print("[action head] grad enabled I:", torch.is_grad_enabled())
                #     print("[action head] debug (requires_grad) II:", id(x), x.requires_grad, x.grad_fn, x.is_leaf)
                #     acc_val = acc_norm(x)   # scalar tensor
                #     print("[action head] debug - acc_norm", acc_val)
                #     # If acc_val is literally zero by definition, gradient will be zero; check that earlier if unexpected
                #     acc_grad = torch.autograd.grad(
                #         outputs=acc_val,
                #         inputs=x,
                #         # grad_outputs=torch.ones_like(acc_val),
                #         # create_graph=False,  # True only if you need higher-order grads
                #         # retain_graph=False
                #     )[0]  # shape (1,16,32)

                # acc_grad = torch.func.grad(self._traj_to_acc_norm, argnums=0)(
                #     onestep_prediction, valid_action_dimension
                # )

                # x = torch.randn(1, 16, 32, requires_grad=True, device=onestep_prediction.device)
                # # acc_grad = torch.func.grad(acc_norm, argnums=0)(onestep_prediction)
                # acc_grad = torch.func.grad(acc_norm, argnums=0)(x)

                # acc_grad = acc_jacobian(onestep_prediction, valid_action_dimension)

                # print("[action head] debug:", acc_grad.shape, torch.linalg.norm(acc_grad[:, :, :26]))
                # print("[action head] debug:", acc_grad[0, :, 0])

                # acc_grad_flatten = acc_grad.reshape(action_size)
                # acceleration_guidance = (Jacobian_flatten @ acc_grad_flatten).view(*acc_grad.shape)

                # Update actions using euler integration.
                actions = actions + dt * (pred_velocity + guidance_coeff * guidance_term)
                # actions = actions + dt * (pred_velocity + guidance_coeff * guidance_term - 1 * acceleration_guidance)
                # print("[action head] guidance scales:", torch.linalg.norm(pred_velocity[:, :, :26]), torch.linalg.norm(guidance_term[:, :, :26]), torch.linalg.norm(acceleration_guidance[:, :, :26]))

        return BatchFeature(data={"action_pred": actions})

    def _action_to_flow(self, actions, vl_embs, state_features, embodiment_id, t, num_steps, batch_size, device):
        t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
        t_discretized = int(t_cont * self.num_timestep_buckets)

        # Embed noised action trajectory.
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device
        )
        action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        # Run model forward.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred = self.action_decoder(model_output, embodiment_id)

        pred_velocity = pred[:, -self.action_horizon :]
        return pred_velocity

    # def _traj_to_acc_norm(self, onestep_prediction, valid_action_dimension):
    #     # return (traj[:, 2:, :valid_action_dimension] - 2*traj[:, 1:-1, :valid_action_dimension] + traj[:, :-2, :valid_action_dimension]).pow(2).sum()
    #     print("[action head] grad enabled II:", torch.is_grad_enabled())
    #     print("[action head] - acc method", id(onestep_prediction), onestep_prediction.requires_grad, onestep_prediction.grad_fn, onestep_prediction.is_leaf)
    #     diff_2 = onestep_prediction[:, 2:, :valid_action_dimension] - 2*onestep_prediction[:, 1:-1, :valid_action_dimension] + onestep_prediction[:, :-2, :valid_action_dimension]
    #     print("[action head] - acc method", id(diff_2), diff_2.requires_grad, diff_2.grad_fn, diff_2.is_leaf, diff_2.shape)
    #     return (diff_2 ** 2).sum()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
