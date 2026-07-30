# Copyright 2026 Applied Intuition, Inc.
# SPDX-License-Identifier: CC-BY-NC-4.0

import types
from typing import List, Optional, Tuple, Union
import torch

from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
import os
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.filesystem import FileSystemReader

# from settings import MODEL_FOLDER
MODEL_FOLDER = None  # Set via config: text_encoder_path / vae_path, or wan_model_folder


def _select_chunk_prompt_embeds(
    prompt_embeds: torch.Tensor,
    conditional_dict: dict,
    *,
    current_frames: int,
    freqs_offset: int,
) -> torch.Tensor:
    """Select chunk-local text contexts for one model invocation.

    ReMind keeps one context per chunk. Streaming inference slices the
    pre-encoded schedule by absolute RoPE frame position. Partial or
    non-contiguous history passes expand to one context per frame so prompt
    boundaries remain exact.
    """
    if prompt_embeds.ndim != 4:
        return prompt_embeds
    prompt_chunk_size = int(conditional_dict.get("prompt_chunk_size", 0) or 0)
    if prompt_chunk_size <= 0:
        raise ValueError(
            "4D prompt_embeds require conditional_dict['prompt_chunk_size']"
        )
    num_prompt_chunks = prompt_embeds.shape[1]
    explicit_frames = conditional_dict.get("prompt_frame_indices")
    if explicit_frames is not None:
        frame_indices = torch.as_tensor(
            explicit_frames, device=prompt_embeds.device, dtype=torch.long
        )
        if frame_indices.ndim != 1 or frame_indices.numel() != current_frames:
            raise ValueError(
                "prompt_frame_indices must contain one absolute index per "
                f"current frame; got {tuple(frame_indices.shape)} for "
                f"current_frames={current_frames}"
            )
        chunk_indices = torch.div(
            frame_indices, prompt_chunk_size, rounding_mode="floor"
        )
        if (
            int(chunk_indices.min().item()) < 0
            or int(chunk_indices.max().item()) >= num_prompt_chunks
        ):
            raise ValueError(
                f"prompt frame indices map outside {num_prompt_chunks} chunks"
            )
        return prompt_embeds.index_select(1, chunk_indices)

    start_frame = int(freqs_offset)
    stop_frame = start_frame + int(current_frames)
    if start_frame % prompt_chunk_size == 0 and current_frames % prompt_chunk_size == 0:
        start_chunk = start_frame // prompt_chunk_size
        stop_chunk = stop_frame // prompt_chunk_size
        if start_chunk >= 0 and stop_chunk <= num_prompt_chunks:
            return prompt_embeds[:, start_chunk:stop_chunk]

    frame_indices = torch.arange(
        start_frame, stop_frame, device=prompt_embeds.device, dtype=torch.long
    )
    chunk_indices = torch.div(frame_indices, prompt_chunk_size, rounding_mode="floor")
    if (
        int(chunk_indices.min().item()) < 0
        or int(chunk_indices.max().item()) >= num_prompt_chunks
    ):
        raise ValueError(
            f"prompt range [{start_frame}, {stop_frame}) maps outside "
            f"{num_prompt_chunks} chunks of size {prompt_chunk_size}"
        )
    return prompt_embeds.index_select(1, chunk_indices)


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_folder: str) -> None:
        super().__init__()

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("meta"),
            )
            .eval()
            .requires_grad_(False)
        )
        self.text_encoder.to_empty(device="cpu")

        safetensors_path = os.path.join(
            model_folder, "models_t5_umt5-xxl-enc-bf16.safetensors"
        )
        pth_path = os.path.join(model_folder, "models_t5_umt5-xxl-enc-bf16.pth")
        if os.path.exists(safetensors_path):
            state_dict = safe_load_file(safetensors_path)
        elif os.path.exists(pth_path):
            state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"Missing T5 weights in {model_folder}: expected "
                f"{os.path.basename(safetensors_path)} or {os.path.basename(pth_path)}"
            )
        self.text_encoder.load_state_dict(state_dict)

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(model_folder, "google", "umt5-xxl/"),
            seq_len=512,
            clean="whitespace",
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(
        self,
        text_prompts: Union[List[str], List[List[str]]],
    ) -> dict:
        nested = bool(text_prompts and isinstance(text_prompts[0], (list, tuple)))
        if nested:
            chunk_counts = [len(prompts) for prompts in text_prompts]
            if not chunk_counts or min(chunk_counts) <= 0:
                raise ValueError("chunk-local text prompts cannot be empty")
            if len(set(chunk_counts)) != 1:
                raise ValueError(
                    "all samples must provide the same number of chunk prompts; "
                    f"got {chunk_counts}"
                )
            flat_prompts = [
                str(prompt) for prompts in text_prompts for prompt in prompts
            ]
        else:
            flat_prompts = [str(prompt) for prompt in text_prompts]

        # Chunk-local prompts repeat the base caption on every non-event
        # chunk; encode each UNIQUE string once and scatter back (typically
        # 7 prompts -> 3-4 unique, ~2x cheaper umt5-xxl pass).
        unique_prompts = list(dict.fromkeys(flat_prompts))
        index_of = {p: i for i, p in enumerate(unique_prompts)}
        gather_idx = [index_of[p] for p in flat_prompts]

        ids, mask = self.tokenizer(
            unique_prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0
        if len(unique_prompts) != len(flat_prompts):
            context = context[
                torch.as_tensor(gather_idx, device=context.device, dtype=torch.long)
            ]
        if nested:
            batch_size = len(text_prompts)
            num_chunks = chunk_counts[0]
            context = context.view(
                batch_size, num_chunks, context.shape[1], context.shape[2]
            )
        result = {"prompt_embeds": context}
        return result


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_folder: str):
        super().__init__()
        wan22_vae_path = os.path.join(model_folder, "Wan2.2_VAE.pth")
        if os.path.exists(wan22_vae_path):
            from wan.modules.vae_wan22 import WanVideoVAE38

            vae = WanVideoVAE38()
            state_dict = torch.load(
                wan22_vae_path, map_location="cpu", weights_only=True
            )
            if state_dict and next(iter(state_dict)).startswith("model."):
                vae.load_state_dict(state_dict, strict=True)
            else:
                vae.model.load_state_dict(state_dict, strict=True)
            self.mean = vae.mean.to(dtype=torch.float32)
            self.std = vae.std.to(dtype=torch.float32)
            self.model = vae.model.eval().requires_grad_(False)
            self.z_dim = int(vae.z_dim)
            self.upsampling_factor = int(vae.upsampling_factor)
            print(
                f"WanVAEWrapper loaded {wan22_vae_path} "
                f"(z_dim={self.z_dim}, upsampling_factor={self.upsampling_factor})"
            )
            return

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        vae_path = os.path.join(model_folder, "Wan2.1_VAE.pth")
        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )
        self.z_dim = 16
        self.upsampling_factor = 8
        print(
            f"WanVAEWrapper loaded {vae_path} "
            f"(z_dim={self.z_dim}, upsampling_factor={self.upsampling_factor})"
        )

    def forward(
        self, x: torch.Tensor, method: str = "encode", **kwargs
    ) -> torch.Tensor:
        if method == "encode":
            return self.encode_to_latent(x)
        elif method == "decode":
            return self.decode_to_pixel(x, **kwargs)
        else:
            raise ValueError(f"Unknown method {method}")

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel
        ]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(
                decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


def load_state_dict_from_folder_safetensors(file_path):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if (
            "." in file_name
            and "diffusion" in file_name
            and file_name.split(".")[-1] in ["safetensors"]
        ):
            state_dict.update(safe_load_file(os.path.join(file_path, file_name)))
    return state_dict


def _filter_state_dict_keys(state_dict, skip_substrings):
    """
    Filter out (do not load) weights whose keys contain any of `skip_substrings`.
    Returns (filtered_state_dict, skipped_keys).
    """
    if not skip_substrings:
        return state_dict, []
    skipped = []
    filtered = {}
    for k, v in state_dict.items():
        if any(s in k for s in skip_substrings):
            skipped.append(k)
            continue
        filtered[k] = v
    return filtered, skipped


def _slice_prefix_tensor_for_live_shape(state_dict, key, live_tensor, label):
    """Adapt a pretrained tensor to the live module shape when safe.

    Wan2.2-TI2V-5B ships a 48-channel input/output head. The ReMind
    continuous-latent training target is still 16 Wan VAE channels, so 5B
    i2v16 configs instantiate smaller patch/head tensors and keep the prefix
    rows/channels from the pretrained checkpoint.
    """
    tensor = state_dict.get(key)
    if tensor is None or live_tensor is None:
        return
    live_shape = tuple(live_tensor.shape)
    ckpt_shape = tuple(tensor.shape)
    if ckpt_shape == live_shape:
        return
    if tensor.dim() == live_tensor.dim() == 5:
        if (
            ckpt_shape[0] == live_shape[0]
            and ckpt_shape[2:] == live_shape[2:]
            and ckpt_shape[1] >= live_shape[1]
        ):
            print(
                f"[{label} surgery] slicing {key} {list(ckpt_shape)} "
                f"-> {list(live_shape)} on input channels"
            )
            state_dict[key] = tensor[:, : live_shape[1]].contiguous()
            return
    if tensor.dim() == live_tensor.dim() == 2:
        if ckpt_shape[1] == live_shape[1] and ckpt_shape[0] >= live_shape[0]:
            print(
                f"[{label} surgery] slicing {key} {list(ckpt_shape)} "
                f"-> {list(live_shape)} on output rows"
            )
            state_dict[key] = tensor[: live_shape[0]].contiguous()
            return
    if tensor.dim() == live_tensor.dim() == 1:
        if ckpt_shape[0] >= live_shape[0]:
            print(
                f"[{label} surgery] slicing {key} {list(ckpt_shape)} "
                f"-> {list(live_shape)}"
            )
            state_dict[key] = tensor[: live_shape[0]].contiguous()
            return
    print(
        f"[{label} surgery] cannot adapt {key}: ckpt={list(ckpt_shape)} "
        f"live={list(live_shape)}"
    )


def dcp_load_dict(path):
    if path.endswith(".safetensors"):
        auto_state_dict = safe_load_file(path)
        state_dict = {}
        for key, value in auto_state_dict.items():
            # Remove FSDP wrapper prefix if present
            if "._fsdp_wrapped_module." in key:
                key = key.replace("._fsdp_wrapped_module.", ".")
            # Remove model. prefix if present
            if "model." in key:
                key = key.replace("model.", "")
            state_dict[key] = value
        return state_dict

    safe_file_path = path + "/model.safetensors"
    if os.path.exists(safe_file_path):
        state_dict = safe_load_file(safe_file_path)
    else:
        reader = FileSystemReader(path)
        metadata = reader.read_metadata()

        auto_state_dict = {}
        for key, entry in metadata.state_dict_metadata.items():
            auto_state_dict[key] = torch.empty(
                entry.size, dtype=entry.properties.dtype, device=torch.device("meta")
            )

        dcp.load(state_dict=auto_state_dict, storage_reader=reader, no_dist=True)
        state_dict = {}
        for key, value in auto_state_dict.items():
            # Remove FSDP wrapper prefix if present
            if "._fsdp_wrapped_module." in key:
                key = key.replace("._fsdp_wrapped_module.", ".")
            # Remove model. prefix if present
            if "model." in key:
                key = key.replace("model.", "")
            state_dict[key] = value
        safe_save_file(state_dict, safe_file_path)
    return state_dict


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name="Wan2.1-T2V-1.3B",
        load_path=None,
        timestep_shift=5.0,
        is_causal=False,
        ckpt_path=None,
        weight_list=[],
        filter_list=[],
        in_dim=36,
        out_dim=None,
        model_type=None,
        dual_model=False,
        high_noise_threshold=0.5,
        prope_temporal_dim=0,  # ProPE split: temporal RoPE dims (causal only)
        cc_rope_mode="dual_prope",  # RoPE variant: standard | dual_prope | cc_basic | cc_output | cc_dual_channel | cc_dual_output | prope_residual | cc_value | cc_full
        cc_phase_slots=16,  # dual_channel only: # freq slots dedicated to camera
        degradation_control_dim=0,
        degradation_control_hidden_dim=256,
        require_full_weight_coverage=False,
    ):
        super().__init__()
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        load_generator_on_all_ranks = os.environ.get(
            "REMIND_LOAD_GENERATOR_ON_ALL_RANKS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        num_threads = int(os.environ.get("TORCH_NUM_THREADS", "32"))
        if torch.get_num_threads() != num_threads:
            torch.set_num_threads(num_threads)

        # model_path: use the first weight_list path's directory as the model config source,
        # or fall back to model_name if weight_list is empty
        if weight_list:
            model_path = weight_list[0]["path"]
        else:
            model_path = model_name

        # Wan2.2 dual model: config.json is inside high_noise_model/ subdir
        config_path = model_path
        if dual_model and os.path.isdir(os.path.join(model_path, "high_noise_model")):
            config_path = os.path.join(model_path, "high_noise_model")

        # Initialize primary model
        if is_causal:
            config = CausalWanModel.load_config(config_path)
            config = dict(config)
            config["in_dim"] = in_dim
            if out_dim is not None:
                config["out_dim"] = out_dim
            if model_type is not None:
                config["model_type"] = model_type
            config["prope_temporal_dim"] = prope_temporal_dim
            config["cc_rope_mode"] = cc_rope_mode
            config["cc_phase_slots"] = cc_phase_slots
            config["degradation_control_dim"] = degradation_control_dim
            config["degradation_control_hidden_dim"] = degradation_control_hidden_dim
            with torch.device("meta"):
                self.model = CausalWanModel(**config)
            self.model.to_empty(device="cpu")
            self._cc_rope_mode = cc_rope_mode
        else:
            config = WanModel.load_config(config_path)
            config = dict(config)
            config["in_dim"] = in_dim
            with torch.device("meta"):
                self.model = WanModel(**config)
            self.model.to_empty(device="cpu")

        # Initialize secondary model for dual-model mode (Wan2.2)
        self.model_2 = None
        self.dual_model = dual_model
        self.high_noise_threshold = high_noise_threshold

        if dual_model and not is_causal:
            # Use same config for model_2
            with torch.device("meta"):
                self.model_2 = WanModel(**config)
            self.model_2.to_empty(device="cpu")

        if rank == 0 or load_generator_on_all_ranks:
            if rank != 0 and load_generator_on_all_ranks:
                print(
                    f"[Rank {rank}] loading generator weights locally "
                    "because REMIND_LOAD_GENERATOR_ON_ALL_RANKS=1"
                )
            state_dict_full = None
            state_dict_full_2 = None  # For model_2
            primary_missing_keys = None

            if ckpt_path is not None:
                state_dict_full = dcp_load_dict(ckpt_path)
            else:
                for weight_config in weight_list:
                    weight_path = weight_config["path"]
                    is_model_2 = weight_config.get("is_model_2", False)
                    should_load_weights = weight_config.get("load_weights", True)
                    if isinstance(should_load_weights, str):
                        should_load_weights = should_load_weights.lower() not in {
                            "0",
                            "false",
                            "no",
                            "off",
                        }
                    if not should_load_weights:
                        print(
                            f"load_model {weight_path}: skipped weight load (load_weights=false)"
                        )
                        continue

                    # For Wan2.2 dual model: automatically determine high/low noise model
                    # based on directory structure if not explicitly specified
                    if (
                        dual_model
                        and not is_causal
                        and "is_model_2" not in weight_config
                    ):
                        # Check if path contains high/low noise indicators
                        if (
                            "high_noise" in weight_path.lower()
                            or "high" in os.path.basename(weight_path).lower()
                        ):
                            is_model_2 = False  # high noise -> primary model
                        elif (
                            "low_noise" in weight_path.lower()
                            or "low" in os.path.basename(weight_path).lower()
                        ):
                            is_model_2 = True  # low noise -> model_2

                    if os.path.isdir(weight_path):
                        state_dict = load_state_dict_from_folder_safetensors(
                            weight_path
                        )
                    else:
                        state_dict = safe_load_file(weight_path)

                    if is_model_2 and dual_model and not is_causal:
                        # This weight is for model_2 (low noise model in Wan2.2)
                        if state_dict_full_2 is None:
                            state_dict_full_2 = state_dict
                        else:
                            state_dict_full_2.update(state_dict)
                    else:
                        # This weight is for model (primary/high noise model)
                        if state_dict_full is None:
                            state_dict_full = state_dict
                        else:
                            state_dict_full.update(state_dict)

            # Load primary model
            if state_dict_full is not None:
                state_dict_full, _ = _filter_state_dict_keys(
                    state_dict_full, skip_substrings=filter_list
                )

                # in_dim=16 surgery: the checkpoint's patch_embedding.weight is
                # shaped [dim, 36, 1, 2, 2] (16 video + 4 mask + 16 render),
                # but when we instantiate the model with in_dim=16 the conv
                # expects [dim, 16, 1, 2, 2].  Slice the checkpoint tensor to
                # the first 16 input channels (the "video" branch) — those
                # weights are the ones we want to keep for pure-latent I2V.
                # The dropped 20 channels were already getting zeros fed into
                # them at runtime (render_latent_input=None → zero-pad), so
                # slicing is bit-exact equivalent to the zero-pad regime at
                # init, with the added benefit that gradients no longer drift
                # those 20 channels away from zero over training.
                # Both causal students and full-attention teachers may use a
                # pure 16-channel latent interface with an I2V checkpoint whose
                # patch embedding has extra mask/render channels. Keep the
                # pretrained video-channel prefix in either case.
                pe_key = "patch_embedding.weight"
                _slice_prefix_tensor_for_live_shape(
                    state_dict_full,
                    pe_key,
                    self.model.patch_embedding.weight,
                    "in_dim",
                )
                if is_causal:
                    _slice_prefix_tensor_for_live_shape(
                        state_dict_full,
                        "head.head.weight",
                        self.model.head.head.weight,
                        "out_dim",
                    )
                    _slice_prefix_tensor_for_live_shape(
                        state_dict_full,
                        "head.head.bias",
                        self.model.head.head.bias,
                        "out_dim",
                    )

                missing_keys, unexpected_keys = self.model.load_state_dict(
                    state_dict_full, strict=False
                )
                primary_missing_keys = set(missing_keys)
                print(
                    f"load_model {model_path} (primary) missing_keys: {len(missing_keys)} unexpected_keys: {len(unexpected_keys)}"
                )
                if require_full_weight_coverage and (missing_keys or unexpected_keys):
                    raise RuntimeError(
                        f"incomplete pretrained weight coverage for {model_path}: "
                        f"missing={len(missing_keys)} {missing_keys[:20]} "
                        f"unexpected={len(unexpected_keys)} "
                        f"{unexpected_keys[:20]}"
                    )
            elif require_full_weight_coverage:
                raise RuntimeError(f"no pretrained weights loaded for {model_path}")

            # Causal models are constructed on `meta` then materialized with
            # to_empty(), so "zero-init" modules whose keys are absent from the
            # source checkpoint must be explicitly reset after materialization.
            # Otherwise the camera phase MLP reads uninitialized memory at step
            # 0 and breaks the pretrained-identity invariant for cc_* modes.
            if is_causal and cc_rope_mode in (
                "cc_basic",
                "cc_output",
                "cc_value",
                "cc_full",
                "cc_dual_channel",
                "cc_dual_output",
            ):
                n_zeroed = 0
                n_present = 0
                missing = primary_missing_keys or set()
                for i, blk in enumerate(self.model.blocks):
                    mlp = getattr(blk.self_attn, "camera_phase_mlp", None)
                    if mlp is None:
                        continue
                    n_present += 1
                    key = f"blocks.{i}.self_attn.camera_phase_mlp.proj.weight"
                    if primary_missing_keys is not None and key not in missing:
                        continue
                    with torch.no_grad():
                        mlp.proj.weight.zero_()
                    n_zeroed += 1
                print(
                    f"[CC-RoPE {cc_rope_mode}] zeroed camera_phase_mlp on "
                    f"{n_zeroed}/{n_present} blocks with missing checkpoint keys"
                )

            control_embedding = getattr(
                self.model, "degradation_control_embedding", None
            )
            if is_causal and control_embedding is not None:
                control_key = "degradation_control_embedding.0.weight"
                if primary_missing_keys is None or control_key in (
                    primary_missing_keys or set()
                ):
                    self.model.reset_degradation_control_parameters()
                    print(
                        "[DegradationControl] initialized missing adapter with "
                        "a zero output projection"
                    )

            # CC-RoPE modes: pretrained checkpoints (e.g. HY-WorldPlay /
            # some adapted checkpoints) may ship non-zero prope_proj
            # weights learned for the dual-attention path. For cc_output,
            # cc_dual_output, prope_residual, and cc_full they'd be fed a
            # differently-distributed input (P·x_std vs x_p_from_2nd_attn),
            # so we zero them post-load to guarantee the bit-exact-identity
            # invariant at step 0.
            # `cc_basic` / `cc_dual_channel` / `cc_value` don't instantiate
            # prope_proj at all (set to None in __init__) — stale checkpoint
            # keys simply land in `unexpected_keys`, no action needed here.
            if is_causal and cc_rope_mode in (
                "cc_output",
                "cc_dual_output",
                "prope_residual",
                "cc_full",
            ):
                n_zeroed = 0
                for blk in self.model.blocks:
                    pp = blk.self_attn.prope_proj
                    if pp is None:
                        continue
                    if pp.weight.abs().sum().item() > 0.0:
                        n_zeroed += 1
                    with torch.no_grad():
                        pp.weight.zero_()
                        if pp.bias is not None:
                            pp.bias.zero_()
                print(
                    f"[CC-RoPE {cc_rope_mode}] re-zeroed prope_proj on {n_zeroed}/{len(self.model.blocks)} blocks (was non-zero from pretrained ckpt)"
                )

            # cc_value / cc_full: re-zero `value_proj` for the same
            # step-0-bit-exact invariant. Pretrained ckpts won't have this
            # key, but if a future ckpt ships value_proj weights, they must
            # not contaminate step 0. `cc_basic`/`cc_dual_channel`/
            # `dual_prope`/`cc_output` don't instantiate value_proj at all
            # (set to None in __init__).
            if is_causal and cc_rope_mode in ("cc_value", "cc_full"):
                n_zeroed = 0
                for blk in self.model.blocks:
                    vp = blk.self_attn.value_proj
                    if vp is None:
                        continue
                    if vp.weight.abs().sum().item() > 0.0:
                        n_zeroed += 1
                    with torch.no_grad():
                        vp.weight.zero_()
                        if vp.bias is not None:
                            vp.bias.zero_()
                print(
                    f"[CC-RoPE {cc_rope_mode}] re-zeroed value_proj on {n_zeroed}/{len(self.model.blocks)} blocks (was non-zero from pretrained ckpt)"
                )

            # Load secondary model (only for dual_model and non-causal mode)
            if dual_model and not is_causal and state_dict_full_2 is not None:
                state_dict_full_2, _ = _filter_state_dict_keys(
                    state_dict_full_2, skip_substrings=filter_list
                )
                missing_keys_2, unexpected_keys_2 = self.model_2.load_state_dict(
                    state_dict_full_2, strict=False
                )
                print(
                    f"load_model_2 {model_path} (low noise model for Wan2.2) missing_keys: {len(missing_keys_2)} unexpected_keys: {len(unexpected_keys_2)}"
                )

        if dist.is_initialized():
            dist.barrier()

        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 1560 * 24  # [1, 12 * 2, 16, 60, 104]
        self.post_init()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        kv_size: Optional[Tuple[int, int]] = (0, 0),
        image_latent_input: Optional[torch.Tensor] = None,
        render_latent_input: Optional[torch.Tensor] = None,
        freqs_offset: int = 0,
        freqs_positions: Optional[torch.Tensor] = None,
        viewmats: Optional[torch.Tensor] = None,  # [B, F, 4, 4] c2w
        Ks: Optional[torch.Tensor] = None,  # [B, F, 3, 3] intrinsics
        degradation_control: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        prompt_embeds = _select_chunk_prompt_embeds(
            conditional_dict["prompt_embeds"],
            conditional_dict,
            current_frames=noisy_image_or_video.shape[1],
            freqs_offset=freqs_offset,
        )
        if degradation_control is None:
            degradation_control = conditional_dict.get("degradation_control")
        if degradation_control is not None:
            current_frames = noisy_image_or_video.shape[1]
            if degradation_control.shape[1] != current_frames:
                start = int(freqs_offset)
                stop = start + current_frames
                if degradation_control.shape[1] < stop:
                    raise ValueError(
                        "degradation_control does not cover the requested "
                        f"frame range [{start}, {stop}); shape is "
                        f"{tuple(degradation_control.shape)}"
                    )
                degradation_control = degradation_control[:, start:stop]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # X0 prediction
        # Handle None inputs for T2V mode
        image_latent_permuted = (
            image_latent_input.permute(0, 2, 1, 3, 4).contiguous()
            if image_latent_input is not None
            else None
        )
        render_latent_permuted = (
            render_latent_input.permute(0, 2, 1, 3, 4).contiguous()
            if render_latent_input is not None
            else None
        )
        if kv_cache is None:
            raise ValueError("ReMind inference requires an initialized KV cache")
        if self.dual_model:
            raise ValueError("KV-cache inference does not support dual-model mode")
        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous(),
            t=input_timestep,
            context=prompt_embeds,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            kv_size=kv_size,
            image_latent_input=image_latent_permuted,
            render_latent_input=render_latent_permuted,
            freqs_offset=freqs_offset,
            freqs_positions=freqs_positions,
            viewmats=viewmats,
            Ks=Ks,
            degradation_control=degradation_control,
        ).permute(0, 2, 1, 3, 4)
        if kv_size[1] < 0:
            return flow_pred

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def forward_wan22(
        self,
        latent_list: List[torch.Tensor],
        t: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        **kwargs,
    ) -> List[torch.Tensor]:
        """
        Forward method specifically for Wan2.2 dual-model inference.
        Compatible with T2VAlignedInferencePipeline's direct model call signature.

        Args:
            latent_list: List of latent tensors [B, C, F, H, W]
            t: Timestep tensor [B]
            context: Text embeddings
            seq_len: Sequence length
            **kwargs: Additional arguments

        Returns:
            List of flow predictions
        """
        if not self.dual_model:
            raise ValueError("forward_wan22 is only available for dual-model mode")

        # Select model based on timestep
        normalized_timestep = t.float() / 1000.0
        use_high_noise = (normalized_timestep >= self.high_noise_threshold).all().item()
        selected_model = self.model if use_high_noise else self.model_2

        # Process each latent in the list
        output_list = []
        for latent in latent_list:
            flow_pred = selected_model(
                latent, t=t, context=context, seq_len=seq_len, **kwargs
            )
            output_list.append(flow_pred)

        return output_list

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
