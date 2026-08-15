#!/usr/bin/env python3
"""Export all RSL-RL PyTorch checkpoints in a run directory to ONNX.

This script intentionally avoids importing Isaac Lab, Isaac Sim, or the play script.
It reconstructs the inference actor directly from checkpoint tensors.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn


ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}

METADATA_KEYS = (
    "run_path",
    "joint_names",
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "command_names",
    "observation_names",
    "action_scale",
    "anchor_body_name",
    "body_names",
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}

G1_ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

G1_OBSERVATION_NAMES = [
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]

G1_ACTUATOR_GROUPS = (
    {
        "names": [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
        "effort": {
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 88.0,
            ".*_knee_joint": 139.0,
        },
        "stiffness": {
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
        },
        "damping": {
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
        },
    },
    {
        "names": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        "effort": 50.0,
        "stiffness": 2.0 * STIFFNESS_5020,
        "damping": 2.0 * DAMPING_5020,
    },
    {
        "names": ["waist_roll_joint", "waist_pitch_joint"],
        "effort": 50.0,
        "stiffness": 2.0 * STIFFNESS_5020,
        "damping": 2.0 * DAMPING_5020,
    },
    {
        "names": ["waist_yaw_joint"],
        "effort": 88.0,
        "stiffness": STIFFNESS_7520_14,
        "damping": DAMPING_7520_14,
    },
    {
        "names": [
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        "effort": {
            ".*_shoulder_pitch_joint": 25.0,
            ".*_shoulder_roll_joint": 25.0,
            ".*_shoulder_yaw_joint": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        "stiffness": {
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        "damping": {
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
    },
)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else -1, path.name)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} is not a dict checkpoint.")
    return checkpoint


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(item) for item in _parse_csv(value)]


def _matches(pattern: str, name: str) -> bool:
    return re.fullmatch(pattern, name) is not None


def _expand_pattern_values(joint_names: list[str], values_by_pattern: dict[str, float], default: float = 0.0) -> list[float]:
    expanded = []
    for joint_name in joint_names:
        value = default
        for pattern, pattern_value in values_by_pattern.items():
            if _matches(pattern, joint_name):
                value = pattern_value
        expanded.append(value)
    return expanded


def _expand_actuator_values(joint_names: list[str], value_name: str) -> dict[str, float]:
    expanded: dict[str, float] = {}
    for group in G1_ACTUATOR_GROUPS:
        names = group["names"]
        values = group[value_name]
        if not isinstance(values, dict):
            values = {pattern: values for pattern in names}
        for pattern, value in values.items():
            for joint_name in joint_names:
                if _matches(pattern, joint_name):
                    expanded[joint_name] = float(value)
    return expanded


def _g1_metadata() -> dict[str, Any]:
    joint_names = G1_ISAAC_JOINT_NAMES
    effort = _expand_actuator_values(joint_names, "effort")
    stiffness = _expand_actuator_values(joint_names, "stiffness")
    damping = _expand_actuator_values(joint_names, "damping")
    action_scale = [
        0.25 * effort[joint_name] / stiffness[joint_name] if joint_name in effort and joint_name in stiffness else 1.0
        for joint_name in joint_names
    ]
    return {
        "joint_names": joint_names,
        "joint_stiffness": [stiffness.get(joint_name, 0.0) for joint_name in joint_names],
        "joint_damping": [damping.get(joint_name, 0.0) for joint_name in joint_names],
        "default_joint_pos": _expand_pattern_values(joint_names, G1_DEFAULT_JOINT_POS),
        "action_scale": action_scale,
        "command_names": ["motion"],
        "observation_names": G1_OBSERVATION_NAMES,
        "anchor_body_name": "pelvis",
        "body_names": G1_BODY_NAMES,
    }


def _list_to_csv_str(values: list[Any], decimals: int = 3) -> str:
    fmt = f"{{:.{decimals}f}}"
    return ",".join(fmt.format(value) if isinstance(value, (int, float)) else str(value) for value in values)


def _load_metadata_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.expanduser().open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return {key: loaded[key] for key in METADATA_KEYS if key in loaded}


def _metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    metadata = _g1_metadata() if args.robot == "g1" else {}
    metadata.update(_load_metadata_json(args.metadata_json))
    if args.run_path is not None:
        metadata["run_path"] = args.run_path
    if args.joint_names is not None:
        metadata["joint_names"] = _parse_csv(args.joint_names)
    if args.joint_stiffness is not None:
        metadata["joint_stiffness"] = _parse_float_csv(args.joint_stiffness)
    if args.joint_damping is not None:
        metadata["joint_damping"] = _parse_float_csv(args.joint_damping)
    if args.default_joint_pos is not None:
        metadata["default_joint_pos"] = _parse_float_csv(args.default_joint_pos)
    if args.command_names is not None:
        metadata["command_names"] = _parse_csv(args.command_names)
    if args.observation_names is not None:
        metadata["observation_names"] = _parse_csv(args.observation_names)
    if args.action_scale is not None:
        metadata["action_scale"] = _parse_float_csv(args.action_scale)
    if args.anchor_body_name is not None:
        metadata["anchor_body_name"] = args.anchor_body_name
    if args.body_names is not None:
        metadata["body_names"] = _parse_csv(args.body_names)
    return metadata


def _attach_onnx_metadata(onnx_path: Path, metadata: dict[str, Any]) -> None:
    if not metadata:
        return

    import onnx

    model = onnx.load(onnx_path)
    existing = {entry.key: entry for entry in model.metadata_props}
    for key in METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        value_str = _list_to_csv_str(value) if isinstance(value, list) else str(value)
        if key in existing:
            existing[key].value = value_str
        else:
            entry = onnx.StringStringEntryProto()
            entry.key = key
            entry.value = value_str
            model.metadata_props.append(entry)
    onnx.save(model, onnx_path)


def _model_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid model_state_dict.")
    return state_dict


def _actor_prefix(state_dict: dict[str, torch.Tensor]) -> str:
    prefixes = ("actor.", "module.actor.", "policy.actor.", "actor_critic.actor.")
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            return prefix
    raise KeyError("Could not find actor weights in checkpoint. Expected keys like 'actor.0.weight'.")


def _linear_layer_ids(state_dict: dict[str, torch.Tensor], prefix: str) -> list[int]:
    layer_ids: list[int] = []
    for key, value in state_dict.items():
        if not key.startswith(prefix) or not key.endswith(".weight") or value.ndim != 2:
            continue
        layer_name = key[len(prefix) : -len(".weight")]
        if layer_name.isdigit():
            layer_ids.append(int(layer_name))
    if not layer_ids:
        raise KeyError("Could not find linear actor layers in checkpoint.")
    return sorted(layer_ids)


def _build_actor(state_dict: dict[str, torch.Tensor], activation_name: str) -> nn.Sequential:
    prefix = _actor_prefix(state_dict)
    layer_ids = _linear_layer_ids(state_dict, prefix)
    activation_cls = ACTIVATIONS[activation_name]

    modules: list[nn.Module] = []
    for index, layer_id in enumerate(layer_ids):
        weight = state_dict[f"{prefix}{layer_id}.weight"].detach().cpu()
        bias = state_dict.get(f"{prefix}{layer_id}.bias")
        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        layer.weight.data.copy_(weight)
        if bias is not None:
            layer.bias.data.copy_(bias.detach().cpu())
        modules.append(layer)
        if index != len(layer_ids) - 1:
            modules.append(activation_cls())

    return nn.Sequential(*modules).eval()


def _normalizer_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    for key in ("obs_norm_state_dict", "obs_normalizer_state_dict", "normalizer_state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    return None


def _get_buffer(state: dict[str, torch.Tensor], *names: str) -> torch.Tensor | None:
    for name in names:
        value = state.get(name)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
    return None


class ObservationNormalizer(nn.Module):
    """Small compatibility wrapper for RSL-RL empirical normalization checkpoints."""

    def __init__(self, state: dict[str, torch.Tensor], eps: float):
        super().__init__()
        mean = _get_buffer(state, "mean", "_mean", "running_mean")
        std = _get_buffer(state, "std", "_std")
        var = _get_buffer(state, "var", "_var", "running_var")
        if mean is None:
            raise KeyError("Observation normalizer is missing a mean buffer.")
        if std is None:
            if var is None:
                raise KeyError("Observation normalizer is missing a std or variance buffer.")
            std = torch.sqrt(var)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = eps

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.mean) / (self.std + self.eps)


class IdentityNormalizer(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs


class OnnxPolicy(nn.Module):
    def __init__(self, actor: nn.Module, normalizer: nn.Module):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(obs))


def _inline_external_data(onnx_path: Path) -> None:
    """Rewrite the ONNX as a single self-contained file (weights inline, no .onnx.data sidecar).

    The dynamo-based ``torch.onnx.export`` in this torch version writes weights as external data by
    default, producing a ``<name>.onnx.data`` sidecar. Some deploy runtimes expect a single file, so
    re-load with the external data pulled into memory and re-save without external data.
    """
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save(model, onnx_path, save_as_external_data=False)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()


def export_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    activation: str,
    opset: int,
    normalizer_eps: float,
    metadata: dict[str, Any],
    verbose: bool,
    inline_weights: bool = True,
) -> None:
    checkpoint = _load_checkpoint(checkpoint_path)
    actor = _build_actor(_model_state_dict(checkpoint), activation)
    normalizer_state = _normalizer_state(checkpoint)
    normalizer: nn.Module = (
        ObservationNormalizer(normalizer_state, normalizer_eps) if normalizer_state is not None else IdentityNormalizer()
    )
    policy = OnnxPolicy(actor, normalizer.eval()).eval()

    obs_dim = actor[0].in_features
    dummy_obs = torch.zeros(1, obs_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        policy,
        dummy_obs,
        output_path,
        export_params=True,
        opset_version=opset,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={},
    )

    # Attach metadata first (operates on the external-data model), then optionally collapse to one file.
    _attach_onnx_metadata(output_path, metadata)
    if inline_weights:
        _inline_external_data(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Run directory containing model_*.pt checkpoints. Omit when using --wandb-path.",
    )
    parser.add_argument(
        "--wandb-path",
        default=None,
        help=(
            "W&B run path to download from instead of a local run_dir: 'org/project/run_id' (downloads the "
            "latest model_*.pt) or 'org/project/run_id/model_N.pt' (that exact checkpoint). Downloads to "
            "./logs/rsl_rl/temp. Also fills the metadata run_path if --run-path is not given."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for ONNX files. Defaults to <run_dir>/exported_onnx (or ./logs/rsl_rl/temp/exported_onnx with --wandb-path).",
    )
    parser.add_argument("--pattern", default="model_*.pt", help="Checkpoint glob pattern inside run_dir.")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="elu", help="Actor hidden activation.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument(
        "--robot",
        choices=("g1", "none"),
        default="g1",
        help="Built-in robot metadata preset. Use 'none' when providing all metadata yourself.",
    )
    parser.add_argument(
        "--normalizer-eps",
        type=float,
        default=1.0e-2,
        help="Epsilon added to observation std. Matches RSL-RL EmpiricalNormalization default.",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help=(
            "JSON object with play.py metadata keys: run_path, joint_names, joint_stiffness, joint_damping, "
            "default_joint_pos, command_names, observation_names, action_scale, anchor_body_name, body_names."
        ),
    )
    parser.add_argument("--run-path", default=None, help="Metadata run_path value.")
    parser.add_argument("--joint-names", default=None, help="Comma-separated metadata joint_names.")
    parser.add_argument("--joint-stiffness", default=None, help="Comma-separated metadata joint_stiffness.")
    parser.add_argument("--joint-damping", default=None, help="Comma-separated metadata joint_damping.")
    parser.add_argument("--default-joint-pos", default=None, help="Comma-separated metadata default_joint_pos.")
    parser.add_argument("--command-names", default=None, help="Comma-separated metadata command_names.")
    parser.add_argument("--observation-names", default=None, help="Comma-separated metadata observation_names.")
    parser.add_argument("--action-scale", default=None, help="Comma-separated metadata action_scale.")
    parser.add_argument("--anchor-body-name", default=None, help="Metadata anchor_body_name.")
    parser.add_argument("--body-names", default=None, help="Comma-separated metadata body_names.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose ONNX exporter output.")
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Keep weights in a separate <name>.onnx.data sidecar. Default: inline into a single .onnx file.",
    )
    return parser.parse_args()


def _download_from_wandb(wandb_path: str) -> Path:
    """Download a checkpoint from a W&B run path (same semantics as play.py) and return its local path."""
    import wandb

    run_path = wandb_path
    api = wandb.Api()
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)
    files = [f.name for f in wandb_run.files() if "model" in f.name]
    if "model" in wandb_path:
        file = wandb_path.split("/")[-1]
    else:
        if not files:
            raise FileNotFoundError(f"No model_*.pt files found in W&B run {run_path}")
        file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    dest_dir = Path("./logs/rsl_rl/temp")
    dest_dir.mkdir(parents=True, exist_ok=True)
    wandb_run.file(str(file)).download(str(dest_dir), replace=True)
    local = dest_dir / str(file)
    print(f"[INFO] Downloaded checkpoint from W&B: {run_path}/{file} -> {local}")
    return local


def main() -> None:
    args = parse_args()

    if args.wandb_path:
        checkpoint_path = _download_from_wandb(args.wandb_path).expanduser().resolve()
        if args.run_path is None:
            args.run_path = args.wandb_path
        output_dir = (args.output_dir or checkpoint_path.parent / "exported_onnx").expanduser().resolve()
        metadata = _metadata_from_args(args)
        checkpoints = [checkpoint_path]
    else:
        if args.run_dir is None:
            raise SystemExit("Provide a run_dir or --wandb-path <org/project/run_id[/model_N.pt]>.")
        run_dir = args.run_dir.expanduser().resolve()
        output_dir = (args.output_dir or run_dir / "exported_onnx").expanduser().resolve()
        metadata = _metadata_from_args(args)
        checkpoints = sorted(run_dir.glob(args.pattern), key=_checkpoint_sort_key)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints matching {args.pattern!r} found in {run_dir}.")

    for checkpoint_path in checkpoints:
        output_path = output_dir / f"{checkpoint_path.stem}.onnx"
        export_checkpoint(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            activation=args.activation,
            opset=args.opset,
            normalizer_eps=args.normalizer_eps,
            metadata=metadata,
            verbose=args.verbose,
            inline_weights=not args.external_data,
        )
        print(f"[OK] {checkpoint_path.name} -> {output_path}")


if __name__ == "__main__":
    main()
