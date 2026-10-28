import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

# Checkpoints (model_*.pt, exported .onnx) stay on local disk only; uploading them to wandb exceeded the
# storage quota. Metrics are still logged. Set WANDB_UPLOAD_CKPT=1 to upload again.
UPLOAD_CKPT = os.environ.get("WANDB_UPLOAD_CKPT", "0") == "1"


def _save_checkpoint(runner, path, infos):
    """OnPolicyRunner.save, without its writer.save_model() upload unless UPLOAD_CKPT."""
    if UPLOAD_CKPT:
        OnPolicyRunner.save(runner, path, infos)
        return
    disable = runner.disable_logs
    runner.disable_logs = True          # only gates the upload in OnPolicyRunner.save
    try:
        OnPolicyRunner.save(runner, path, infos)
    finally:
        runner.disable_logs = disable


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        _save_checkpoint(self, path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            if UPLOAD_CKPT:
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        _save_checkpoint(self, path, infos)
        if self.logger_type in ["wandb"] and not self.disable_logs:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            if UPLOAD_CKPT:
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
