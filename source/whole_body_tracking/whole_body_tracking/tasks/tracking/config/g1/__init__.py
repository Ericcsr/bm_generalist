import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Grasp-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatEnvGraspCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-3pt-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-3pt-Binded-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-3pt-Binded-Compliance-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedComplianceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-chip-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptComplianceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Rough-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Terrain-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Multi-Terrain-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptMultiTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-By-Exp-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainByExpEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Contact-Collision-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainContactCollisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Object-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiObjectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Compliance-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectComplianceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-NoContact-Baseline-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectNoContactBaselineEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectUnilateralContactEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-NoFootContact-Baseline-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectUnilateralContactNoFootContactBaselineEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectUnilateralContact5SeatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-2Bit-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectUnilateralContact2BitEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-NoContactObs-Baseline-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitNoContactObsBaselineEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Unilateral-Contact-5Seat-2Bit-Compliance-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectUnilateralContact5Seat2BitComplianceEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-2BitContact-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObject2BitContactEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Merged-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectMergedEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-HandMid-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectHandMidEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-CriticCat-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectCriticCategoryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-Merged-CriticCat-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectMergedCriticCategoryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)

gym.register(
    id="Tracking-3pt-Binded-Multi-Terrain-Object-HandMid-CriticCat-G1-v0",
    entry_point="whole_body_tracking.envs:ManagerBasedRLEnvWithKinematicObjects",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.G1Flat3ptBindedMultiTerrainObjectHandMidCriticCategoryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOBindedMultiTerrainObjectRunnerCfg",
    },
)
