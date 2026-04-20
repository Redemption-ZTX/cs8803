"""Imitation / inverse-RL utilities for snapshot-036+ learned reward shaping.

This package provides:
- TrajectoryRecorder: per-step trajectory dumper with failure-bucket label.
- save_trajectory: persist trajectory as compressed npz + json metadata.

Future modules (snapshot-036 path C):
- learned_reward_trainer.py: train multi-head reward model from W/L data.
"""
