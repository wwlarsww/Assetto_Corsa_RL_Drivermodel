"""Stable names and scales for the trajectory predictor contract."""

from __future__ import annotations

from assetto_corsa_rl.types import PREDICTOR_TIME_OFFSETS_S

BEZIER_TRAJECTORY_CODEC = "bezier-origin-5x2-10x0p1s-normalized-v1"
PREDICTOR_COORDINATE_SCALE_M = 100.0
PREDICTOR_CONTROL_POINT_COUNT = 5
PREDICTOR_COORDINATE_DIMENSION = 2
PREDICTOR_WAYPOINT_COUNT = int(PREDICTOR_TIME_OFFSETS_S.size)
PREDICTOR_HORIZON_S = float(PREDICTOR_TIME_OFFSETS_S[-1])
