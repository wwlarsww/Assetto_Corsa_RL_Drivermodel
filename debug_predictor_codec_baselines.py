from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


DEFAULT_DATASET = Path(
    r"data\processed\bmw_z4_gt3_monza\bc_dataset.npz"
)
OFFSETS_S = np.arange(1, 21, dtype=np.float64) * 0.25
EPSILON = 1e-6


def bezier_basis(
    parameter: np.ndarray,
    control_point_count: int,
) -> np.ndarray:
    degree = control_point_count - 1
    return np.column_stack(
        [
            math.comb(degree, index)
            * (1.0 - parameter) ** (degree - index)
            * parameter**index
            for index in range(control_point_count)
        ]
    )


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float, float]:
    distance = np.linalg.norm(prediction - target, axis=-1)
    ade_per_sample = np.mean(distance, axis=1)
    fde_per_sample = distance[:, -1]
    return (
        float(np.mean(ade_per_sample)),
        float(np.median(ade_per_sample)),
        float(np.mean(fde_per_sample)),
        float(np.median(fde_per_sample)),
    )


def print_metrics(
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    ade_mean, ade_median, fde_mean, fde_median = metrics(
        prediction,
        target,
    )
    print(
        f"{name:34s} "
        f"ADE mean/median={ade_mean:8.3f}/{ade_median:8.3f} m  "
        f"FDE mean/median={fde_mean:8.3f}/{fde_median:8.3f} m"
    )


dataset_path = Path(
    os.environ.get("BC_DATASET_PATH", str(DEFAULT_DATASET))
)
if not dataset_path.is_file():
    raise FileNotFoundError(dataset_path.resolve())

with np.load(dataset_path, allow_pickle=False) as data:
    mean = np.asarray(data["normalizer_mean"], dtype=np.float64)
    std = np.maximum(
        np.asarray(data["normalizer_std"], dtype=np.float64),
        EPSILON,
    )

    splits: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("train", "validation", "test"):
        observations = np.asarray(
            data[f"{split_name}_observations"],
            dtype=np.float64,
        )
        targets = np.asarray(
            data[f"{split_name}_predictor_target_xz_m"],
            dtype=np.float64,
        )
        valid = np.asarray(
            data[f"{split_name}_predictor_valid_mask"],
            dtype=bool,
        )
        raw = observations * std[None, :] + mean[None, :]
        splits[split_name] = {
            "speed": raw[:, 0],
            "target": targets,
            "valid": valid,
        }


train_valid_target = splits["train"]["target"][
    splits["train"]["valid"].astype(bool)
]
mean_train_target = np.mean(train_valid_target, axis=0)

# Aktueller Codec: fünf freie Kontrollpunkte. Die 20 Zielpunkte werden
# gleichmäßig auf den Bézier Parameterbereich 0 bis 1 abgebildet.
current_parameter = np.linspace(0.0, 1.0, 20, dtype=np.float64)
current_basis = bezier_basis(current_parameter, 5)
current_pseudoinverse = np.linalg.pinv(current_basis)

# Alternativer Codec: Fahrzeugursprung als fester P0, anschließend fünf
# vorhergesagte Kontrollpunkte P1 bis P5. Die Zielzeiten 0,25 bis 5,0 s
# werden auf t/5,0 abgebildet. Die Netzausgabe bleibt damit zehn Werte groß.
anchored_parameter = OFFSETS_S / OFFSETS_S[-1]
anchored_basis_full = bezier_basis(anchored_parameter, 6)
anchored_free_basis = anchored_basis_full[:, 1:]
anchored_pseudoinverse = np.linalg.pinv(anchored_free_basis)

print("=" * 100)
print("Prädiktor Zielstatistik und erreichbare Bézier Fehler")
print("=" * 100)
print("Datensatz:", dataset_path.resolve())

for split_name, split in splits.items():
    valid = split["valid"].astype(bool)
    target = split["target"][valid]
    speed = split["speed"][valid]

    first_distance = np.linalg.norm(target[:, 0], axis=1)
    last_distance = np.linalg.norm(target[:, -1], axis=1)

    zero_prediction = np.zeros_like(target)

    straight_prediction = np.zeros_like(target)
    straight_prediction[:, :, 1] = speed[:, None] * OFFSETS_S[None, :]

    mean_prediction = np.broadcast_to(
        mean_train_target[None, :, :],
        target.shape,
    )

    current_control = np.einsum(
        "ps,nsc->npc",
        current_pseudoinverse,
        target,
    )
    current_oracle = np.einsum(
        "sp,npc->nsc",
        current_basis,
        current_control,
    )

    anchored_control = np.einsum(
        "ps,nsc->npc",
        anchored_pseudoinverse,
        target,
    )
    anchored_oracle = np.einsum(
        "sp,npc->nsc",
        anchored_free_basis,
        anchored_control,
    )

    print()
    print("-" * 100)
    print(
        f"{split_name}: Samples={split['target'].shape[0]}, "
        f"gültige Ziele={target.shape[0]}"
    )
    print(
        "Zielabstand erster Punkt Median/p95:",
        f"{np.median(first_distance):.3f}/"
        f"{np.quantile(first_distance, 0.95):.3f} m",
    )
    print(
        "Zielabstand letzter Punkt Median/p95:",
        f"{np.median(last_distance):.3f}/"
        f"{np.quantile(last_distance, 0.95):.3f} m",
    )
    print(
        "Anteil letzter Zielpunkt z > 0:",
        f"{100.0 * np.mean(target[:, -1, 1] > 0.0):.2f} Prozent",
    )
    print()
    print_metrics("Nulltrajektorie", zero_prediction, target)
    print_metrics(
        "Gerade mit aktueller Geschwindigkeit",
        straight_prediction,
        target,
    )
    print_metrics(
        "Mittlere Trainingstrajektorie",
        mean_prediction,
        target,
    )
    print_metrics(
        "Oracle aktueller 5x2 Codec",
        current_oracle,
        target,
    )
    print_metrics(
        "Oracle Ursprung plus 5x2 Codec",
        anchored_oracle,
        target,
    )

print()
print("=" * 100)
print("Interpretation")
print("=" * 100)
print(
    "Ein Oracle Fehler nahe null bedeutet: Der Codec kann die Zielkurven "
    "darstellen und das Problem liegt im Training beziehungsweise in der "
    "Skalierung."
)
print(
    "Ein hoher Oracle Fehler bedeutet: Die Bézier Repräsentation selbst "
    "passt nicht ausreichend zu den Zieltrajektorien."
)
print(
    "Ist 'Ursprung plus 5x2' ähnlich gut wie der aktuelle Codec, kann die "
    "Kurve sauber bei [0, 0] verankert werden, ohne die zehn Netzausgaben "
    "zu vergrößern."
)
