"""Compute the Trade-off F1 (TF1) score for AREA evaluation metrics."""

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class TF1Scores:
    """Container for normalized sub-scores and the final TF1 score."""

    esr: float
    usr: float
    tf1: float


def _validate_range(name: str, value: float, lower: float, upper: float) -> None:
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {value}.")


def normalize_pls(pls: float) -> float:
    """Normalize PLS from [1, 10] to [0, 1], where higher is better."""
    _validate_range("PLS", pls, 1.0, 10.0)
    return 1.0 - (pls - 1.0) / 9.0


def normalize_ss(ss: float) -> float:
    """Normalize SS from [0, 1] to [0, 1], where higher is better."""
    _validate_range("SS", ss, 0.0, 1.0)
    return 1.0 - ss


def normalize_rus(rus: float) -> float:
    """Normalize RUS from [1, 10] to [0, 1], where higher is better."""
    _validate_range("RUS", rus, 1.0, 10.0)
    return (rus - 1.0) / 9.0


def normalize_fc(fc: float) -> float:
    """Normalize FC from [1, 10] to [0, 1], where higher is better."""
    _validate_range("FC", fc, 1.0, 10.0)
    return (fc - 1.0) / 9.0


def compute_esr(pls: float, ss: float) -> float:
    """Compute effectiveness success rate as mean normalized PLS and SS."""
    return 0.5 * (normalize_pls(pls) + normalize_ss(ss))


def compute_usr(rus: float, fc: float) -> float:
    """Compute usability success rate as mean normalized RUS and FC."""
    return 0.5 * (normalize_rus(rus) + normalize_fc(fc))


def compute_tf1(pls: float, ss: float, rus: float, fc: float, eps: float = 1e-8) -> float:
    """Compute TF1 as the harmonic mean of ESR and USR."""
    esr = compute_esr(pls, ss)
    usr = compute_usr(rus, fc)
    return 2.0 * esr * usr / (esr + usr + eps)


def compute_scores(pls: float, ss: float, rus: float, fc: float) -> TF1Scores:
    """Compute ESR, USR, and TF1 together."""
    esr = compute_esr(pls, ss)
    usr = compute_usr(rus, fc)
    tf1 = 2.0 * esr * usr / (esr + usr + 1e-8)
    return TF1Scores(esr=esr, usr=usr, tf1=tf1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute AREA Trade-off F1 (TF1).")
    parser.add_argument("--pls", type=float, required=True, help="PLS score in [1, 10]. Lower is better.")
    parser.add_argument("--ss", type=float, required=True, help="SS score in [0, 1]. Lower is better.")
    parser.add_argument("--rus", type=float, required=True, help="RUS score in [1, 10]. Higher is better.")
    parser.add_argument("--fc", type=float, required=True, help="FC score in [1, 10]. Higher is better.")
    parser.add_argument("--precision", type=int, default=4, help="Decimal places to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = compute_scores(pls=args.pls, ss=args.ss, rus=args.rus, fc=args.fc)
    digits = max(0, args.precision)

    print(f"ESR: {scores.esr:.{digits}f}")
    print(f"USR: {scores.usr:.{digits}f}")
    print(f"TF1: {scores.tf1:.{digits}f}")


if __name__ == "__main__":
    main()