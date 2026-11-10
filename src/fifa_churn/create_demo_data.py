import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def create_demo_data(output: Path, rows: int = 400) -> None:
    if rows < 20:
        raise ValueError("Demo data needs at least 20 rows.")
    rng = np.random.default_rng(42)
    ages = rng.integers(18, 41, size=rows)
    overall = rng.integers(48, 90, size=rows)
    data = pd.DataFrame(
        {
            "age": ages,
            "overall": overall,
            "potential": np.maximum(
                overall, rng.integers(55, 96, size=rows)
            ),
            "height_cm": rng.integers(160, 201, size=rows),
            "weight_kg": rng.integers(55, 96, size=rows),
            "weak_foot": rng.integers(1, 6, size=rows),
            "skill_moves": rng.integers(1, 6, size=rows),
            "international_reputation": rng.integers(1, 5, size=rows),
            "player_positions": rng.choice(
                ["CB", "CM", "ST", "GK", "LW"], size=rows
            ),
            "preferred_foot": rng.choice(["Left", "Right"], size=rows),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f"[SUCCESS] Generated {rows} fictional demo rows at {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fictional, non-FIFA rows for CI/demo use."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=400)
    args = parser.parse_args()
    create_demo_data(args.output, args.rows)


if __name__ == "__main__":
    main()
