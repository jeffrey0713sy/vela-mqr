"""
Create reproducible train/validation/external_test splits for Vela MQR.

Default behavior:
- source: data/reference_population_master.json
- include only markets with mqr_rating
- split ratio: 70/15/15
- seed: 20260324
- output:
    data/splits/train.json
    data/splits/validation.json
    data/splits/external_test.json
"""

import argparse
import json
import random
from pathlib import Path


def stable_market_key(market: dict) -> str:
    mid = str(market.get("id", "")).strip()
    if mid:
        return mid
    name = str(market.get("market_name", "")).strip()
    year = str(market.get("ref_year", "")).strip()
    return f"{name}::{year}"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible dataset splits.")
    parser.add_argument("--source", type=str, default="data/reference_population_master.json")
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    parser.add_argument("--external-test", type=float, default=0.15)
    parser.add_argument("--include-unrated", action="store_true")
    args = parser.parse_args()

    ratios_sum = args.train + args.validation + args.external_test
    if abs(ratios_sum - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios_sum:.6f}")

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"source file not found: {source}")

    data = json.loads(source.read_text(encoding="utf-8"))
    markets = data.get("markets", [])
    if not isinstance(markets, list):
        raise ValueError("invalid data format: markets must be a list")

    if args.include_unrated:
        filtered = markets
    else:
        filtered = [m for m in markets if m.get("mqr_rating")]

    keys = [stable_market_key(m) for m in filtered]
    unique_keys = list(dict.fromkeys(keys))

    if not unique_keys:
        raise ValueError("no eligible markets found to split")

    rng = random.Random(args.seed)
    rng.shuffle(unique_keys)

    n = len(unique_keys)
    n_train = int(n * args.train)
    n_val = int(n * args.validation)
    # external_test takes the remainder to avoid rounding-loss
    n_test = n - n_train - n_val

    train_keys = unique_keys[:n_train]
    val_keys = unique_keys[n_train:n_train + n_val]
    test_keys = unique_keys[n_train + n_val:]

    out_dir = Path("data/splits")
    write_json(out_dir / "train.json", train_keys)
    write_json(out_dir / "validation.json", val_keys)
    write_json(out_dir / "external_test.json", test_keys)

    manifest = {
        "split_version": "v1",
        "seed": args.seed,
        "ratios": {
            "train": args.train,
            "validation": args.validation,
            "external_test": args.external_test,
        },
        "source": str(source),
        "include_unrated": bool(args.include_unrated),
        "counts": {
            "total": n,
            "train": len(train_keys),
            "validation": len(val_keys),
            "external_test": len(test_keys),
        },
    }
    write_json(out_dir / "manifest.json", manifest)

    print("Split generation completed.")
    print(f"Total: {n}")
    print(f"Train: {len(train_keys)}")
    print(f"Validation: {len(val_keys)}")
    print(f"External test: {len(test_keys)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
