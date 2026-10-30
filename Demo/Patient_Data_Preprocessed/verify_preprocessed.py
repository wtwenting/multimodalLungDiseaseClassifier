
"""Verify preprocessed CT and X-ray inputs for frozen fusion."""

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

CT_SHAPE = (64, 192, 192)
XRAY_SHAPE = (224, 224, 3)


def check_array(name, array, expected_shape):
    """Validate tensor shape, type, range and finite values."""
    print(f"\n{name}")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")

    valid = True

    if array.shape != expected_shape:
        print(f"  FAIL: Expected {expected_shape}")
        valid = False

    if not np.issubdtype(array.dtype, np.floating):
        print("  FAIL: Expected floating-point tensor")
        valid = False

    if not np.isfinite(array).all():
        print("  FAIL: NaN or infinite values detected")
        valid = False
    else:
        minimum = float(array.min())
        maximum = float(array.max())

        print(f"  Range: {minimum:.6f} to {maximum:.6f}")
        print(f"  Mean: {array.mean():.6f}")
        print(f"  Std: {array.std():.6f}")

        if minimum < -0.01 or maximum > 1.01:
            print("  FAIL: Expected normalization 0–1")
            valid = False

    print("  RESULT:", "PASS" if valid else "FAIL")
    return valid


def verify_ct(path):
    """Read the V5 preprocessed CT tensor."""
    with np.load(path, allow_pickle=False) as data:
        print("CT archive keys:", data.files)

        if "volume" in data.files:
            array = data["volume"]
        elif "tensor" in data.files:
            array = data["tensor"]
        else:
            raise ValueError(
                f"Unknown CT archive keys: {data.files}"
            )

        return check_array(
            path.name, array, CT_SHAPE
        )


def verify_xray(path):
    """Read a preprocessed X-ray encoder input."""
    array = np.load(path, allow_pickle=False)

    return check_array(
        path.name, array, XRAY_SHAPE
    )


def main():
    ct_path = ROOT / "ct_v5_preprocessed.npz"

    files = [
        ROOT / f"image_{i:02d}_model_{model}_input.npy"
        for i in range(1, 4)
        for model in ("b", "c")
    ]

    results = []

    for path, verifier in [
        (ct_path, verify_ct),
        *[(p, verify_xray) for p in files],
    ]:
        try:
            if not path.is_file():
                raise FileNotFoundError(path)

            results.append(verifier(path))

        except Exception as exc:
            print(f"\n{path.name}: FAIL — {exc}")
            results.append(False)

    print("\n" + "=" * 45)
    print(f"Passed: {sum(results)}/{len(results)}")
    print("STATUS:", "ALL PASS" if all(results) else "REVIEW REQUIRED")


if __name__ == "__main__":
    main()
