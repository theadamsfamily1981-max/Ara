#!/usr/bin/env python3
"""
HDC Hardware Co-Verification Test
==================================

Generates test vectors for verifying the bit_serial_neuron HDC mode.

Usage:
    1. Run this script to generate test vectors
    2. Feed vectors to Verilog simulation
    3. Compare hardware output with expected values

The script also serves as a reference implementation for the
HolographicProcessor Python code to ensure software/hardware match.
"""

import numpy as np
from pathlib import Path


def generate_hypervector(dim: int = 8192, seed: int = None) -> np.ndarray:
    """Generate a random binary hypervector."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=dim, dtype=np.uint8)


def bind_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HDC Bind operation: element-wise XOR."""
    return np.bitwise_xor(a, b)


def bundle_majority(vectors: list) -> np.ndarray:
    """HDC Bundle operation: majority vote."""
    if not vectors:
        return np.zeros_like(vectors[0])

    # Convert to bipolar {-1, +1} for voting
    bipolar = [np.where(v > 0.5, 1, -1) for v in vectors]
    summed = np.sum(bipolar, axis=0)

    # Majority vote (random tie-breaking)
    result = (summed > 0).astype(np.uint8)
    ties = (summed == 0)
    result[ties] = np.random.randint(0, 2, size=np.sum(ties))

    return result


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Hamming similarity in [0, 1]."""
    return 1.0 - (np.sum(a != b) / len(a))


def generate_test_vectors(output_dir: Path, dim: int = 64, n_tests: int = 10):
    """
    Generate test vectors for hardware simulation.

    Creates files:
        - hv_a.hex: Hypervector A (one line per test)
        - hv_b.hex: Hypervector B (one line per test)
        - hv_c_expected.hex: Expected A XOR B
        - test_info.txt: Human-readable test info
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_tests} test vectors with dim={dim}")

    # Storage
    all_a = []
    all_b = []
    all_c = []

    with open(output_dir / "test_info.txt", "w") as info_file:
        info_file.write(f"HDC Bind Test Vectors\n")
        info_file.write(f"Dimension: {dim}\n")
        info_file.write(f"Tests: {n_tests}\n")
        info_file.write(f"{'='*50}\n\n")

        for test_id in range(n_tests):
            # Generate random hypervectors
            hv_a = generate_hypervector(dim, seed=test_id * 2)
            hv_b = generate_hypervector(dim, seed=test_id * 2 + 1)

            # Compute expected bind
            hv_c = bind_xor(hv_a, hv_b)

            all_a.append(hv_a)
            all_b.append(hv_b)
            all_c.append(hv_c)

            # Log
            info_file.write(f"Test {test_id}:\n")
            info_file.write(f"  A[0:8]  = {hv_a[:8].tolist()}\n")
            info_file.write(f"  B[0:8]  = {hv_b[:8].tolist()}\n")
            info_file.write(f"  C[0:8]  = {hv_c[:8].tolist()}\n")
            info_file.write(f"  sim(A,B) = {similarity(hv_a, hv_b):.3f}\n")
            info_file.write(f"  sim(A,C) = {similarity(hv_a, hv_c):.3f}\n")
            info_file.write(f"  sim(B,C) = {similarity(hv_b, hv_c):.3f}\n\n")

    # Write hex files (one hypervector per line, bits packed into hex)
    def pack_to_hex(hv: np.ndarray) -> str:
        """Pack binary vector to hex string."""
        # Pad to multiple of 4
        padded = np.zeros(((len(hv) + 3) // 4) * 4, dtype=np.uint8)
        padded[:len(hv)] = hv

        # Pack 4 bits at a time into hex nibbles
        hex_chars = []
        for i in range(0, len(padded), 4):
            nibble = (padded[i] << 3) | (padded[i+1] << 2) | (padded[i+2] << 1) | padded[i+3]
            hex_chars.append(f"{nibble:x}")

        return ''.join(hex_chars)

    with open(output_dir / "hv_a.hex", "w") as f:
        for hv in all_a:
            f.write(pack_to_hex(hv) + "\n")

    with open(output_dir / "hv_b.hex", "w") as f:
        for hv in all_b:
            f.write(pack_to_hex(hv) + "\n")

    with open(output_dir / "hv_c_expected.hex", "w") as f:
        for hv in all_c:
            f.write(pack_to_hex(hv) + "\n")

    # Also write as binary files for direct memory loading
    np.save(output_dir / "hv_a.npy", np.array(all_a))
    np.save(output_dir / "hv_b.npy", np.array(all_b))
    np.save(output_dir / "hv_c_expected.npy", np.array(all_c))

    print(f"Test vectors written to {output_dir}/")
    print(f"  - hv_a.hex, hv_b.hex, hv_c_expected.hex (for Verilog $readmemh)")
    print(f"  - hv_a.npy, hv_b.npy, hv_c_expected.npy (for Python verification)")


def verify_hardware_output(output_dir: Path):
    """
    Verify hardware output against expected values.

    Expects hardware output in: output_dir/hv_c_actual.hex or .npy
    """
    output_dir = Path(output_dir)

    # Load expected
    expected = np.load(output_dir / "hv_c_expected.npy")

    # Try to load actual
    actual_path = output_dir / "hv_c_actual.npy"
    if not actual_path.exists():
        print(f"No hardware output found at {actual_path}")
        print("Run Verilog simulation and save output to hv_c_actual.npy")
        return

    actual = np.load(actual_path)

    # Compare
    n_tests = len(expected)
    errors = 0

    for i in range(n_tests):
        if np.array_equal(expected[i], actual[i]):
            print(f"Test {i}: PASS")
        else:
            diff_bits = np.sum(expected[i] != actual[i])
            print(f"Test {i}: FAIL ({diff_bits} bits differ)")
            errors += 1

    print(f"\n{'='*40}")
    if errors == 0:
        print(f"ALL {n_tests} TESTS PASSED!")
    else:
        print(f"{errors}/{n_tests} tests failed")


def run_software_reference_test():
    """
    Run software reference test to verify Python implementation.

    This ensures the HolographicProcessor matches the hardware model.
    """
    print("\n" + "="*50)
    print("Software Reference Test")
    print("="*50 + "\n")

    dim = 8192

    # Test 1: Bind is self-inverse
    print("Test 1: Bind self-inverse (A ⊗ A = 0)")
    a = generate_hypervector(dim, seed=42)
    a_bind_a = bind_xor(a, a)
    assert np.all(a_bind_a == 0), "FAIL: A XOR A should be 0"
    print("  PASS: A XOR A = 0 (all zeros)")

    # Test 2: Bind is commutative
    print("\nTest 2: Bind commutativity (A ⊗ B = B ⊗ A)")
    b = generate_hypervector(dim, seed=43)
    ab = bind_xor(a, b)
    ba = bind_xor(b, a)
    assert np.array_equal(ab, ba), "FAIL: A XOR B != B XOR A"
    print("  PASS: A XOR B = B XOR A")

    # Test 3: Unbind recovers original
    print("\nTest 3: Unbind recovery (A ⊗ B ⊗ B = A)")
    c = bind_xor(a, b)
    c_unbind_b = bind_xor(c, b)  # Should recover A
    assert np.array_equal(c_unbind_b, a), "FAIL: Unbind didn't recover original"
    print("  PASS: (A XOR B) XOR B = A")

    # Test 4: Bundle preserves similarity
    print("\nTest 4: Bundle similarity preservation")
    vectors = [generate_hypervector(dim, seed=i) for i in range(5)]
    bundled = bundle_majority(vectors)

    avg_sim = np.mean([similarity(bundled, v) for v in vectors])
    print(f"  Average similarity to bundled vector: {avg_sim:.3f}")
    assert avg_sim > 0.55, f"FAIL: Bundle should be similar to components (got {avg_sim})"
    print("  PASS: Bundled vector similar to all components")

    # Test 5: Random vectors are nearly orthogonal
    print("\nTest 5: Random vector orthogonality")
    r1 = generate_hypervector(dim, seed=100)
    r2 = generate_hypervector(dim, seed=200)
    sim = similarity(r1, r2)
    print(f"  Similarity of two random HVs: {sim:.3f}")
    assert 0.45 < sim < 0.55, f"FAIL: Random vectors should be ~0.5 similar (got {sim})"
    print("  PASS: Random vectors are nearly orthogonal (~0.5)")

    print("\n" + "="*50)
    print("ALL SOFTWARE REFERENCE TESTS PASSED!")
    print("="*50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HDC Hardware Test Vector Generator")
    parser.add_argument("--generate", action="store_true", help="Generate test vectors")
    parser.add_argument("--verify", action="store_true", help="Verify hardware output")
    parser.add_argument("--reference", action="store_true", help="Run software reference test")
    parser.add_argument("--output", type=str, default="./test_vectors", help="Output directory")
    parser.add_argument("--dim", type=int, default=64, help="Hypervector dimension")
    parser.add_argument("--tests", type=int, default=10, help="Number of tests")

    args = parser.parse_args()

    if args.reference or (not args.generate and not args.verify):
        run_software_reference_test()

    if args.generate:
        generate_test_vectors(Path(args.output), dim=args.dim, n_tests=args.tests)

    if args.verify:
        verify_hardware_output(Path(args.output))
