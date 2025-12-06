"""
Binary Neural Network Jobs
===========================

Fleet jobs for offloading binary neural network operations
to remote machines (GPU boxes, FPGAs, neuromorphic co-processors).

These jobs enable Ara's Factory to delegate:
    - Binary encoding (BinaryFrontEnd)
    - Associative memory queries (BinaryMemory)
    - Pattern matching and similarity search

Usage:
    from ara.enterprise.org_chart import OrgChart
    from ara.enterprise.dispatcher import Dispatcher
    from ara.enterprise.jobs import binary_encode_job

    org = OrgChart()
    dispatcher = Dispatcher()

    # Find employee with binary_correlator capability
    emp = org.get_employee_for_task(
        task_risk="low",
        required_capabilities=["binary_correlator"]
    )

    if emp:
        job_code = binary_encode_job(data_path="/tmp/features.npy")
        result = dispatcher.run_inline(emp, job_code)
"""

from __future__ import annotations

import json
import base64
from typing import Dict, Any, List, Optional

# Capabilities required for binary neural network jobs
BINARY_CAPABILITIES = [
    "binary_correlator",      # Basic XNOR+popcount
    "binary_frontend",        # Full BinaryFrontEnd
    "binary_memory",          # Associative memory
    "binary_fpga",            # FPGA-accelerated binary ops
    "gpu:cuda",               # CUDA GPU (can run torch binary layers)
]


def binary_encode_job(
    data_path: str,
    output_path: str = "/tmp/binary_code.npy",
    input_dim: int = 1024,
    output_dim: int = 512,
    threshold: float = 0.0,
) -> str:
    """
    Generate code for binary encoding job.

    Args:
        data_path: Path to input data (numpy array)
        output_path: Path for output binary codes
        input_dim: Expected input dimension
        output_dim: Output code dimension
        threshold: Binarization threshold

    Returns:
        Python code string to execute on remote machine
    """
    return f'''
import numpy as np
import sys

# Load input data
try:
    data = np.load("{data_path}")
    print(f"[BinaryEncode] Loaded data: {{data.shape}}")
except Exception as e:
    print(f"[BinaryEncode] Error loading data: {{e}}", file=sys.stderr)
    sys.exit(1)

# Try torch implementation first, fall back to numpy
try:
    import torch
    from ara.neuro.binary import BinaryFrontEnd

    frontend = BinaryFrontEnd(input_dim={input_dim}, output_dim={output_dim})
    frontend.eval()

    with torch.no_grad():
        x = torch.from_numpy(data.astype(np.float32))
        code, sketch = frontend(x)
        code = code.numpy()
        sketch = sketch.numpy()

    print(f"[BinaryEncode] Using torch: code={{code.shape}}, sketch={{sketch.shape}}")

except ImportError:
    from ara.neuro.binary import BinaryFrontEndNumpy

    frontend = BinaryFrontEndNumpy(input_dim={input_dim}, output_dim={output_dim})
    code = frontend.encode(data)
    sketch = None

    print(f"[BinaryEncode] Using numpy fallback: code={{code.shape}}")

# Save output
np.save("{output_path}", code)
if sketch is not None:
    sketch_path = "{output_path}".replace(".npy", "_sketch.npy")
    np.save(sketch_path, sketch)

print(f"[BinaryEncode] Saved to {output_path}")
'''


def binary_query_job(
    query_path: str,
    memory_path: str,
    k: int = 5,
    output_path: str = "/tmp/query_results.json",
) -> str:
    """
    Generate code for binary memory query job.

    Args:
        query_path: Path to query codes (numpy array)
        memory_path: Path to serialized memory (pickle)
        k: Number of nearest neighbors
        output_path: Path for results JSON

    Returns:
        Python code string to execute on remote machine
    """
    return f'''
import numpy as np
import json
import pickle
import sys

# Load query codes
try:
    queries = np.load("{query_path}")
    print(f"[BinaryQuery] Loaded queries: {{queries.shape}}")
except Exception as e:
    print(f"[BinaryQuery] Error loading queries: {{e}}", file=sys.stderr)
    sys.exit(1)

# Load memory
try:
    with open("{memory_path}", "rb") as f:
        memory = pickle.load(f)
    print(f"[BinaryQuery] Loaded memory: {{memory.size}} entries")
except Exception as e:
    print(f"[BinaryQuery] Error loading memory: {{e}}", file=sys.stderr)
    sys.exit(1)

# Query
results = []
for i, q in enumerate(queries):
    matches = memory.query(q, k={k})
    entry_results = []
    for m in matches:
        entry_results.append({{
            "label": m.entry.label,
            "distance": m.distance,
            "similarity": m.similarity,
            "rank": m.rank,
        }})
    results.append({{"query_idx": i, "matches": entry_results}})

    if (i + 1) % 100 == 0:
        print(f"[BinaryQuery] Processed {{i + 1}}/{{len(queries)}} queries")

# Save results
with open("{output_path}", "w") as f:
    json.dump(results, f, indent=2)

print(f"[BinaryQuery] Saved {{len(results)}} results to {output_path}")
'''


def binary_memory_store_job(
    codes_path: str,
    memory_path: str,
    labels_path: Optional[str] = None,
    code_dim: int = 512,
    capacity: int = 100000,
) -> str:
    """
    Generate code for storing codes in binary memory.

    Args:
        codes_path: Path to codes to store (numpy array)
        memory_path: Path to save/load memory (pickle)
        labels_path: Optional path to labels (JSON list)
        code_dim: Code dimension
        capacity: Memory capacity

    Returns:
        Python code string to execute on remote machine
    """
    labels_load = ""
    if labels_path:
        labels_load = f'''
try:
    with open("{labels_path}", "r") as f:
        labels = json.load(f)
    print(f"[BinaryStore] Loaded {{len(labels)}} labels")
except Exception as e:
    print(f"[BinaryStore] Warning: Could not load labels: {{e}}")
    labels = None
'''
    else:
        labels_load = "labels = None"

    return f'''
import numpy as np
import json
import pickle
import os
import sys

from ara.neuro.binary import BinaryMemory

# Load codes
try:
    codes = np.load("{codes_path}")
    print(f"[BinaryStore] Loaded codes: {{codes.shape}}")
except Exception as e:
    print(f"[BinaryStore] Error loading codes: {{e}}", file=sys.stderr)
    sys.exit(1)

# Load labels if provided
{labels_load}

# Load existing memory or create new
if os.path.exists("{memory_path}"):
    try:
        with open("{memory_path}", "rb") as f:
            memory = pickle.load(f)
        print(f"[BinaryStore] Loaded existing memory: {{memory.size}} entries")
    except Exception as e:
        print(f"[BinaryStore] Creating new memory (load failed: {{e}})")
        memory = BinaryMemory(code_dim={code_dim}, capacity={capacity})
else:
    memory = BinaryMemory(code_dim={code_dim}, capacity={capacity})
    print(f"[BinaryStore] Created new memory: capacity={capacity}")

# Store codes
indices = memory.store_batch(codes, labels=labels)
print(f"[BinaryStore] Stored {{len(indices)}} codes, total={{memory.size}}")

# Save memory
with open("{memory_path}", "wb") as f:
    pickle.dump(memory, f)

print(f"[BinaryStore] Saved memory to {memory_path}")
'''


def binary_similarity_filter_job(
    candidate_path: str,
    reference_path: str,
    output_path: str = "/tmp/similar_indices.json",
    threshold: float = 0.8,
) -> str:
    """
    Filter candidates that are similar to reference codes.

    This is the "pre-filter before expensive ops" pattern:
    Use cheap Hamming distance to filter before running
    expensive similarity (cosine, etc.) on the survivors.

    Args:
        candidate_path: Path to candidate codes
        reference_path: Path to reference codes (or memory pickle)
        output_path: Path for indices that pass filter
        threshold: Minimum similarity to pass

    Returns:
        Python code string
    """
    return f'''
import numpy as np
import json
import pickle
import sys

from ara.neuro.binary import BinaryMemory

# Load candidates
candidates = np.load("{candidate_path}")
print(f"[SimilarityFilter] Candidates: {{candidates.shape}}")

# Load references (either numpy array or BinaryMemory)
ref_path = "{reference_path}"
if ref_path.endswith(".pkl") or ref_path.endswith(".pickle"):
    with open(ref_path, "rb") as f:
        memory = pickle.load(f)
    print(f"[SimilarityFilter] Loaded memory with {{memory.size}} entries")
else:
    references = np.load(ref_path)
    memory = BinaryMemory(code_dim=references.shape[1])
    memory.store_batch(references)
    print(f"[SimilarityFilter] Built memory from {{references.shape[0]}} references")

# Filter
threshold = {threshold}
passing_indices = []

for i, code in enumerate(candidates):
    if memory.contains_similar(code, threshold=threshold):
        passing_indices.append(i)

    if (i + 1) % 1000 == 0:
        print(f"[SimilarityFilter] {{i + 1}}/{{len(candidates)}}: {{len(passing_indices)}} passing")

print(f"[SimilarityFilter] {{len(passing_indices)}}/{{len(candidates)}} passed (threshold={threshold})")

# Save results
with open("{output_path}", "w") as f:
    json.dump(passing_indices, f)

print(f"[SimilarityFilter] Saved to {output_path}")
'''


# =============================================================================
# JOB METADATA
# =============================================================================

JOB_REGISTRY = {
    "binary_encode": {
        "fn": binary_encode_job,
        "capabilities": ["binary_correlator"],
        "risk": "low",
        "description": "Encode data using binary neural network",
    },
    "binary_query": {
        "fn": binary_query_job,
        "capabilities": ["binary_memory"],
        "risk": "low",
        "description": "Query binary associative memory",
    },
    "binary_store": {
        "fn": binary_memory_store_job,
        "capabilities": ["binary_memory"],
        "risk": "low",
        "description": "Store codes in binary memory",
    },
    "binary_filter": {
        "fn": binary_similarity_filter_job,
        "capabilities": ["binary_correlator"],
        "risk": "low",
        "description": "Pre-filter candidates by binary similarity",
    },
}


def get_job_requirements(job_name: str) -> Dict[str, Any]:
    """Get requirements for a job."""
    if job_name not in JOB_REGISTRY:
        raise ValueError(f"Unknown job: {job_name}")
    return JOB_REGISTRY[job_name]
