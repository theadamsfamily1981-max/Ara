"""
Fleet Jobs Package
==================

Predefined job types that can be dispatched to Fleet employees.

Jobs:
    binary_encode: Encode data using binary neural network
    binary_query: Query binary associative memory
    snn_preproc: Preprocess data for SNN core

Each job is a function that:
    1. Takes data + config
    2. Returns a code snippet to run on remote machine
    3. Includes required capabilities for employee selection
"""

from ara.enterprise.jobs.binary_jobs import (
    binary_encode_job,
    binary_query_job,
    binary_memory_store_job,
    BINARY_CAPABILITIES,
)

__all__ = [
    'binary_encode_job',
    'binary_query_job',
    'binary_memory_store_job',
    'BINARY_CAPABILITIES',
]
