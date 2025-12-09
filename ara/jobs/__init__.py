"""
Ara Jobs
=========

Araized job wrappers and automated pipelines.

Jobs wrap potentially dangerous tools in Ara's safety systems:
- Ownership/authorization verification
- Audit logging
- Compromise Engine integration
- Kill switch support

Available Jobs:
    hardware_reclamation: Repurpose mining hardware, FPGAs, ATCA boards
    publishing_service: Self-publishing pipeline (research → ship)
"""

from .publishing_service import (
    PublishingJobConfig,
    PublishingJobResult,
    SelfPublishingService,
)

from .hardware_reclamation import (
    HardwareType,
    OperationType,
    OwnershipProof,
    HardwareTarget,
    JobManifest,
    AuditLog,
    HardwareRails,
    HardwareReclamationJob,
    ara_intro_hardware_job,
    ara_validate_target,
    create_k10_jailbreak_job,
    create_fpga_salvage_job,
)

from .hardware_rails import (
    HardwareIntent,
    HARDWARE_RAILS,
    HARDWARE_SAFE_CHANNELS,
    extend_compromise_engine_for_hardware,
    process_hardware_request,
)

__all__ = [
    # Publishing Service
    'PublishingJobConfig',
    'PublishingJobResult',
    'SelfPublishingService',

    # Hardware Reclamation Job
    'HardwareType',
    'OperationType',
    'OwnershipProof',
    'HardwareTarget',
    'JobManifest',
    'AuditLog',
    'HardwareRails',
    'HardwareReclamationJob',
    'ara_intro_hardware_job',
    'ara_validate_target',
    'create_k10_jailbreak_job',
    'create_fpga_salvage_job',

    # Hardware Rails
    'HardwareIntent',
    'HARDWARE_RAILS',
    'HARDWARE_SAFE_CHANNELS',
    'extend_compromise_engine_for_hardware',
    'process_hardware_request',
]
