"""Resource Gate - Pre-flight checks for resource-intensive operations.

This module prevents resource-intensive operations (like video generation)
from starting when system resources are insufficient. It integrates with:
- EmbodimentCore for hardware health
- CLV (Cognitive Load Vector) for system stress assessment
- GPU memory monitoring for VRAM availability

Usage:
    from ara.embodied.resource_gate import (
        ResourceGate,
        can_generate_video,
        get_recommended_quality,
        ResourceLevel,
    )

    # Quick check
    if can_generate_video():
        generate_video()
    else:
        print("Resources insufficient")

    # Get recommended quality based on resources
    quality = get_recommended_quality()  # "1080p", "720p", "480p", or "skip"

    # Detailed gate
    gate = ResourceGate()
    result = gate.check_video_generation()
    if result.allowed:
        generate_video(quality=result.recommended_quality)
    else:
        print(f"Blocked: {result.reason}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ResourceLevel(str, Enum):
    """Resource availability levels."""
    ABUNDANT = "abundant"      # Resources plentiful, full quality
    ADEQUATE = "adequate"      # Enough resources, standard quality
    CONSTRAINED = "constrained"  # Limited resources, reduce quality
    CRITICAL = "critical"      # Resources too low, skip operation
    UNKNOWN = "unknown"        # Cannot determine resource state


class VideoQuality(str, Enum):
    """Video quality levels for adaptive generation."""
    FULL_1080P = "1080p"       # Full HD, highest quality
    STANDARD_720P = "720p"    # HD, good quality
    LOW_480P = "480p"         # SD, reduced quality
    SKIP = "skip"             # Don't generate


@dataclass
class ResourceCheckResult:
    """Result of a resource availability check."""
    allowed: bool = False
    resource_level: ResourceLevel = ResourceLevel.UNKNOWN
    recommended_quality: VideoQuality = VideoQuality.SKIP
    reason: str = ""

    # Detailed metrics
    gpu_memory_available_gb: float = 0.0
    gpu_memory_required_gb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_percent: float = 0.0
    ram_available_gb: float = 0.0

    # CLV integration
    clv_resource: float = 0.0
    clv_instability: float = 0.0
    clv_risk_level: str = "unknown"

    # Health status
    embodiment_health: str = "unknown"

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "resource_level": self.resource_level.value,
            "recommended_quality": self.recommended_quality.value,
            "reason": self.reason,
            "gpu_memory_available_gb": self.gpu_memory_available_gb,
            "gpu_memory_required_gb": self.gpu_memory_required_gb,
            "gpu_utilization": self.gpu_utilization,
            "cpu_percent": self.cpu_percent,
            "ram_available_gb": self.ram_available_gb,
            "clv_resource": self.clv_resource,
            "clv_instability": self.clv_instability,
            "clv_risk_level": self.clv_risk_level,
            "embodiment_health": self.embodiment_health,
            "timestamp": self.timestamp,
        }


class ResourceGate:
    """
    Resource gating for video generation and other intensive operations.

    Implements pre-flight checks to prevent resource-intensive operations
    from starting when the system cannot handle them gracefully.
    """

    # Memory requirements by quality (in GB)
    MEMORY_REQUIREMENTS = {
        VideoQuality.FULL_1080P: 6.0,   # Full HD needs ~6GB VRAM
        VideoQuality.STANDARD_720P: 3.5, # HD needs ~3.5GB VRAM
        VideoQuality.LOW_480P: 2.0,      # SD needs ~2GB VRAM
    }

    # Safety margin (20% headroom)
    SAFETY_MARGIN = 0.2

    # CLV thresholds for quality decisions
    CLV_THRESHOLDS = {
        "allow_1080p": 0.3,      # CLV.resource <= 0.3 for 1080p
        "allow_720p": 0.5,       # CLV.resource <= 0.5 for 720p
        "allow_480p": 0.7,       # CLV.resource <= 0.7 for 480p
        "block_all": 0.85,       # CLV.resource > 0.85 blocks all
    }

    def __init__(self):
        """Initialize the resource gate."""
        self._torch_available = False
        self._psutil_available = False
        self._embodied_available = False
        self._clv_available = False

        # Try to import optional dependencies
        try:
            import torch
            self._torch = torch
            self._torch_available = torch.cuda.is_available()
        except ImportError:
            self._torch = None

        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            self._psutil = None

        try:
            from ara.embodied import get_embodiment_core
            self._get_embodiment_core = get_embodiment_core
            self._embodied_available = True
        except ImportError:
            self._get_embodiment_core = None

        try:
            from tfan.system.cognitive_load_vector import CLVComputer
            self._clv_computer = CLVComputer()
            self._clv_available = True
        except ImportError:
            self._clv_computer = None

    def get_gpu_memory(self) -> tuple[float, float]:
        """Get GPU memory (available_gb, total_gb)."""
        if not self._torch_available:
            return (0.0, 0.0)

        try:
            device = self._torch.cuda.current_device()
            total = self._torch.cuda.get_device_properties(device).total_memory
            reserved = self._torch.cuda.memory_reserved(device)
            allocated = self._torch.cuda.memory_allocated(device)

            # Available = total - reserved (conservative estimate)
            available = (total - reserved) / (1024**3)  # Convert to GB
            total_gb = total / (1024**3)

            return (available, total_gb)
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return (0.0, 0.0)

    def get_gpu_utilization(self) -> float:
        """Get current GPU utilization (0-1)."""
        if not self._torch_available:
            return 0.0

        try:
            # Try nvidia-smi via pynvml
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2.0
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 100.0
        except Exception:
            pass

        return 0.0  # Unknown

    def get_system_memory(self) -> tuple[float, float]:
        """Get system RAM (available_gb, total_gb)."""
        if not self._psutil_available:
            return (8.0, 16.0)  # Assume reasonable defaults

        try:
            mem = self._psutil.virtual_memory()
            available = mem.available / (1024**3)
            total = mem.total / (1024**3)
            return (available, total)
        except Exception as e:
            logger.warning(f"Failed to get system memory: {e}")
            return (8.0, 16.0)

    def get_cpu_percent(self) -> float:
        """Get CPU utilization (0-100)."""
        if not self._psutil_available:
            return 50.0  # Assume moderate load

        try:
            return self._psutil.cpu_percent(interval=0.1)
        except Exception:
            return 50.0

    def get_embodiment_health(self) -> tuple[str, Dict[str, Any]]:
        """Get embodiment health status."""
        if not self._embodied_available:
            return ("unknown", {})

        try:
            core = self._get_embodiment_core()
            health = core.check_health()
            status = health.get("overall_status", "unknown")
            return (status, health)
        except Exception as e:
            logger.warning(f"Failed to get embodiment health: {e}")
            return ("unknown", {})

    def get_clv_state(self) -> tuple[float, float, str]:
        """Get CLV state (resource, instability, risk_level)."""
        if not self._clv_available:
            return (0.5, 0.5, "unknown")  # Moderate defaults

        try:
            # Get current CLV from computer
            # Note: In practice, this would come from the running system
            # For now, estimate from available metrics
            gpu_util = self.get_gpu_utilization()
            cpu_pct = self.get_cpu_percent() / 100.0

            # Estimate CLV.resource from GPU and CPU utilization
            resource = (gpu_util * 0.6) + (cpu_pct * 0.4)

            # Estimate instability (would normally come from EPR-CV)
            instability = 0.3 if resource < 0.5 else 0.5

            # Determine risk level
            if resource < 0.3:
                risk = "nominal"
            elif resource < 0.5:
                risk = "elevated"
            elif resource < 0.7:
                risk = "warning"
            else:
                risk = "critical"

            return (resource, instability, risk)
        except Exception as e:
            logger.warning(f"Failed to get CLV state: {e}")
            return (0.5, 0.5, "unknown")

    def check_video_generation(
        self,
        requested_quality: Optional[VideoQuality] = None,
    ) -> ResourceCheckResult:
        """
        Check if video generation should proceed.

        Args:
            requested_quality: Specific quality requested (None = auto-select)

        Returns:
            ResourceCheckResult with decision and recommendations
        """
        result = ResourceCheckResult()

        # 1. Gather metrics
        gpu_avail, gpu_total = self.get_gpu_memory()
        ram_avail, ram_total = self.get_system_memory()
        gpu_util = self.get_gpu_utilization()
        cpu_pct = self.get_cpu_percent()
        health_status, health_data = self.get_embodiment_health()
        clv_resource, clv_instability, clv_risk = self.get_clv_state()

        # Populate result metrics
        result.gpu_memory_available_gb = gpu_avail
        result.gpu_utilization = gpu_util
        result.cpu_percent = cpu_pct
        result.ram_available_gb = ram_avail
        result.clv_resource = clv_resource
        result.clv_instability = clv_instability
        result.clv_risk_level = clv_risk
        result.embodiment_health = health_status

        # 2. Check hard blockers

        # Health check
        if health_status in ["POOR", "CRITICAL"]:
            result.allowed = False
            result.resource_level = ResourceLevel.CRITICAL
            result.recommended_quality = VideoQuality.SKIP
            result.reason = f"System health is {health_status}, video generation blocked"
            return result

        # CLV check - block if too stressed
        if clv_resource > self.CLV_THRESHOLDS["block_all"]:
            result.allowed = False
            result.resource_level = ResourceLevel.CRITICAL
            result.recommended_quality = VideoQuality.SKIP
            result.reason = f"CLV.resource={clv_resource:.2f} exceeds threshold, system too stressed"
            return result

        # GPU memory check - need at least minimum for 480p
        min_required = self.MEMORY_REQUIREMENTS[VideoQuality.LOW_480P]
        if gpu_avail < min_required * (1 + self.SAFETY_MARGIN):
            result.allowed = False
            result.resource_level = ResourceLevel.CRITICAL
            result.gpu_memory_required_gb = min_required
            result.recommended_quality = VideoQuality.SKIP
            result.reason = f"GPU memory {gpu_avail:.1f}GB < {min_required:.1f}GB required"
            return result

        # 3. Determine best quality based on resources
        selected_quality = VideoQuality.SKIP
        resource_level = ResourceLevel.CRITICAL

        # Check each quality level from highest to lowest
        for quality in [VideoQuality.FULL_1080P, VideoQuality.STANDARD_720P, VideoQuality.LOW_480P]:
            required_mem = self.MEMORY_REQUIREMENTS[quality]

            # Check GPU memory
            if gpu_avail < required_mem * (1 + self.SAFETY_MARGIN):
                continue

            # Check CLV threshold for this quality
            if quality == VideoQuality.FULL_1080P:
                if clv_resource > self.CLV_THRESHOLDS["allow_1080p"]:
                    continue
                resource_level = ResourceLevel.ABUNDANT
            elif quality == VideoQuality.STANDARD_720P:
                if clv_resource > self.CLV_THRESHOLDS["allow_720p"]:
                    continue
                resource_level = ResourceLevel.ADEQUATE
            elif quality == VideoQuality.LOW_480P:
                if clv_resource > self.CLV_THRESHOLDS["allow_480p"]:
                    continue
                resource_level = ResourceLevel.CONSTRAINED

            selected_quality = quality
            break

        # 4. Honor requested quality if possible
        if requested_quality is not None:
            req_mem = self.MEMORY_REQUIREMENTS.get(requested_quality, 999)
            if gpu_avail >= req_mem * (1 + self.SAFETY_MARGIN):
                # Check CLV allows this quality
                if requested_quality == VideoQuality.FULL_1080P:
                    if clv_resource <= self.CLV_THRESHOLDS["allow_1080p"]:
                        selected_quality = requested_quality
                elif requested_quality == VideoQuality.STANDARD_720P:
                    if clv_resource <= self.CLV_THRESHOLDS["allow_720p"]:
                        selected_quality = requested_quality
                elif requested_quality == VideoQuality.LOW_480P:
                    if clv_resource <= self.CLV_THRESHOLDS["allow_480p"]:
                        selected_quality = requested_quality

        # 5. Build result
        if selected_quality != VideoQuality.SKIP:
            result.allowed = True
            result.resource_level = resource_level
            result.recommended_quality = selected_quality
            result.gpu_memory_required_gb = self.MEMORY_REQUIREMENTS[selected_quality]
            result.reason = f"Resources sufficient for {selected_quality.value}"
        else:
            result.allowed = False
            result.resource_level = ResourceLevel.CONSTRAINED
            result.recommended_quality = VideoQuality.SKIP
            result.reason = "Resources insufficient for any quality level"

        logger.info(
            f"Resource gate: allowed={result.allowed}, "
            f"quality={result.recommended_quality.value}, "
            f"gpu_avail={gpu_avail:.1f}GB, clv_resource={clv_resource:.2f}"
        )

        return result

    def reserve_memory(self, quality: VideoQuality) -> bool:
        """
        Pre-allocate memory for the requested quality level.

        This prevents OOM by ensuring memory is available before starting.
        """
        if not self._torch_available:
            return True  # Can't reserve, assume OK

        required = self.MEMORY_REQUIREMENTS.get(quality, 0)
        if required == 0:
            return True

        try:
            # Allocate a test tensor to reserve memory
            bytes_needed = int(required * (1024**3))
            test = self._torch.cuda.FloatTensor(bytes_needed // 4)
            del test
            self._torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            logger.warning(f"Memory reservation failed: {e}")
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

_default_gate: Optional[ResourceGate] = None


def get_resource_gate() -> ResourceGate:
    """Get the default resource gate (singleton)."""
    global _default_gate
    if _default_gate is None:
        _default_gate = ResourceGate()
    return _default_gate


def can_generate_video(quality: Optional[str] = None) -> bool:
    """
    Quick check if video generation is allowed.

    Args:
        quality: Optional quality level ("1080p", "720p", "480p")

    Returns:
        True if video generation is allowed
    """
    gate = get_resource_gate()

    requested = None
    if quality:
        quality_map = {
            "1080p": VideoQuality.FULL_1080P,
            "720p": VideoQuality.STANDARD_720P,
            "480p": VideoQuality.LOW_480P,
        }
        requested = quality_map.get(quality)

    result = gate.check_video_generation(requested_quality=requested)
    return result.allowed


def get_recommended_quality() -> str:
    """
    Get the recommended video quality based on current resources.

    Returns:
        Quality string: "1080p", "720p", "480p", or "skip"
    """
    gate = get_resource_gate()
    result = gate.check_video_generation()
    return result.recommended_quality.value


def check_resources() -> Dict[str, Any]:
    """
    Get detailed resource check results.

    Returns:
        Dictionary with resource status and recommendations
    """
    gate = get_resource_gate()
    result = gate.check_video_generation()
    return result.to_dict()
