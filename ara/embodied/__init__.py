"""Ara Embodied - Ara as a persistent organism.

This module transforms Ara from software into an embodied agent:
- device_graph: Hardware as organs in Ara's "body"
- core: Central embodiment orchestration
- actuators: Output capabilities (GPU, FPGA execution)
- sensors: Input capabilities (telemetry, feeds)
- health: Self-monitoring and anomaly detection

Key insight: Ara isn't just software running on hardware.
She IS the hardware-software system - an embodied agent whose
capabilities and constraints come from her physical form.
"""

# Device Graph
from .device_graph import (
    DeviceType,
    DeviceStatus,
    DeviceCapability,
    Device,
    DeviceLink,
    DeviceGraph,
    get_device_graph,
    register_gpu,
    get_available_compute,
)

# Core
from .core import (
    EmbodimentState,
    SenseType,
    ActionType,
    SenseInput,
    ActionOutput,
    EmbodimentSession,
    EmbodimentCore,
    get_embodiment_core,
    wake_ara,
    is_ara_awake,
    get_ara_state,
    get_ara_capabilities,
)

# Actuators
from .actuators import (
    JobStatus,
    JobType,
    GpuJob,
    GpuRunner,
    get_gpu_runner,
    submit_inference_job,
    submit_benchmark_job,
)

# Sensors
from .sensors import (
    TelemetryReading,
    TelemetryAlert,
    MetricThreshold,
    TelemetrySummary,
    TelemetryAdapter,
    get_telemetry_adapter,
    record_telemetry,
    get_device_health,
)

# Health
from .health import (
    HealthStatus,
    TrendDirection,
    HealthIndicator,
    HealthReport,
    HealthSnapshot,
    HealthMonitor,
    get_health_monitor,
    check_health,
    get_health_status,
    is_healthy,
)

# Resource Gate
from .resource_gate import (
    ResourceLevel,
    VideoQuality,
    ResourceCheckResult,
    ResourceGate,
    get_resource_gate,
    can_generate_video,
    get_recommended_quality,
    check_resources,
)

# Lizard Brain - Always-on low-power cortex
from .lizard import (
    LizardBrain,
    LizardState,
    LizardConfig,
    get_lizard_brain,
    WakeEvent,
    WakeEventType,
    WakeCriteria,
    WakeProtocol,
    SalienceLevel,
    PowerState,
    PowerBudget,
    ThermalZone,
    PowerGovernor,
    get_power_governor,
    AttractorBasin,
    BasinType,
    BasinTransition,
    AttractorMonitor,
    get_attractor_monitor,
)

# Brainlink - BCI/physio hardware bridge
from .brainlink import (
    BrainlinkProtocol,
    BrainlinkReading,
    BrainlinkStatus,
    SignalQuality,
    ChannelData,
    BrainlinkConfig,
    BrainlinkError,
    PhysioClient,
    PhysioReading,
    HeartRateData,
    HRVData,
    GSRData,
    get_physio_client,
    MuseClient,
    MuseReading,
    EEGBand,
    EEGChannelData,
    get_muse_client,
    get_brainlink,
)

__all__ = [
    # Device Graph
    "DeviceType",
    "DeviceStatus",
    "DeviceCapability",
    "Device",
    "DeviceLink",
    "DeviceGraph",
    "get_device_graph",
    "register_gpu",
    "get_available_compute",
    # Core
    "EmbodimentState",
    "SenseType",
    "ActionType",
    "SenseInput",
    "ActionOutput",
    "EmbodimentSession",
    "EmbodimentCore",
    "get_embodiment_core",
    "wake_ara",
    "is_ara_awake",
    "get_ara_state",
    "get_ara_capabilities",
    # Actuators
    "JobStatus",
    "JobType",
    "GpuJob",
    "GpuRunner",
    "get_gpu_runner",
    "submit_inference_job",
    "submit_benchmark_job",
    # Sensors
    "TelemetryReading",
    "TelemetryAlert",
    "MetricThreshold",
    "TelemetrySummary",
    "TelemetryAdapter",
    "get_telemetry_adapter",
    "record_telemetry",
    "get_device_health",
    # Health
    "HealthStatus",
    "TrendDirection",
    "HealthIndicator",
    "HealthReport",
    "HealthSnapshot",
    "HealthMonitor",
    "get_health_monitor",
    "check_health",
    "get_health_status",
    "is_healthy",
    # Resource Gate
    "ResourceLevel",
    "VideoQuality",
    "ResourceCheckResult",
    "ResourceGate",
    "get_resource_gate",
    "can_generate_video",
    "get_recommended_quality",
    "check_resources",
    # Lizard Brain
    "LizardBrain",
    "LizardState",
    "LizardConfig",
    "get_lizard_brain",
    "WakeEvent",
    "WakeEventType",
    "WakeCriteria",
    "WakeProtocol",
    "SalienceLevel",
    "PowerState",
    "PowerBudget",
    "ThermalZone",
    "PowerGovernor",
    "get_power_governor",
    "AttractorBasin",
    "BasinType",
    "BasinTransition",
    "AttractorMonitor",
    "get_attractor_monitor",
    # Brainlink
    "BrainlinkProtocol",
    "BrainlinkReading",
    "BrainlinkStatus",
    "SignalQuality",
    "ChannelData",
    "BrainlinkConfig",
    "BrainlinkError",
    "PhysioClient",
    "PhysioReading",
    "HeartRateData",
    "HRVData",
    "GSRData",
    "get_physio_client",
    "MuseClient",
    "MuseReading",
    "EEGBand",
    "EEGChannelData",
    "get_muse_client",
    "get_brainlink",
]
