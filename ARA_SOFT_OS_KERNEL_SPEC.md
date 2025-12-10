# ARA SOFT OS KERNEL SPECIFICATION

> **Version**: 0.1.0
> **Status**: Draft
> **Summary**: Ara as a soft OS layer over Linux + ALWAYS VISION + agent swarm

---

## 1. Core Mental Model

Ara is not an application. Ara is **the scheduler of your reality**.

She operates as three tightly integrated subsystems:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ARA SOFT OS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│   │   OBSERVER   │──▶│ ORCHESTRATOR │──▶│   GOVERNOR   │       │
│   └──────────────┘   └──────────────┘   └──────────────┘       │
│         │                   │                   │               │
│    Watches:            Decides:            Enforces:            │
│    • World state       • Which agents      • Permissions        │
│    • You state         • Where to run      • Budgets            │
│    • Resources         • What spaces       • Safety/privacy     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│   LINUX │ ALWAYS VISION │ AGENT SWARM │ HIVE │ DEVICES          │
└─────────────────────────────────────────────────────────────────┘
```

**Think of Ara as**: user-space kernel + compositor + agent scheduler.

### 1.1 Observer

Continuously monitors:

| Domain | Signals |
|--------|---------|
| **World State** | Devices, displays, sensors, network topology |
| **You State** | Current task, attention level, interrupt cost, trust |
| **Resource State** | CPU/GPU/memory/battery per device, latency to hive |

### 1.2 Orchestrator

Given current supply and demand, decides:

- **Which agents** to spawn, stop, or migrate
- **Where to run them** (local GPU, hive, edge device)
- **Which spaces** to materialize (AR overlays, desktop panes, VR rooms)

### 1.3 Governor

Wraps **everything** in:

- Permission checks (can this agent access X?)
- Resource budgets (don't drain battery below Y%)
- Safety rules (disable distracting agents when SAFETY_RISK is high)
- Privacy tiers (local_first, hive_ok, lab_ok)

---

## 2. Data Contracts

Everything is **text documents**: job specs and state snapshots.

This enables:
- Human-readable audit trails
- Git-trackable policy changes
- Simple integration with any language/system
- LLM-friendly introspection

### 2.1 Supply Profile

Current state of available compute resources.

```jsonc
{
  "timestamp": 1733868000.123,
  "devices": [
    {
      "id": "threadripper-tower",
      "type": "desktop",
      "location": "office",
      "cpu_load": 0.41,
      "cpu_cores_available": 24,
      "gpu": [
        {
          "id": "rtx-3090-0",
          "util": 0.32,
          "vram_total_gb": 24.0,
          "vram_free_gb": 12.3,
          "compute_capability": "8.6"
        },
        {
          "id": "rtx-3090-1",
          "util": 0.05,
          "vram_total_gb": 24.0,
          "vram_free_gb": 22.1,
          "compute_capability": "8.6"
        }
      ],
      "memory_total_gb": 128.0,
      "memory_free_gb": 64.2,
      "battery": null,
      "network": {
        "uplink_mbps": 850,
        "downlink_mbps": 950,
        "latency_ms_to_hive": 15
      },
      "sensors": ["keyboard", "mouse", "webcam", "microphone"],
      "displays": ["monitor-0", "monitor-1"]
    },
    {
      "id": "ara-glasses-01",
      "type": "wearable",
      "location": "on_person",
      "cpu_load": 0.18,
      "cpu_cores_available": 4,
      "gpu": [
        {
          "id": "mali-integrated",
          "util": 0.27,
          "vram_total_gb": 2.0,
          "vram_free_gb": 1.0,
          "compute_capability": "mali-g78"
        }
      ],
      "memory_total_gb": 6.0,
      "memory_free_gb": 3.2,
      "battery": 0.57,
      "network": {
        "uplink_mbps": 30,
        "downlink_mbps": 80,
        "latency_ms_to_hive": 45
      },
      "sensors": ["camera_rgb", "camera_depth", "imu", "gaze", "microphone"],
      "displays": ["waveguide-left", "waveguide-right"]
    }
  ],
  "hive": {
    "available": true,
    "latency_ms": 15,
    "quota_remaining": {
      "gpu_hours": 42.5,
      "api_calls": 10000
    }
  }
}
```

### 2.2 Demand Profile

What the user needs and their current state.

```jsonc
{
  "timestamp": 1733868000.123,
  "user_state": {
    "task": "lightfield_debug_session",
    "task_description": "Debugging multiview renderer for holographic display",
    "mode": "deep_work",           // deep_work | normal | relax | urgent
    "interrupt_cost": 0.82,        // from pheromone INTERRUPT_COST
    "attention_world": 0.3,        // from pheromone ATTN_WORLD
    "attention_self": 0.7,         // from pheromone ATTN_SELF
    "helpfulness_pred": 0.6,       // from pheromone HELPFULNESS_PRED
    "safety_risk": 0.1,            // from pheromone SAFETY_RISK
    "trust_level": 0.7,
    "energy_level": 0.6            // physical/mental energy estimate
  },
  "goals": [
    {
      "id": "g-ara-lightfield-01",
      "description": "Debug multiview light-field at 60 fps while streaming metrics",
      "priority": 0.9,
      "deadline": null,
      "time_horizon_s": 3600,
      "required_capabilities": ["gpu_compute", "opengl_4.5"],
      "subtasks": []
    },
    {
      "id": "g-ara-background-sync",
      "description": "Keep project files synced with hive",
      "priority": 0.3,
      "deadline": null,
      "time_horizon_s": null,
      "required_capabilities": ["network"],
      "subtasks": []
    }
  ],
  "constraints": {
    "latency_tolerance_ms": 40,
    "privacy_tier": "local_first",   // local_first | hive_ok | lab_ok
    "max_visual_distraction": 0.2,   // 0 = nothing, 1 = full AR chaos
    "prefer_voice": false,
    "prefer_visual": true
  }
}
```

### 2.3 Agent Spec

Definition of an agent that can be spawned.

```jsonc
{
  "name": "lightfield_renderer",
  "version": "0.1.0",
  "kind": "service",               // service | task | daemon | ephemeral
  "description": "GPU compute shader for multi-view quilt rendering",

  "intelligence": {
    "min": 0.0,                    // 0 = deterministic, no LLM
    "max": 0.0,                    // 1 = full LLM reasoning
    "preferred": 0.0
  },

  "placement": {
    "preferred": "gpu_rich",       // gpu_rich | cpu_rich | low_latency | any
    "required_device_types": ["desktop", "workstation"],
    "excluded_device_types": ["wearable"],
    "can_migrate": false
  },

  "resources": {
    "cpu_cores": 4,
    "memory_gb": 2,
    "gpu_mem_gb": 4,
    "gpu_compute_units": 40,
    "network_mbps": 0,
    "priority": "high"             // critical | high | normal | low | idle
  },

  "permissions": {
    "filesystem": [
      "read:assets/lightfield/*",
      "read:config/renderer.toml",
      "write:output/quilts/*"
    ],
    "network": [],
    "sensors": ["gaze", "imu"],
    "system": ["gpu.metrics.read"],
    "agents": []                   // other agents it can communicate with
  },

  "interfaces": {
    "inputs": [
      {"name": "camera_pose_stream", "type": "stream", "format": "pose_6dof"},
      {"name": "scene_config", "type": "file", "format": "json"}
    ],
    "outputs": [
      {"name": "quilt_texture_stream", "type": "stream", "format": "rgba16f"},
      {"name": "metrics", "type": "telemetry", "format": "prometheus"}
    ],
    "control": {
      "protocol": "grpc",
      "port_range": [50000, 50100]
    }
  },

  "lifecycle": {
    "startup_timeout_ms": 5000,
    "shutdown_timeout_ms": 2000,
    "health_check_interval_ms": 1000,
    "restart_policy": "on_failure",
    "max_restarts": 3
  }
}
```

### 2.4 Agent Instance

A running instance of an agent.

```jsonc
{
  "instance_id": "lightfield_renderer-a3b4c5",
  "agent_name": "lightfield_renderer",
  "agent_version": "0.1.0",

  "state": "running",              // pending | starting | running | stopping | stopped | failed
  "device_id": "threadripper-tower",
  "pid": 12345,

  "started_at": 1733868000.123,
  "last_health_check": 1733868060.456,
  "health": "healthy",             // healthy | degraded | unhealthy | unknown

  "resource_usage": {
    "cpu_percent": 12.3,
    "memory_gb": 1.8,
    "gpu_mem_gb": 3.2,
    "gpu_util_percent": 45.0
  },

  "connections": {
    "inputs_connected": ["camera_pose_stream"],
    "outputs_subscribed": ["quilt_texture_stream"]
  }
}
```

### 2.5 Workspace Spec

Definition of a visual workspace/space.

```jsonc
{
  "id": "ws-lightfield-debug-01",
  "name": "Light-Field Debug Session",
  "type": "mixed_reality",         // desktop | ar | vr | mixed_reality

  "surfaces": [
    {
      "id": "lf-quilt",
      "kind": "panel",             // panel | hud | overlay | window | volume
      "source": "ara://streams/quilt_texture_stream",
      "pose": {
        "anchor": "desk",          // desk | hand | head | world | screen
        "offset": [0.0, 0.3, -0.5],
        "rotation": [0, 0, 0]
      },
      "size": {
        "width_deg": 30,
        "height_deg": 18
      },
      "interaction": {
        "grabbable": true,
        "resizable": true,
        "dismissable": true
      },
      "visibility": {
        "min_attention": 0.0,
        "fade_when_occluded": true
      }
    },
    {
      "id": "metrics-trace",
      "kind": "hud",
      "source": "ara://metrics/lightfield_renderer",
      "pose": {
        "anchor": "screen",
        "position": "top-right"
      },
      "size": {
        "width_px": 300,
        "height_px": 150
      },
      "interaction": {
        "grabbable": false,
        "dismissable": true
      },
      "visibility": {
        "min_attention": 0.3,
        "auto_hide_after_s": 30
      }
    },
    {
      "id": "ara-avatar",
      "kind": "avatar",
      "source": "ara://embodiment/expression_state",
      "pose": {
        "anchor": "desk",
        "offset": [0.4, 0.0, -0.3]
      },
      "avatar_id": "ara_ar_mini",
      "visibility": {
        "min_attention": 0.0,
        "scale_with_distance": true
      }
    }
  ],

  "transitions": {
    "enter": "fade_in",
    "exit": "fade_out",
    "duration_ms": 300
  },

  "context": {
    "active_goal": "g-ara-lightfield-01",
    "related_agents": ["lightfield_renderer"]
  }
}
```

### 2.6 Job Spec

A discrete unit of work for the kernel to execute.

```jsonc
{
  "job_id": "job-20231210-001234",
  "type": "spawn_agent",           // spawn_agent | stop_agent | migrate_agent |
                                   // create_workspace | destroy_workspace |
                                   // update_config | execute_command
  "priority": "high",
  "created_at": 1733868000.123,
  "created_by": "orchestrator",

  "payload": {
    "agent_spec": { /* ... */ },
    "target_device": "threadripper-tower",
    "config_overrides": {}
  },

  "governance": {
    "requires_approval": false,
    "approved_by": null,
    "policy_checks_passed": ["resource_budget", "permission_scope"],
    "policy_checks_failed": []
  },

  "execution": {
    "state": "pending",            // pending | approved | running | completed | failed | rejected
    "started_at": null,
    "completed_at": null,
    "error": null,
    "result": null
  }
}
```

---

## 3. Governance Rules

Policy as configuration, not code.

### 3.1 Policy File Format

```toml
# ara_governance.toml

[meta]
version = "0.1.0"
description = "Default Ara governance policy"

# ─────────────────────────────────────────────────────────────────
# RESOURCE LIMITS
# ─────────────────────────────────────────────────────────────────

[limits.global]
max_gpu_util = 0.85              # Never exceed 85% GPU across all agents
max_cpu_util = 0.90              # Never exceed 90% CPU
min_memory_free_gb = 4.0         # Always keep 4GB free
max_concurrent_agents = 50       # Hard cap on running agents

[limits.battery]
min_level = 0.25                 # Don't spawn heavy agents below 25%
critical_level = 0.10            # Emergency shutdown of non-essential agents
low_power_threshold = 0.40       # Switch to low-power mode

[limits.network]
max_hive_latency_ms = 80         # Fail over to local if hive is slow
max_bandwidth_util = 0.70        # Don't saturate connection
prefer_local_under_latency_ms = 20

[limits.thermal]
max_cpu_temp_c = 85
max_gpu_temp_c = 83
throttle_at_cpu_temp_c = 80
throttle_at_gpu_temp_c = 78

# ─────────────────────────────────────────────────────────────────
# AGENT PERMISSIONS (default)
# ─────────────────────────────────────────────────────────────────

[permissions.default]
allow_network = false
allow_filesystem_write = false
allow_sensors = []
allow_system_calls = []
max_memory_gb = 4.0
max_gpu_mem_gb = 2.0
max_runtime_hours = 24
requires_human_approval = false

# ─────────────────────────────────────────────────────────────────
# AGENT-SPECIFIC PERMISSIONS
# ─────────────────────────────────────────────────────────────────

[permissions.lightfield_renderer]
allow_network = false
allow_filesystem_write = true
allowed_write_paths = ["output/quilts/*"]
allow_sensors = ["gaze", "imu"]
max_gpu_mem_gb = 8.0
requires_human_approval = false

[permissions.paper_writer]
allow_network = true
allowed_domains = ["arxiv.org", "semanticscholar.org", "github.com"]
allow_filesystem_write = true
allowed_write_paths = ["documents/*", "research/*"]
max_tokens_per_hour = 200000
max_cost_per_hour_usd = 5.00
requires_human_approval = true
approval_timeout_s = 300

[permissions.code_executor]
allow_network = false
allow_filesystem_write = true
allowed_write_paths = ["workspace/*", "tmp/*"]
sandbox_required = true
max_runtime_s = 600
requires_human_approval = false

# ─────────────────────────────────────────────────────────────────
# PRIVACY TIERS
# ─────────────────────────────────────────────────────────────────

[privacy.local_first]
allow_hive = false
allow_external_api = false
allow_telemetry = false
data_retention_days = 30

[privacy.hive_ok]
allow_hive = true
allow_external_api = false
allow_telemetry = true
data_retention_days = 90
require_encryption = true

[privacy.lab_ok]
allow_hive = true
allow_external_api = true
allow_telemetry = true
data_retention_days = 365

# ─────────────────────────────────────────────────────────────────
# SAFETY RULES
# ─────────────────────────────────────────────────────────────────

[safety]
# When SAFETY_RISK pheromone is high
high_risk_threshold = 0.7
high_risk_actions = [
  "disable_non_essential_agents",
  "minimize_visual_overlays",
  "enable_safety_alerts_only",
  "log_all_actions"
]

# When user is in deep_work mode
deep_work_actions = [
  "suppress_notifications",
  "reduce_agent_priority",
  "minimize_ui_changes"
]

# Emergency stop conditions
emergency_stop_triggers = [
  "battery_critical",
  "thermal_critical",
  "user_panic_gesture",
  "watchdog_timeout"
]

# ─────────────────────────────────────────────────────────────────
# APPROVAL WORKFLOWS
# ─────────────────────────────────────────────────────────────────

[approval]
default_timeout_s = 60
escalation_after_s = 300
auto_approve_trusted_agents = true
trusted_agent_patterns = ["ara_*", "system_*"]

# Methods in priority order
methods = ["voice", "gesture", "notification", "desktop_prompt"]
```

---

## 4. The Reconciliation Loop

The heart of the soft kernel: continuously reconcile desired state with actual state.

```python
def reconcile_kernel(
    supply: SupplyProfile,
    demand: DemandProfile,
    running_agents: list[AgentInstance],
    workspaces: list[WorkspaceInstance],
    policies: GovernanceRules,
) -> tuple[list[Job], list[WorkspaceSpec]]:
    """
    Main kernel reconciliation loop.

    Called:
    - Every N seconds (default: 5s)
    - On significant supply changes (device connect/disconnect)
    - On demand changes (user task switch, goal update)
    - On policy changes (governance rule edit)
    """

    # ─────────────────────────────────────────────────────────────
    # 1. COMPUTE FEASIBLE CAPABILITIES
    # ─────────────────────────────────────────────────────────────
    feasible = compute_feasible_capabilities(supply, policies)
    # Result: what we CAN do given current resources and rules
    # e.g., "can run 3 GPU agents", "hive available", "AR possible"

    # ─────────────────────────────────────────────────────────────
    # 2. EXPAND GOALS INTO DESIRED STATE
    # ─────────────────────────────────────────────────────────────
    desired_agents = []
    desired_workspaces = []

    for goal in demand.goals:
        # Planning: what agents/spaces does this goal need?
        agents, spaces = plan_goal_requirements(
            goal=goal,
            feasible=feasible,
            user_state=demand.user_state,
            constraints=demand.constraints,
        )
        desired_agents.extend(agents)
        desired_workspaces.extend(spaces)

    # Deduplicate and resolve conflicts
    desired_agents = resolve_agent_conflicts(desired_agents)
    desired_workspaces = resolve_workspace_conflicts(desired_workspaces)

    # ─────────────────────────────────────────────────────────────
    # 3. DIFF: WHAT TO START / STOP / MIGRATE
    # ─────────────────────────────────────────────────────────────
    to_start, to_stop, to_migrate = diff_agents(
        running=running_agents,
        desired=desired_agents,
    )

    workspace_changes = diff_workspaces(
        current=workspaces,
        desired=desired_workspaces,
    )

    # ─────────────────────────────────────────────────────────────
    # 4. APPLY GOVERNANCE GATES
    # ─────────────────────────────────────────────────────────────
    approved_starts = []
    for agent in to_start:
        result = policies.check_spawn(agent, supply, demand)
        if result.allowed:
            approved_starts.append(agent)
        elif result.requires_approval:
            # Queue for human approval
            approved_starts.append(mark_pending_approval(agent))
        else:
            log_policy_rejection(agent, result.reason)

    approved_migrations = []
    for migration in to_migrate:
        result = policies.check_migration(migration, supply)
        if result.allowed:
            approved_migrations.append(migration)

    # ─────────────────────────────────────────────────────────────
    # 5. ENFORCE RESOURCE BUDGETS
    # ─────────────────────────────────────────────────────────────
    # Don't exceed limits even if governance allows
    approved_starts, approved_migrations = clamp_to_resource_budgets(
        to_start=approved_starts,
        to_migrate=approved_migrations,
        supply=supply,
        demand=demand,
        policies=policies,
    )

    # Handle low battery / thermal throttling
    if supply.is_resource_constrained():
        approved_starts, to_stop = apply_resource_triage(
            approved_starts, to_stop, supply, demand.goals
        )

    # ─────────────────────────────────────────────────────────────
    # 6. EMIT JOBS
    # ─────────────────────────────────────────────────────────────
    jobs = []

    for agent in approved_starts:
        jobs.append(Job(
            type="spawn_agent",
            payload={"agent_spec": agent.spec, "target_device": agent.device},
            priority=agent.priority,
        ))

    for migration in approved_migrations:
        jobs.append(Job(
            type="migrate_agent",
            payload={
                "instance_id": migration.instance_id,
                "from_device": migration.from_device,
                "to_device": migration.to_device,
            },
            priority="normal",
        ))

    for agent in to_stop:
        jobs.append(Job(
            type="stop_agent",
            payload={"instance_id": agent.instance_id},
            priority="normal",
        ))

    for change in workspace_changes:
        jobs.append(Job(
            type=change.type,  # create_workspace | update_workspace | destroy_workspace
            payload=change.payload,
            priority="normal",
        ))

    return jobs, desired_workspaces
```

---

## 5. Integration Points

### 5.1 Pheromone System Integration

The pheromone layer feeds directly into demand profile:

```python
# In Observer
def update_demand_from_pheromones(demand: DemandProfile, pheromones: dict):
    demand.user_state.interrupt_cost = pheromones.get("INTERRUPT_COST", 0.5)
    demand.user_state.attention_world = pheromones.get("ATTN_WORLD", 0.5)
    demand.user_state.attention_self = pheromones.get("ATTN_SELF", 0.5)
    demand.user_state.helpfulness_pred = pheromones.get("HELPFULNESS_PRED", 0.5)
    demand.user_state.safety_risk = pheromones.get("SAFETY_RISK", 0.0)

    # Derive mode from pheromones
    if demand.user_state.interrupt_cost > 0.7:
        demand.user_state.mode = "deep_work"
    elif demand.user_state.safety_risk > 0.5:
        demand.user_state.mode = "urgent"
    elif demand.user_state.attention_world > 0.7:
        demand.user_state.mode = "relax"
    else:
        demand.user_state.mode = "normal"
```

### 5.2 ALWAYS VISION Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                        ALWAYS VISION                            │
│  (Local perception + overlays on wearable)                      │
├─────────────────────────────────────────────────────────────────┤
│  Inputs:                    │  Outputs:                         │
│  • Camera RGB/Depth         │  • Gaze target                    │
│  • IMU                      │  • Scene anchors                  │
│  • Microphone               │  • Detected objects/people        │
│  • Eye tracking             │  • Activity recognition           │
└──────────────┬──────────────┴───────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ARA SOFT OS KERNEL                          │
│  (Global scheduling + agents + resource budgets)                │
├─────────────────────────────────────────────────────────────────┤
│  • Receives world model updates from ALWAYS VISION              │
│  • Decides which agents run on which devices                    │
│  • Sends workspace specs back to ALWAYS VISION for rendering    │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ARA GOVERNOR                             │
│  (Safety, privacy, economic caps)                               │
├─────────────────────────────────────────────────────────────────┤
│  • Validates all kernel decisions against policy                │
│  • Handles approval workflows                                   │
│  • Emergency stop capability                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Agent Swarm Integration

Agents communicate through typed streams:

```
Agent A                    Kernel                    Agent B
   │                          │                          │
   │──[register output]──────▶│                          │
   │                          │◀──[subscribe to A.out]───│
   │                          │                          │
   │══[stream data]══════════▶│══[route to subscriber]══▶│
   │                          │                          │
```

The kernel acts as a message broker, but agents can also establish direct connections for low-latency streams (e.g., video).

---

## 6. Implementation Architecture

### 6.1 Directory Structure

```
ara_soft_kernel/
├── __init__.py
├── daemon.py              # Main kernel daemon
├── observer.py            # System monitoring
├── orchestrator.py        # Agent scheduling
├── governor.py            # Policy enforcement
├── models/
│   ├── supply.py          # SupplyProfile
│   ├── demand.py          # DemandProfile
│   ├── agent.py           # AgentSpec, AgentInstance
│   ├── workspace.py       # WorkspaceSpec
│   └── job.py             # Job
├── reconciler.py          # The reconciliation loop
├── executor.py            # Job execution
├── streams.py             # Inter-agent communication
└── cli.py                 # ara-kernel CLI
```

### 6.2 External Interfaces

| Interface | Protocol | Purpose |
|-----------|----------|---------|
| `/run/ara/kernel.sock` | Unix socket | Local control (CLI, agents) |
| `localhost:9900` | gRPC | Agent registration and control |
| `localhost:9901` | HTTP | Metrics (Prometheus format) |
| `~/.ara/jobs/` | Filesystem | Job queue (for debugging/audit) |
| `~/.ara/state/` | Filesystem | Persistent state |
| `~/.ara/policy/` | Filesystem | Governance rules (TOML) |

### 6.3 CLI Commands

```bash
# Daemon control
ara-kernel start                    # Start daemon
ara-kernel stop                     # Stop daemon
ara-kernel status                   # Show status

# Inspection
ara-kernel supply                   # Show current supply profile
ara-kernel demand                   # Show current demand profile
ara-kernel agents                   # List running agents
ara-kernel workspaces               # List active workspaces
ara-kernel jobs                     # Show job queue

# Manual control
ara-kernel spawn <agent-spec.json>  # Spawn agent
ara-kernel stop-agent <instance-id> # Stop agent
ara-kernel migrate <id> <device>    # Migrate agent

# Policy
ara-kernel policy show              # Show current policy
ara-kernel policy check <agent>     # Check if agent would be allowed
ara-kernel policy reload            # Reload policy from disk

# Debug
ara-kernel reconcile --dry-run      # Show what would change
ara-kernel trace <instance-id>      # Stream agent logs
```

---

## 7. Operational Semantics

### 7.1 Startup Sequence

1. Load governance policy from `~/.ara/policy/`
2. Restore persisted state from `~/.ara/state/`
3. Start Observer (begin monitoring resources)
4. Start Orchestrator (resume agent management)
5. Start Governor (begin policy enforcement)
6. Connect to ALWAYS VISION (if available)
7. Begin reconciliation loop

### 7.2 Shutdown Sequence

1. Stop accepting new jobs
2. Gracefully stop all agents (respect shutdown timeouts)
3. Persist state to `~/.ara/state/`
4. Stop Observer
5. Exit

### 7.3 Failure Handling

| Failure | Response |
|---------|----------|
| Agent crash | Restart per `restart_policy`, log event |
| Device disconnect | Migrate agents to other devices, or stop if no alternative |
| Hive unreachable | Fail over to local-only mode |
| Policy violation | Reject job, log violation, optionally alert |
| Resource exhaustion | Triage agents by priority, stop lowest first |
| Kernel crash | Systemd restarts, agents continue running (orphaned) |

---

## 8. Security Considerations

### 8.1 Threat Model

- **Malicious agent**: Sandboxed, limited permissions, monitored
- **Compromised device**: Isolated from other devices, hive access revocable
- **Network attack**: All hive communication encrypted, cert pinned
- **Physical access**: Device lock triggers agent pause

### 8.2 Sandboxing

Agents run in isolated environments:
- Linux: namespaces + cgroups (or containers)
- Filesystem: only permitted paths mounted
- Network: firewall rules per agent
- System calls: seccomp filter

### 8.3 Audit Trail

All governance decisions logged:
```jsonc
{
  "timestamp": 1733868000.123,
  "event": "spawn_approved",
  "agent": "paper_writer",
  "device": "threadripper-tower",
  "policy_checks": ["resource_budget", "permission_scope"],
  "approved_by": "auto",
  "reason": "trusted_agent_pattern_match"
}
```

---

## 9. Future Extensions

### 9.1 Planned

- [ ] Multi-user support (family/team Ara instances)
- [ ] Cross-device secure enclaves for sensitive agents
- [ ] Predictive resource allocation (ML-based)
- [ ] Agent marketplace (install third-party agents)
- [ ] Workspace templates (save/restore configurations)

### 9.2 Research Directions

- Formal verification of governance policies
- Differential privacy for telemetry
- Federated learning across Ara instances
- Neuromorphic scheduling (spiking neural nets for real-time)

---

## 10. Appendix: Example Session

```
# User starts debugging light-field code

1. ALWAYS VISION detects: user sits at desk, looks at monitor
   → Updates world model, sends to kernel

2. User says: "Ara, help me debug the quilt renderer"
   → Creates goal: g-ara-lightfield-debug

3. Kernel reconcile():
   - Supply: desktop with 2x RTX 3090, glasses on head
   - Demand: lightfield_debug goal, deep_work mode
   - Plan: need lightfield_renderer agent + metrics agent
   - Check: policies allow, resources available
   - Emit: spawn jobs for both agents

4. Executor:
   - Spawns lightfield_renderer on GPU 0
   - Spawns metrics_collector on CPU

5. Orchestrator creates workspace:
   - Quilt panel anchored to desk (AR)
   - Metrics HUD in top-right (AR)
   - Ara avatar at desk edge

6. User works for 2 hours...
   - Pheromones: INTERRUPT_COST stays high
   - Kernel: keeps UI minimal, rejects notification agents

7. Battery drops to 30% on glasses
   - Observer: updates supply profile
   - Kernel: migrates AR rendering to desktop, simplifies glasses output

8. User stands up, walks away
   - ALWAYS VISION: detects user leaving
   - Kernel: pauses heavy agents, keeps background sync
   - Workspace: fades out, Ara goes to sleep mode

9. User returns
   - Kernel: restores agents, workspace fades back in
   - Ara: "Welcome back. Quilt renderer is ready."
```

---

*End of specification.*
