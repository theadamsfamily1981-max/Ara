# Ara Cluster Configuration

Three-node topology for Ara v0.7+

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ARA CLUSTER TOPOLOGY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   🏛️ CATHEDRAL (ara-cathedral)                                              │
│   ├── Role: Brainstem + Orchestrator                                        │
│   ├── CPU: Threadripper Pro 5955WX (16c/32t)                               │
│   ├── RAM: 128 GB DDR4                                                      │
│   ├── GPU: 2× RTX 3090 24GB                                                │
│   ├── FPGA: BittWare A10PED (Phase 2)                                      │
│   ├── Storage: Micron SB852 + 8× NVMe RAID0                                │
│   └── Services: ara_realtime, ara_storage, ara_orchestrator                │
│                                                                             │
│   💻 HOME (ara-home)                                                        │
│   ├── Role: Daily Ara + Kitten Guardian                                     │
│   ├── GPU: RTX 5060 16GB                                                    │
│   ├── Security: SQRL Forest Kitten (Phase 2)                               │
│   ├── Services: ara_frontend, ara_companion                                │
│   └── Can run offline                                                       │
│                                                                             │
│   🖥️ WORKER (ara-worker-v100)                                               │
│   ├── Role: Training Mule                                                   │
│   ├── GPU: V100 16GB                                                        │
│   ├── Services: ara_trainer                                                │
│   └── Mounts cathedral:/data/ara                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### v0.7 Minimum (Cathedral Only)

```bash
# On cathedral
cd /path/to/ara
./cluster/start_cathedral.sh
```

### Full Cluster

```bash
# On cathedral
./cluster/start_cathedral.sh

# On home (separate machine)
./cluster/start_home.sh --connect cathedral.lan:7777

# On worker (separate machine)
./cluster/start_worker.sh --connect cathedral.lan:7777
```

## Who Does What

| Role | Primary Node | Fallback | GPU Required |
|------|--------------|----------|--------------|
| Real-time nervous system | Cathedral | Home | No |
| Model training | Worker | Cathedral | Yes (16GB+) |
| Daily inference | Home | Cathedral | Yes (8GB+) |
| Covenant signing | Home (kitten) | Cathedral | No |
| Dataset serving | Cathedral | Worker | No |

## Phase 2 Hardware

Not required for v0.7, but interfaces are ready:

| Hardware | Node | Role | Interface |
|----------|------|------|-----------|
| BittWare A10PED | Cathedral | Audio front-end | `AudioFrontEnd` trait |
| SQRL Forest Kitten | Home | Covenant guardian | `CovenantGuard` trait |
| Micron SB852 | Cathedral | Dataset cache | `StorageBackend` trait |

## Configuration

Edit `cluster/cluster.toml` to customize:

- Node hostnames and addresses
- GPU assignments
- Service distribution
- Interface implementations

## v0.7 Constraints

Ara v0.7 can run on **cathedral alone** with:
- Just the CPU (nervous system is Rust, runs fine on CPU)
- At least 1 GPU for training (or skip training)
- Standard NVMe storage

The full cluster is for **production deployment**, not required for development.
