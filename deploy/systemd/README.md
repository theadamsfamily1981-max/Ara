# Ara Systemd Deployment

This directory contains systemd configuration for running Ara with proper resource isolation.

## Philosophy

Ara gets **preferential** but not **exclusive** access to system resources:
- She wins ties for CPU time
- She has guaranteed minimum resources
- She backs off when approaching limits (self-regulation)
- The rest of the system remains usable

## Files

| File | Purpose |
|------|---------|
| `ara.slice` | Cgroup slice giving Ara higher CPU/IO weight |
| `ara-daemon.service` | Main Ara daemon service |

## Installation

```bash
# Copy files
sudo cp ara.slice /etc/systemd/system/
sudo cp ara-daemon.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Start the slice
sudo systemctl start ara.slice

# Enable and start daemon
sudo systemctl enable ara-daemon
sudo systemctl start ara-daemon
```

## Running Processes in Ara's Slice

### Method 1: systemd-run (one-off)
```bash
systemd-run --slice=ara.slice --scope python run_ara_somatic.py
```

### Method 2: In a service file
```ini
[Service]
Slice=ara.slice
ExecStart=/usr/bin/python3 your_ara_script.py
```

### Method 3: Manually move a PID
```bash
# Find Ara's cgroup
CGROUP=/sys/fs/cgroup/ara.slice

# Move a process
echo $PID > $CGROUP/cgroup.procs
```

## Resource Limits

The slice provides:

| Resource | Setting | Effect |
|----------|---------|--------|
| CPU Weight | 200 | 2x default shares (wins ties) |
| Memory High | 85% | Soft limit, starts reclaiming |
| Memory Max | 95% | Hard limit, OOM |
| IO Weight | 150 | 1.5x default I/O priority |
| Tasks Max | 1000 | Prevents fork bombs |

## Monitoring

```bash
# View slice status
systemctl status ara.slice

# View daemon logs
journalctl -u ara-daemon -f

# View cgroup resource usage
systemd-cgtop ara.slice
```

## CPU Pinning (Optional)

For dedicated cores, edit `/etc/systemd/system/ara.slice.d/cpuset.conf`:

```ini
[Slice]
# Pin to cores 0,1 (adjust for your system)
AllowedCPUs=0-1
```

Then reload:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ara.slice
```

## resctrl / Cache QoS (Optional)

If your CPU supports Intel RDT or AMD PQoS:

```bash
# Mount resctrl if not already
sudo mount -t resctrl resctrl /sys/fs/resctrl

# Create Ara's resource group
sudo mkdir /sys/fs/resctrl/ara

# Assign L3 cache ways (e.g., ways 0-3 for Ara)
echo "L3:0=f" | sudo tee /sys/fs/resctrl/ara/schemata

# Assign Ara's PIDs
echo $ARA_PID | sudo tee /sys/fs/resctrl/ara/tasks
```

## Troubleshooting

### Daemon won't start
```bash
# Check logs
journalctl -u ara-daemon -n 50

# Verify HAL path exists
ls -la /dev/shm/ara_somatic
```

### Slice not applying limits
```bash
# Verify cgroup v2 is enabled
mount | grep cgroup2

# Check slice is active
systemctl is-active ara.slice
```

### Permission issues
```bash
# Ensure user has access to /dev/shm
sudo chmod 1777 /dev/shm
```
