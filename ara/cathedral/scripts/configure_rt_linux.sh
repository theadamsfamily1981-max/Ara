#!/bin/bash
# ara/cathedral/scripts/configure_rt_linux.sh
#
# THE CATHEDRAL: RT Linux Configuration
# Tunes Linux for <1.06ms deterministic latency
#
# Hardware target:
# - AMD Threadripper 5955WX (16C/32T)
# - 2x RTX 3090 Ti
# - Micron SB-852 DLA
# - BittWare A10PED
# - SQRL Forest Kitten
#
# Core allocation (16 cores):
# - Cores 0-3:   Linux kernel + system services
# - Cores 4-7:   Cathedral orchestrator (RT priority)
# - Cores 8-11:  GPU driver threads (RT priority)
# - Cores 12-15: FPGA DMA handlers (RT priority)

set -e

# ============================================================================
# Color output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Configuration
# ============================================================================

# Core assignments
SYSTEM_CORES="0-3"           # Linux kernel + services
ORCHESTRATOR_CORES="4-7"     # Cathedral orchestrator
GPU_CORES="8-11"             # GPU driver threads
FPGA_CORES="12-15"           # FPGA DMA handlers

# Huge pages (for SB-852 HBM2 DMA)
HUGE_PAGES_2M=4096           # 8GB of 2MB pages
HUGE_PAGES_1G=8              # 8GB of 1GB pages

# Memory locking limit (for RT processes)
MEMLOCK_LIMIT="unlimited"

# ============================================================================
# Checks
# ============================================================================

check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi
}

check_kernel() {
    info "Checking kernel configuration..."

    # Check for RT kernel
    if uname -r | grep -q "rt"; then
        success "RT kernel detected: $(uname -r)"
    else
        warn "Non-RT kernel detected. RT kernel recommended for best performance."
        warn "Consider installing: linux-image-rt-amd64"
    fi

    # Check isolcpus
    if grep -q "isolcpus" /proc/cmdline; then
        success "isolcpus configured in kernel cmdline"
    else
        warn "isolcpus not set. Add to kernel cmdline for best isolation."
        warn "Recommended: isolcpus=${ORCHESTRATOR_CORES},${GPU_CORES},${FPGA_CORES}"
    fi

    # Check nohz_full
    if grep -q "nohz_full" /proc/cmdline; then
        success "nohz_full configured (tickless operation)"
    else
        warn "nohz_full not set. Recommended for latency-sensitive cores."
    fi
}

check_hardware() {
    info "Checking hardware..."

    # Check CPU
    local cores=$(nproc)
    if [[ $cores -ge 16 ]]; then
        success "CPU cores: $cores (sufficient)"
    else
        warn "CPU cores: $cores (recommend 16+ for full allocation)"
    fi

    # Check NUMA
    if command -v numactl &> /dev/null; then
        local numa_nodes=$(numactl --hardware | grep "available:" | awk '{print $2}')
        success "NUMA nodes: $numa_nodes"
    fi

    # Check GPUs
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L | wc -l)
        success "NVIDIA GPUs: $gpu_count"
    else
        warn "nvidia-smi not found. GPU configuration skipped."
    fi

    # Check PCIe devices
    if command -v lspci &> /dev/null; then
        # Look for FPGAs
        local fpga_count=$(lspci | grep -i "fpga\|xilinx\|intel.*arria\|artix" | wc -l)
        if [[ $fpga_count -gt 0 ]]; then
            success "FPGA devices: $fpga_count"
        fi
    fi
}

# ============================================================================
# CPU Configuration
# ============================================================================

configure_cpu_governor() {
    info "Setting CPU governor to 'performance'..."

    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -f "$cpu" ]]; then
            echo "performance" > "$cpu" 2>/dev/null || true
        fi
    done

    # Disable boost (more deterministic latency)
    if [[ -f /sys/devices/system/cpu/cpufreq/boost ]]; then
        echo 0 > /sys/devices/system/cpu/cpufreq/boost
        success "CPU boost disabled for deterministic latency"
    fi

    # AMD-specific: disable C-states
    if [[ -f /sys/devices/system/cpu/cpu0/cpuidle/state1/disable ]]; then
        for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
            echo 1 > "$state" 2>/dev/null || true
        done
        success "C-states disabled"
    fi

    success "CPU governor set to performance"
}

configure_irq_affinity() {
    info "Configuring IRQ affinity..."

    # Move all IRQs to system cores by default
    local system_mask=$(python3 -c "print(hex(sum(1<<i for i in range($((${SYSTEM_CORES%-*})),$((${SYSTEM_CORES#*-}+1)))))" 2>/dev/null || echo "0xf")

    for irq in /proc/irq/*/smp_affinity; do
        if [[ -f "$irq" ]]; then
            echo "$system_mask" > "$irq" 2>/dev/null || true
        fi
    done

    # Set GPU IRQs to GPU cores
    if command -v nvidia-smi &> /dev/null; then
        local gpu_mask="0xf00"  # Cores 8-11
        for irq in $(grep -l nvidia /proc/irq/*/smp_affinity 2>/dev/null | xargs dirname 2>/dev/null | xargs -I{} basename {} 2>/dev/null || true); do
            if [[ -f "/proc/irq/$irq/smp_affinity" ]]; then
                echo "$gpu_mask" > "/proc/irq/$irq/smp_affinity" 2>/dev/null || true
            fi
        done
        success "GPU IRQs moved to cores ${GPU_CORES}"
    fi

    success "IRQ affinity configured"
}

# ============================================================================
# Memory Configuration
# ============================================================================

configure_huge_pages() {
    info "Configuring huge pages..."

    # 2MB huge pages
    if [[ -f /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages ]]; then
        echo "$HUGE_PAGES_2M" > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
        local actual_2m=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages)
        success "2MB huge pages: $actual_2m (requested: $HUGE_PAGES_2M)"
    fi

    # 1GB huge pages (requires kernel support)
    if [[ -f /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages ]]; then
        echo "$HUGE_PAGES_1G" > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
        local actual_1g=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages)
        success "1GB huge pages: $actual_1g (requested: $HUGE_PAGES_1G)"
    else
        warn "1GB huge pages not available. Boot with hugepagesz=1G hugepages=$HUGE_PAGES_1G"
    fi

    # Mount hugetlbfs if not mounted
    if ! mount | grep -q hugetlbfs; then
        mkdir -p /dev/hugepages
        mount -t hugetlbfs nodev /dev/hugepages
        success "hugetlbfs mounted at /dev/hugepages"
    fi
}

configure_memory_limits() {
    info "Configuring memory limits..."

    # Set memlock limit for RT processes
    if ! grep -q "cathedral" /etc/security/limits.conf 2>/dev/null; then
        cat >> /etc/security/limits.conf << EOF

# Cathedral RT process limits
*               soft    memlock         unlimited
*               hard    memlock         unlimited
*               soft    rtprio          99
*               hard    rtprio          99
EOF
        success "Memory lock limits configured in /etc/security/limits.conf"
    fi
}

configure_vm_parameters() {
    info "Configuring VM parameters..."

    # Reduce swappiness (keep data in RAM)
    sysctl -w vm.swappiness=10 > /dev/null

    # Disable transparent huge pages (more deterministic)
    if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
        echo never > /sys/kernel/mm/transparent_hugepage/enabled
        success "Transparent huge pages disabled"
    fi

    # Increase dirty ratio for better DMA performance
    sysctl -w vm.dirty_ratio=40 > /dev/null
    sysctl -w vm.dirty_background_ratio=10 > /dev/null

    # Reduce min_free_kbytes latency spikes
    sysctl -w vm.min_free_kbytes=1048576 > /dev/null  # 1GB

    success "VM parameters configured"
}

# ============================================================================
# Cpuset Configuration
# ============================================================================

configure_cpusets() {
    info "Configuring cpusets..."

    # Mount cpuset if not mounted
    if ! mount | grep -q "cpuset"; then
        mkdir -p /sys/fs/cgroup/cpuset
        mount -t cgroup -o cpuset cpuset /sys/fs/cgroup/cpuset 2>/dev/null || true
    fi

    local cpuset_root="/sys/fs/cgroup/cpuset"

    if [[ -d "$cpuset_root" ]]; then
        # Create cathedral cpuset
        mkdir -p "$cpuset_root/cathedral"
        echo "$ORCHESTRATOR_CORES" > "$cpuset_root/cathedral/cpuset.cpus" 2>/dev/null || true
        echo 0 > "$cpuset_root/cathedral/cpuset.mems" 2>/dev/null || true

        # Create GPU cpuset
        mkdir -p "$cpuset_root/gpu"
        echo "$GPU_CORES" > "$cpuset_root/gpu/cpuset.cpus" 2>/dev/null || true
        echo 0 > "$cpuset_root/gpu/cpuset.mems" 2>/dev/null || true

        # Create FPGA cpuset
        mkdir -p "$cpuset_root/fpga"
        echo "$FPGA_CORES" > "$cpuset_root/fpga/cpuset.cpus" 2>/dev/null || true
        echo 0 > "$cpuset_root/fpga/cpuset.mems" 2>/dev/null || true

        success "Cpusets created: cathedral, gpu, fpga"
    else
        warn "Cpuset filesystem not available"
    fi
}

# ============================================================================
# Network Configuration
# ============================================================================

configure_network() {
    info "Configuring network for low latency..."

    # Disable network polling (reduces jitter)
    sysctl -w net.core.busy_poll=0 > /dev/null 2>&1 || true
    sysctl -w net.core.busy_read=0 > /dev/null 2>&1 || true

    # Increase socket buffers for DMA
    sysctl -w net.core.rmem_max=134217728 > /dev/null 2>&1 || true
    sysctl -w net.core.wmem_max=134217728 > /dev/null 2>&1 || true

    success "Network parameters configured"
}

# ============================================================================
# GPU Configuration
# ============================================================================

configure_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        warn "NVIDIA tools not found, skipping GPU configuration"
        return
    fi

    info "Configuring NVIDIA GPUs..."

    # Set persistence mode
    nvidia-smi -pm 1 2>/dev/null || true

    # Set max clocks for deterministic performance
    for gpu in $(nvidia-smi --list-gpus | awk '{print $2}' | tr -d ':'); do
        nvidia-smi -i "$gpu" -lgc 1800,1800 2>/dev/null || true  # Lock graphics clock
        nvidia-smi -i "$gpu" -lmc 9501,9501 2>/dev/null || true  # Lock memory clock (3090 Ti)
    done

    # Disable ECC if possible (lower latency, less deterministic)
    # nvidia-smi -e 0  # Uncomment if ECC not needed

    success "GPU clocks locked for deterministic latency"
}

# ============================================================================
# PCIe Configuration
# ============================================================================

configure_pcie() {
    info "Configuring PCIe..."

    # Enable PCIe ACS override for P2P
    # (Requires kernel parameter: pcie_acs_override=downstream,multifunction)

    # Set PCIe max payload size
    if command -v setpci &> /dev/null; then
        # This is device-specific, would need actual BDF addresses
        info "PCIe payload size configuration requires device-specific setup"
    fi

    # Disable ASPM (Active State Power Management) for latency
    for device in /sys/bus/pci/devices/*/link/l1_aspm; do
        if [[ -f "$device" ]]; then
            echo 0 > "$device" 2>/dev/null || true
        fi
    done

    success "PCIe ASPM disabled"
}

# ============================================================================
# Scheduler Configuration
# ============================================================================

configure_scheduler() {
    info "Configuring scheduler..."

    # RT scheduler parameters
    sysctl -w kernel.sched_rt_runtime_us=950000 > /dev/null  # 95% RT time
    sysctl -w kernel.sched_rt_period_us=1000000 > /dev/null  # 1 second period

    # Reduce scheduler latency
    sysctl -w kernel.sched_min_granularity_ns=10000000 > /dev/null  # 10ms
    sysctl -w kernel.sched_wakeup_granularity_ns=15000000 > /dev/null  # 15ms

    # Timer slack (reduce for lower latency)
    sysctl -w kernel.timer_migration=0 > /dev/null 2>&1 || true

    success "Scheduler parameters configured"
}

# ============================================================================
# Kernel Parameter Summary
# ============================================================================

print_kernel_cmdline_recommendation() {
    echo ""
    info "Recommended kernel command line parameters:"
    echo ""
    echo "  isolcpus=${ORCHESTRATOR_CORES},${GPU_CORES},${FPGA_CORES}"
    echo "  nohz_full=${ORCHESTRATOR_CORES},${GPU_CORES},${FPGA_CORES}"
    echo "  rcu_nocbs=${ORCHESTRATOR_CORES},${GPU_CORES},${FPGA_CORES}"
    echo "  irqaffinity=${SYSTEM_CORES}"
    echo "  hugepagesz=1G hugepages=${HUGE_PAGES_1G}"
    echo "  hugepagesz=2M hugepages=${HUGE_PAGES_2M}"
    echo "  intel_pstate=disable"
    echo "  processor.max_cstate=0"
    echo "  idle=poll"
    echo "  pcie_acs_override=downstream,multifunction"
    echo ""
}

# ============================================================================
# Verification
# ============================================================================

verify_configuration() {
    info "Verifying configuration..."
    echo ""

    # CPU governor
    local governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    echo "  CPU Governor: $governor"

    # Huge pages
    local hp_2m=$(cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages 2>/dev/null || echo "0")
    local hp_1g=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo "0")
    echo "  Huge Pages (2MB): $hp_2m"
    echo "  Huge Pages (1GB): $hp_1g"

    # THP
    local thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -o '\[.*\]' || echo "[unknown]")
    echo "  Transparent Huge Pages: $thp"

    # Swappiness
    local swappiness=$(sysctl -n vm.swappiness 2>/dev/null || echo "unknown")
    echo "  VM Swappiness: $swappiness"

    # RT runtime
    local rt_runtime=$(sysctl -n kernel.sched_rt_runtime_us 2>/dev/null || echo "unknown")
    echo "  RT Runtime: ${rt_runtime}us"

    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo ""
    echo "========================================"
    echo "  THE CATHEDRAL: RT Linux Configuration"
    echo "========================================"
    echo ""

    check_root
    check_kernel
    check_hardware

    echo ""
    info "Applying configuration..."
    echo ""

    configure_cpu_governor
    configure_irq_affinity
    configure_huge_pages
    configure_memory_limits
    configure_vm_parameters
    configure_cpusets
    configure_network
    configure_gpu
    configure_pcie
    configure_scheduler

    echo ""
    verify_configuration
    print_kernel_cmdline_recommendation

    success "Cathedral RT configuration complete!"
    echo ""
}

# Run main if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
