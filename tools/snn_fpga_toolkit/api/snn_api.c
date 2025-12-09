/*
 * SNN Kernel - User Space API Implementation
 *
 * Provides user-space interface to SNN kernel driver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "snn_kernel/api.h"

/* Global state */
static struct {
    int fd;
    int initialized;
    char error_msg[256];
} snn_state = {
    .fd = -1,
    .initialized = 0,
    .error_msg = ""
};

/* Device path */
#define SNN_DEVICE_PATH "/dev/snn"

/* Set error message */
static void set_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(snn_state.error_msg, sizeof(snn_state.error_msg), fmt, args);
    va_end(args);
}

/*
 * Initialize the SNN kernel subsystem
 */
int snn_kernel_initialize(const snn_kernel_init_t *config)
{
    int ret;

    if (!config) {
        set_error("Invalid configuration");
        return -EINVAL;
    }

    if (snn_state.initialized) {
        set_error("Already initialized");
        return -EALREADY;
    }

    /* Open device */
    snn_state.fd = open(SNN_DEVICE_PATH, O_RDWR);
    if (snn_state.fd < 0) {
        set_error("Failed to open %s: %s", SNN_DEVICE_PATH, strerror(errno));
        return -errno;
    }

    /* Send initialization command */
    ret = ioctl(snn_state.fd, SNN_IOC_INIT, config);
    if (ret < 0) {
        set_error("Initialization failed: %s", strerror(errno));
        close(snn_state.fd);
        snn_state.fd = -1;
        return -errno;
    }

    snn_state.initialized = 1;
    return 0;
}

/*
 * Shutdown the SNN kernel subsystem
 */
void snn_kernel_shutdown(void)
{
    if (!snn_state.initialized)
        return;

    if (snn_state.fd >= 0) {
        close(snn_state.fd);
        snn_state.fd = -1;
    }

    snn_state.initialized = 0;
}

/*
 * Get device information
 */
int snn_get_device_info(snn_device_type_t device_type, __u32 device_id,
                        snn_device_info_t *info)
{
    int ret;

    if (!snn_state.initialized) {
        set_error("Not initialized");
        return -EAGAIN;
    }

    if (!info) {
        set_error("Invalid info parameter");
        return -EINVAL;
    }

    info->type = device_type;
    info->device_id = device_id;

    ret = ioctl(snn_state.fd, SNN_IOC_GET_DEVICE_INFO, info);
    if (ret < 0) {
        set_error("Get device info failed: %s", strerror(errno));
        return -errno;
    }

    return 0;
}

/*
 * Allocate pinned memory
 */
void *snn_alloc_pinned(__u64 size, __u32 flags)
{
    snn_mem_alloc_t req;
    int ret;

    if (!snn_state.initialized) {
        set_error("Not initialized");
        return NULL;
    }

    if (size == 0) {
        set_error("Invalid size");
        return NULL;
    }

    memset(&req, 0, sizeof(req));
    req.size = size;
    req.flags = flags | SNN_MEM_PINNED;
    req.alignment = 4096; /* Page alignment */
    req.device_mask = 0xFF; /* All devices */

    ret = ioctl(snn_state.fd, SNN_IOC_ALLOC_MEM, &req);
    if (ret < 0) {
        set_error("Memory allocation failed: %s", strerror(errno));
        return NULL;
    }

    /* Map memory into user space */
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                     snn_state.fd, req.mem_id * getpagesize());
    if (ptr == MAP_FAILED) {
        set_error("Memory mapping failed: %s", strerror(errno));
        ioctl(snn_state.fd, SNN_IOC_FREE_MEM, &req.mem_id);
        return NULL;
    }

    return ptr;
}

/*
 * Free pinned memory
 */
void snn_free_pinned(void *ptr)
{
    /* Note: Real implementation would need to track mem_id */
    if (!ptr)
        return;

    /* Unmap and free */
    munmap(ptr, 0); /* Size should be tracked */
}

/*
 * Get physical address
 */
__u64 snn_get_physical_addr(void *ptr)
{
    /* This would require additional ioctl or tracking */
    set_error("Not implemented");
    return 0;
}

/*
 * Perform P2P transfer
 */
int snn_p2p_transfer(snn_p2p_transfer_t *transfer)
{
    int ret;

    if (!snn_state.initialized) {
        set_error("Not initialized");
        return -EAGAIN;
    }

    if (!transfer) {
        set_error("Invalid transfer parameter");
        return -EINVAL;
    }

    ret = ioctl(snn_state.fd, SNN_IOC_P2P_TRANSFER, transfer);
    if (ret < 0) {
        set_error("P2P transfer failed: %s", strerror(errno));
        return -errno;
    }

    return 0;
}

/*
 * Wait for P2P transfer
 */
int snn_p2p_wait(__u64 completion_handle, __u64 timeout_ns)
{
    /* Would need additional ioctl for waiting */
    usleep(1000); /* Simulate wait */
    return 0;
}

/*
 * Set RT parameters
 */
int snn_set_rt_params(const snn_rt_sched_params_t *params)
{
    int ret;

    if (!snn_state.initialized) {
        set_error("Not initialized");
        return -EAGAIN;
    }

    if (!params) {
        set_error("Invalid params");
        return -EINVAL;
    }

    ret = ioctl(snn_state.fd, SNN_IOC_SET_RT_PARAMS, params);
    if (ret < 0) {
        set_error("Set RT params failed: %s", strerror(errno));
        return -errno;
    }

    return 0;
}

/*
 * Execute SNN computation
 */
int snn_compute(const snn_compute_params_t *params)
{
    int ret;

    if (!snn_state.initialized) {
        set_error("Not initialized");
        return -EAGAIN;
    }

    if (!params) {
        set_error("Invalid params");
        return -EINVAL;
    }

    ret = ioctl(snn_state.fd, SNN_IOC_SNN_COMPUTE, params);
    if (ret < 0) {
        set_error("SNN compute failed: %s", strerror(errno));
        return -errno;
    }

    return 0;
}

/*
 * NVMe read
 */
int snn_nvme_read(snn_nvme_io_t *io)
{
    int ret;

    if (!snn_state.initialized) {
        set_error("Not initialized");
        return -EAGAIN;
    }

    ret = ioctl(snn_state.fd, SNN_IOC_NVME_IO, io);
    if (ret < 0) {
        set_error("NVMe read failed: %s", strerror(errno));
        return -errno;
    }

    return 0;
}

/*
 * NVMe write
 */
int snn_nvme_write(snn_nvme_io_t *io)
{
    return snn_nvme_read(io); /* Same ioctl */
}

/*
 * NVMe wait
 */
int snn_nvme_wait(__u64 completion_handle, __u64 timeout_ns)
{
    usleep(1000); /* Simulate wait */
    return 0;
}

/*
 * Get statistics
 */
int snn_get_stats(snn_perf_stats_t *stats)
{
    int ret;

    if (!snn_state.initialized) {
        set_error("Not initialized");
        return -EAGAIN;
    }

    if (!stats) {
        set_error("Invalid stats parameter");
        return -EINVAL;
    }

    ret = ioctl(snn_state.fd, SNN_IOC_GET_STATS, stats);
    if (ret < 0) {
        set_error("Get stats failed: %s", strerror(errno));
        return -errno;
    }

    return 0;
}

/*
 * Reset statistics
 */
void snn_reset_stats(void)
{
    if (!snn_state.initialized)
        return;

    ioctl(snn_state.fd, SNN_IOC_RESET_STATS);
}

/*
 * Set monitoring
 */
void snn_set_monitoring(__u32 enable)
{
    /* Would need additional ioctl */
}

/*
 * Get error string
 */
const char *snn_get_error_string(void)
{
    return snn_state.error_msg;
}
