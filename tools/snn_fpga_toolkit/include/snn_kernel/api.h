/*
 * SNN Kernel - User Space API
 *
 * Public API for applications using the SNN-optimized kernel
 */

#ifndef _SNN_API_H
#define _SNN_API_H

#include "snn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the SNN kernel subsystem
 *
 * @param config: Initialization configuration
 * @return: 0 on success, negative error code on failure
 */
int snn_kernel_initialize(const snn_kernel_init_t *config);

/*
 * Shutdown the SNN kernel subsystem
 */
void snn_kernel_shutdown(void);

/*
 * Get device information
 *
 * @param device_type: Type of device to query
 * @param device_id: Device ID (0-based)
 * @param info: OUT parameter for device information
 * @return: 0 on success, negative error code on failure
 */
int snn_get_device_info(snn_device_type_t device_type, __u32 device_id,
                        snn_device_info_t *info);

/*
 * Allocate pinned memory accessible by specified devices
 *
 * @param size: Size in bytes
 * @param flags: Memory allocation flags (SNN_MEM_*)
 * @return: Pointer to allocated memory, NULL on failure
 */
void *snn_alloc_pinned(__u64 size, __u32 flags);

/*
 * Free pinned memory
 *
 * @param ptr: Pointer to memory allocated by snn_alloc_pinned
 */
void snn_free_pinned(void *ptr);

/*
 * Get physical address of pinned memory
 *
 * @param ptr: Virtual address
 * @return: Physical address, 0 on error
 */
__u64 snn_get_physical_addr(void *ptr);

/*
 * Perform peer-to-peer transfer between devices
 *
 * @param transfer: Transfer descriptor
 * @return: 0 on success, negative error code on failure
 */
int snn_p2p_transfer(snn_p2p_transfer_t *transfer);

/*
 * Wait for async P2P transfer completion
 *
 * @param completion_handle: Handle returned by async transfer
 * @param timeout_ns: Timeout in nanoseconds (0 = infinite)
 * @return: 0 on success, negative error code on failure
 */
int snn_p2p_wait(__u64 completion_handle, __u64 timeout_ns);

/*
 * Set real-time scheduling parameters for current thread
 *
 * @param params: RT scheduling parameters
 * @return: 0 on success, negative error code on failure
 */
int snn_set_rt_params(const snn_rt_sched_params_t *params);

/*
 * Execute SNN computation
 *
 * @param params: Computation parameters
 * @return: 0 on success, negative error code on failure
 */
int snn_compute(const snn_compute_params_t *params);

/*
 * Perform NVMe direct I/O read
 *
 * @param io: I/O request descriptor
 * @return: 0 on success, negative error code on failure
 */
int snn_nvme_read(snn_nvme_io_t *io);

/*
 * Perform NVMe direct I/O write
 *
 * @param io: I/O request descriptor
 * @return: 0 on success, negative error code on failure
 */
int snn_nvme_write(snn_nvme_io_t *io);

/*
 * Wait for NVMe I/O completion
 *
 * @param completion_handle: Handle returned by async I/O
 * @param timeout_ns: Timeout in nanoseconds (0 = infinite)
 * @return: 0 on success, negative error code on failure
 */
int snn_nvme_wait(__u64 completion_handle, __u64 timeout_ns);

/*
 * Get performance statistics
 *
 * @param stats: OUT parameter for statistics
 * @return: 0 on success, negative error code on failure
 */
int snn_get_stats(snn_perf_stats_t *stats);

/*
 * Reset performance statistics
 */
void snn_reset_stats(void);

/*
 * Enable/disable real-time monitoring
 *
 * @param enable: 1 to enable, 0 to disable
 */
void snn_set_monitoring(__u32 enable);

/*
 * Get last error string
 *
 * @return: Human-readable error description
 */
const char *snn_get_error_string(void);

#ifdef __cplusplus
}
#endif

#endif /* _SNN_API_H */
