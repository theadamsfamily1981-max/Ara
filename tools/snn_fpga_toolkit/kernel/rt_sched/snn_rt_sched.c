/*
 * SNN Real-Time Scheduler
 *
 * Provides real-time scheduling extensions for time-critical SNN computations
 * Integrates with Linux RT scheduling to provide deterministic execution
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/sched/rt.h>
#include <linux/sched/signal.h>
#include <linux/cpumask.h>
#include <linux/kthread.h>
#include <linux/slab.h>

#include "../core/snn_core.h"

/*
 * Set CPU affinity for a task
 */
static int set_cpu_affinity(struct task_struct *task, u32 cpu_mask)
{
    struct cpumask mask;
    int cpu;

    cpumask_clear(&mask);

    /* Convert mask to cpumask */
    for_each_online_cpu(cpu) {
        if (cpu_mask & (1 << cpu))
            cpumask_set_cpu(cpu, &mask);
    }

    /* Empty mask means all CPUs */
    if (cpumask_empty(&mask))
        cpumask_copy(&mask, cpu_online_mask);

    /* Set affinity */
    return set_cpus_allowed_ptr(task, &mask);
}

/*
 * Set real-time scheduling policy and priority
 */
static int set_rt_policy(struct task_struct *task, u32 priority)
{
    struct sched_param param;
    int policy;

    /* Validate priority */
    if (priority < SNN_RT_PRIO_MIN || priority > SNN_RT_PRIO_MAX) {
        pr_err("SNN_RT: Invalid priority %u (must be %d-%d)\n",
               priority, SNN_RT_PRIO_MIN, SNN_RT_PRIO_MAX);
        return -EINVAL;
    }

    /* Set SCHED_FIFO policy with specified priority */
    policy = SCHED_FIFO;
    param.sched_priority = priority;

    pr_debug("SNN_RT: Setting task %d to SCHED_FIFO priority %u\n",
             task->pid, priority);

    return sched_setscheduler_nocheck(task, policy, &param);
}

/*
 * Initialize RT scheduler
 */
int snn_rt_sched_init(struct snn_rt_scheduler *sched,
                      const snn_kernel_init_t *config)
{
    if (!sched)
        return -EINVAL;

    pr_info("SNN_RT: Initializing RT scheduler\n");

    spin_lock_init(&sched->lock);
    INIT_LIST_HEAD(&sched->tasks);
    sched->num_tasks = 0;

    atomic64_set(&sched->tasks_executed, 0);
    atomic64_set(&sched->deadline_misses, 0);

    pr_info("SNN_RT: RT scheduler initialized\n");
    return 0;
}

/*
 * Cleanup RT scheduler
 */
void snn_rt_sched_cleanup(struct snn_rt_scheduler *sched)
{
    struct snn_rt_task *rt_task, *tmp;

    if (!sched)
        return;

    pr_info("SNN_RT: Cleaning up RT scheduler\n");

    /* Reset all RT tasks to normal scheduling */
    spin_lock(&sched->lock);
    list_for_each_entry_safe(rt_task, tmp, &sched->tasks, list) {
        struct sched_param param = { .sched_priority = 0 };

        if (rt_task->task && !IS_ERR(rt_task->task)) {
            /* Reset to SCHED_NORMAL */
            sched_setscheduler_nocheck(rt_task->task, SCHED_NORMAL, &param);
        }

        list_del(&rt_task->list);
        kfree(rt_task);
    }
    sched->num_tasks = 0;
    spin_unlock(&sched->lock);

    pr_info("SNN_RT: RT scheduler cleanup complete\n");
}

/*
 * Set real-time scheduling parameters for a task
 */
int snn_rt_sched_set_params(struct snn_rt_scheduler *sched,
                            struct task_struct *task,
                            const snn_rt_sched_params_t *params)
{
    struct snn_rt_task *rt_task;
    bool found = false;
    int ret = 0;

    if (!sched || !task || !params)
        return -EINVAL;

    pr_debug("SNN_RT: Setting RT params for task %d: priority=%u\n",
             task->pid, params->priority);

    /* Find existing RT task or create new one */
    spin_lock(&sched->lock);
    list_for_each_entry(rt_task, &sched->tasks, list) {
        if (rt_task->task == task) {
            found = true;
            break;
        }
    }

    if (!found) {
        /* Create new RT task entry */
        rt_task = kzalloc(sizeof(*rt_task), GFP_ATOMIC);
        if (!rt_task) {
            spin_unlock(&sched->lock);
            return -ENOMEM;
        }

        rt_task->task = task;
        rt_task->last_execution = 0;
        rt_task->deadline_misses = 0;

        list_add_tail(&rt_task->list, &sched->tasks);
        sched->num_tasks++;
    }

    /* Update parameters */
    rt_task->params = *params;
    spin_unlock(&sched->lock);

    /* Set RT priority */
    ret = set_rt_policy(task, params->priority);
    if (ret) {
        pr_err("SNN_RT: Failed to set RT policy: %d\n", ret);
        return ret;
    }

    /* Set CPU affinity if specified */
    if (params->cpu_affinity != 0) {
        ret = set_cpu_affinity(task, params->cpu_affinity);
        if (ret) {
            pr_warn("SNN_RT: Failed to set CPU affinity: %d\n", ret);
            /* Non-fatal, continue */
        }
    }

    pr_info("SNN_RT: Task %d configured: prio=%u, affinity=0x%x\n",
            task->pid, params->priority, params->cpu_affinity);

    return 0;
}

/*
 * Check if task meets its deadline
 * This is called periodically or on task completion
 */
static void check_deadline(struct snn_rt_task *rt_task)
{
    u64 now = ktime_get_ns();
    u64 elapsed;

    if (rt_task->last_execution == 0) {
        rt_task->last_execution = now;
        return;
    }

    elapsed = now - rt_task->last_execution;

    /* Check if deadline was missed */
    if (rt_task->params.deadline_ns > 0 &&
        elapsed > rt_task->params.deadline_ns) {
        rt_task->deadline_misses++;
        pr_warn("SNN_RT: Task %d missed deadline: %llu ns (expected %llu ns)\n",
                rt_task->task->pid, elapsed, rt_task->params.deadline_ns);
    }

    rt_task->last_execution = now;
}

/*
 * Mark task execution (for statistics)
 */
void snn_rt_sched_mark_execution(struct snn_rt_scheduler *sched,
                                 struct task_struct *task)
{
    struct snn_rt_task *rt_task;

    if (!sched || !task)
        return;

    atomic64_inc(&sched->tasks_executed);

    spin_lock(&sched->lock);
    list_for_each_entry(rt_task, &sched->tasks, list) {
        if (rt_task->task == task) {
            check_deadline(rt_task);
            if (rt_task->deadline_misses > 0)
                atomic64_inc(&sched->deadline_misses);
            break;
        }
    }
    spin_unlock(&sched->lock);
}

/*
 * Get scheduler statistics
 */
void snn_rt_sched_get_stats(struct snn_rt_scheduler *sched,
                            snn_perf_stats_t *stats)
{
    if (!sched || !stats)
        return;

    stats->rt_tasks_executed = atomic64_read(&sched->tasks_executed);
    stats->rt_deadline_misses = atomic64_read(&sched->deadline_misses);
}

/*
 * Reset statistics
 */
void snn_rt_sched_reset_stats(struct snn_rt_scheduler *sched)
{
    struct snn_rt_task *rt_task;

    if (!sched)
        return;

    atomic64_set(&sched->tasks_executed, 0);
    atomic64_set(&sched->deadline_misses, 0);

    /* Reset per-task statistics */
    spin_lock(&sched->lock);
    list_for_each_entry(rt_task, &sched->tasks, list) {
        rt_task->deadline_misses = 0;
        rt_task->last_execution = 0;
    }
    spin_unlock(&sched->lock);
}

/*
 * Helper: Get current task RT parameters
 */
int snn_rt_sched_get_params(struct snn_rt_scheduler *sched,
                            struct task_struct *task,
                            snn_rt_sched_params_t *params)
{
    struct snn_rt_task *rt_task;
    bool found = false;

    if (!sched || !task || !params)
        return -EINVAL;

    spin_lock(&sched->lock);
    list_for_each_entry(rt_task, &sched->tasks, list) {
        if (rt_task->task == task) {
            *params = rt_task->params;
            found = true;
            break;
        }
    }
    spin_unlock(&sched->lock);

    return found ? 0 : -ENOENT;
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SNN Real-Time Scheduler Extensions");
