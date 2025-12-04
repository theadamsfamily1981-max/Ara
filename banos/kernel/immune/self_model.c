/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * Self-Model - Syscall N-gram Learning for Immune System
 *
 * During a "healthy" training phase, the system learns the normal
 * system call sequences (N-grams) of critical processes:
 * - Ara daemon
 * - Kernel threads
 * - Init/systemd
 *
 * The Self-Model is a hash table of observed N-gram patterns.
 * During detection, any syscall sequence NOT in the model is
 * flagged as a potential antigen (intrusion).
 *
 * This is inspired by biological negative selection in the immune
 * system, where T-cells that react to "self" are eliminated.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/hashtable.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/tracepoint.h>
#include <linux/syscalls.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/cred.h>
#include <linux/list.h>
#include <linux/rculist.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

#define NGRAM_SIZE      4       /* 4-gram syscall sequences */
#define HASH_BITS       16      /* 65536 buckets */
#define MAX_TRUSTED_PROCS 64    /* Max processes to track */

/*
 * N-gram structure
 * Represents a sequence of NGRAM_SIZE syscalls
 */
struct syscall_ngram {
    u16 syscalls[NGRAM_SIZE];   /* Syscall numbers */
    u32 hash;                   /* Precomputed hash */
    atomic_t count;             /* How many times observed */
    struct hlist_node node;     /* Hash table linkage */
};

/*
 * Per-process syscall history
 */
struct proc_history {
    pid_t pid;
    char comm[TASK_COMM_LEN];
    u16 recent[NGRAM_SIZE];     /* Sliding window of recent syscalls */
    u8 recent_idx;              /* Current position in window */
    u8 is_trusted;              /* Part of self-model */
    atomic_t anomaly_count;     /* Anomalies detected */
    struct list_head list;
};

/*
 * Self-model state
 */
static DEFINE_HASHTABLE(self_model, HASH_BITS);
static DEFINE_SPINLOCK(model_lock);
static LIST_HEAD(tracked_procs);
static DEFINE_SPINLOCK(proc_lock);

static atomic_t ngrams_learned = ATOMIC_INIT(0);
static atomic_t ngrams_matched = ATOMIC_INIT(0);
static atomic_t ngrams_missed = ATOMIC_INIT(0);

static bool learning_mode = true;
static bool detection_enabled = false;

/*
 * Compute hash for an N-gram
 * Uses FNV-1a for good distribution
 */
static u32 ngram_hash(const u16 *syscalls)
{
    u32 hash = 2166136261u;  /* FNV offset basis */
    int i;

    for (i = 0; i < NGRAM_SIZE; i++) {
        hash ^= syscalls[i];
        hash *= 16777619u;   /* FNV prime */
    }

    return hash;
}

/*
 * Check if N-gram exists in self-model
 */
static struct syscall_ngram *ngram_lookup(const u16 *syscalls, u32 hash)
{
    struct syscall_ngram *ng;

    hash_for_each_possible_rcu(self_model, ng, node, hash) {
        if (ng->hash == hash &&
            memcmp(ng->syscalls, syscalls, sizeof(ng->syscalls)) == 0) {
            return ng;
        }
    }

    return NULL;
}

/*
 * Add N-gram to self-model (learning mode)
 */
static int ngram_learn(const u16 *syscalls)
{
    struct syscall_ngram *ng;
    u32 hash = ngram_hash(syscalls);
    unsigned long flags;

    /* Check if already learned */
    rcu_read_lock();
    ng = ngram_lookup(syscalls, hash);
    if (ng) {
        atomic_inc(&ng->count);
        rcu_read_unlock();
        return 0;  /* Already known */
    }
    rcu_read_unlock();

    /* Allocate new N-gram */
    ng = kmalloc(sizeof(*ng), GFP_ATOMIC);
    if (!ng)
        return -ENOMEM;

    memcpy(ng->syscalls, syscalls, sizeof(ng->syscalls));
    ng->hash = hash;
    atomic_set(&ng->count, 1);

    /* Add to hash table */
    spin_lock_irqsave(&model_lock, flags);
    hash_add_rcu(self_model, &ng->node, hash);
    spin_unlock_irqrestore(&model_lock, flags);

    atomic_inc(&ngrams_learned);
    return 1;  /* New pattern learned */
}

/*
 * Check N-gram against self-model (detection mode)
 * Returns:
 *   0 = match (self)
 *   1 = mismatch (potential antigen)
 */
static int ngram_check(const u16 *syscalls)
{
    struct syscall_ngram *ng;
    u32 hash = ngram_hash(syscalls);

    rcu_read_lock();
    ng = ngram_lookup(syscalls, hash);
    rcu_read_unlock();

    if (ng) {
        atomic_inc(&ngrams_matched);
        return 0;  /* Known pattern */
    }

    atomic_inc(&ngrams_missed);
    return 1;  /* Unknown - potential threat */
}

/*
 * Get or create process history
 */
static struct proc_history *get_proc_history(struct task_struct *task)
{
    struct proc_history *ph;
    unsigned long flags;

    /* Search existing */
    spin_lock_irqsave(&proc_lock, flags);
    list_for_each_entry(ph, &tracked_procs, list) {
        if (ph->pid == task->pid) {
            spin_unlock_irqrestore(&proc_lock, flags);
            return ph;
        }
    }
    spin_unlock_irqrestore(&proc_lock, flags);

    /* Create new */
    ph = kzalloc(sizeof(*ph), GFP_ATOMIC);
    if (!ph)
        return NULL;

    ph->pid = task->pid;
    get_task_comm(ph->comm, task);
    ph->recent_idx = 0;
    ph->is_trusted = learning_mode;  /* Trust during learning */
    atomic_set(&ph->anomaly_count, 0);

    spin_lock_irqsave(&proc_lock, flags);
    list_add_rcu(&ph->list, &tracked_procs);
    spin_unlock_irqrestore(&proc_lock, flags);

    return ph;
}

/*
 * Record syscall and check/learn N-gram
 * Called from tracepoint hook
 */
void banos_record_syscall(struct task_struct *task, int syscall_nr)
{
    struct proc_history *ph;
    int i;

    if (!task || syscall_nr < 0 || syscall_nr > 0xFFFF)
        return;

    ph = get_proc_history(task);
    if (!ph)
        return;

    /* Add to sliding window */
    ph->recent[ph->recent_idx] = (u16)syscall_nr;
    ph->recent_idx = (ph->recent_idx + 1) % NGRAM_SIZE;

    /* Need full window to form N-gram */
    if (ph->recent_idx != 0)
        return;

    /* Build N-gram in order */
    u16 ngram[NGRAM_SIZE];
    for (i = 0; i < NGRAM_SIZE; i++) {
        ngram[i] = ph->recent[(ph->recent_idx + i) % NGRAM_SIZE];
    }

    if (learning_mode && ph->is_trusted) {
        /* Learning: add to self-model */
        ngram_learn(ngram);
    } else if (detection_enabled) {
        /* Detection: check against model */
        if (ngram_check(ngram) != 0) {
            /* Anomaly detected! */
            atomic_inc(&ph->anomaly_count);

            /* Alert if threshold exceeded */
            if (atomic_read(&ph->anomaly_count) > 5) {
                pr_warn("BANOS IMMUNE: Anomalous syscall pattern from %s (pid %d)\n",
                       ph->comm, ph->pid);
                pr_warn("  N-gram: [%d, %d, %d, %d]\n",
                       ngram[0], ngram[1], ngram[2], ngram[3]);

                /* Trigger immune response (will be handled by user daemon) */
                /* In full implementation: send signal to Ara daemon */
            }
        } else {
            /* Reset anomaly count on match */
            if (atomic_read(&ph->anomaly_count) > 0)
                atomic_dec(&ph->anomaly_count);
        }
    }
}
EXPORT_SYMBOL_GPL(banos_record_syscall);

/*
 * Mark process as trusted (for learning)
 */
int banos_trust_process(pid_t pid)
{
    struct proc_history *ph;
    struct task_struct *task;
    unsigned long flags;

    task = get_pid_task(find_get_pid(pid), PIDTYPE_PID);
    if (!task)
        return -ESRCH;

    ph = get_proc_history(task);
    put_task_struct(task);

    if (!ph)
        return -ENOMEM;

    spin_lock_irqsave(&proc_lock, flags);
    ph->is_trusted = 1;
    spin_unlock_irqrestore(&proc_lock, flags);

    pr_info("BANOS IMMUNE: Process %d (%s) marked as trusted\n", pid, ph->comm);
    return 0;
}
EXPORT_SYMBOL_GPL(banos_trust_process);

/*
 * Switch from learning to detection mode
 */
void banos_immune_arm(void)
{
    learning_mode = false;
    detection_enabled = true;
    pr_info("BANOS IMMUNE: Armed - %d N-grams in self-model\n",
           atomic_read(&ngrams_learned));
}
EXPORT_SYMBOL_GPL(banos_immune_arm);

/*
 * Switch back to learning mode
 */
void banos_immune_disarm(void)
{
    detection_enabled = false;
    learning_mode = true;
    pr_info("BANOS IMMUNE: Disarmed - returning to learning mode\n");
}
EXPORT_SYMBOL_GPL(banos_immune_disarm);

/*
 * Clear self-model (for retraining)
 */
void banos_immune_reset(void)
{
    struct syscall_ngram *ng;
    struct hlist_node *tmp;
    unsigned long flags;
    int bkt;

    spin_lock_irqsave(&model_lock, flags);
    hash_for_each_safe(self_model, bkt, tmp, ng, node) {
        hash_del_rcu(&ng->node);
        kfree_rcu(ng, node);  /* Note: needs rcu_head in struct */
    }
    spin_unlock_irqrestore(&model_lock, flags);

    atomic_set(&ngrams_learned, 0);
    atomic_set(&ngrams_matched, 0);
    atomic_set(&ngrams_missed, 0);

    pr_info("BANOS IMMUNE: Self-model cleared\n");
}
EXPORT_SYMBOL_GPL(banos_immune_reset);

/*
 * Get statistics
 */
void banos_immune_stats(int *learned, int *matched, int *missed)
{
    if (learned) *learned = atomic_read(&ngrams_learned);
    if (matched) *matched = atomic_read(&ngrams_matched);
    if (missed)  *missed  = atomic_read(&ngrams_missed);
}
EXPORT_SYMBOL_GPL(banos_immune_stats);

/*
 * Procfs interface for status
 */
static int immune_status_show(struct seq_file *m, void *v)
{
    struct proc_history *ph;

    seq_printf(m, "BANOS Immune System Status\n");
    seq_printf(m, "==========================\n");
    seq_printf(m, "Mode: %s\n", learning_mode ? "LEARNING" : "DETECTION");
    seq_printf(m, "Detection: %s\n", detection_enabled ? "ENABLED" : "DISABLED");
    seq_printf(m, "\nSelf-Model Statistics:\n");
    seq_printf(m, "  N-grams learned: %d\n", atomic_read(&ngrams_learned));
    seq_printf(m, "  N-grams matched: %d\n", atomic_read(&ngrams_matched));
    seq_printf(m, "  N-grams missed:  %d\n", atomic_read(&ngrams_missed));

    seq_printf(m, "\nTracked Processes:\n");
    rcu_read_lock();
    list_for_each_entry_rcu(ph, &tracked_procs, list) {
        seq_printf(m, "  PID %d (%s): trusted=%d, anomalies=%d\n",
                  ph->pid, ph->comm, ph->is_trusted,
                  atomic_read(&ph->anomaly_count));
    }
    rcu_read_unlock();

    return 0;
}

static int immune_status_open(struct inode *inode, struct file *file)
{
    return single_open(file, immune_status_show, NULL);
}

static const struct proc_ops immune_proc_ops = {
    .proc_open    = immune_status_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};

static struct proc_dir_entry *immune_proc_entry;

/*
 * Module init
 */
static int __init banos_immune_init(void)
{
    hash_init(self_model);

    immune_proc_entry = proc_create("banos_immune", 0444, NULL, &immune_proc_ops);
    if (!immune_proc_entry) {
        pr_err("BANOS IMMUNE: Failed to create proc entry\n");
        return -ENOMEM;
    }

    pr_info("BANOS Immune System initialized (N-gram size: %d)\n", NGRAM_SIZE);
    return 0;
}

/*
 * Module exit
 */
static void __exit banos_immune_exit(void)
{
    struct proc_history *ph, *tmp_ph;
    struct syscall_ngram *ng;
    struct hlist_node *tmp;
    int bkt;

    proc_remove(immune_proc_entry);

    /* Free self-model */
    hash_for_each_safe(self_model, bkt, tmp, ng, node) {
        hash_del(&ng->node);
        kfree(ng);
    }

    /* Free process histories */
    list_for_each_entry_safe(ph, tmp_ph, &tracked_procs, list) {
        list_del(&ph->list);
        kfree(ph);
    }

    pr_info("BANOS Immune System unloaded\n");
}

module_init(banos_immune_init);
module_exit(banos_immune_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("BANOS Project");
MODULE_DESCRIPTION("BANOS Immune System - Syscall N-gram Self-Model");
MODULE_VERSION("1.0");
