/*
 * BANOS - Bio-Affective Neuromorphic Operating System
 * Negative Selection - Immune Response Module
 *
 * This module implements the "cytotoxic T-cell" response:
 * When the self-model flags a process as anomalous, this
 * module takes immediate action:
 *
 * 1. SIGSTOP the process (freeze it)
 * 2. Log the incident
 * 3. Alert the Ara daemon for human review
 * 4. Optionally SIGKILL if threshold exceeded
 *
 * The key insight: react biologically, not bureaucratically.
 * Stop first, ask questions later.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched/signal.h>
#include <linux/pid.h>
#include <linux/signal.h>
#include <linux/workqueue.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/spinlock.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/ktime.h>
#include <linux/netlink.h>
#include <net/sock.h>

#define NETLINK_BANOS       31      /* Netlink protocol for Ara daemon */
#define MAX_QUARANTINE      64      /* Max quarantined processes */
#define KILL_THRESHOLD      10      /* Anomalies before SIGKILL */

/*
 * Quarantine entry - a "detained" process
 */
struct quarantine_entry {
    pid_t pid;
    char comm[TASK_COMM_LEN];
    u64 timestamp;              /* When quarantined */
    u32 anomaly_count;          /* Total anomalies */
    u16 ngram[4];               /* Offending N-gram */
    u8 state;                   /* 0=stopped, 1=killed, 2=released */
    u8 reviewed;                /* Human reviewed */
    struct list_head list;
};

/*
 * Alert message to user-space
 */
struct banos_alert_msg {
    u32 type;                   /* Alert type */
    pid_t pid;
    char comm[TASK_COMM_LEN];
    u32 anomaly_count;
    u16 ngram[4];
    u64 timestamp;
};

static LIST_HEAD(quarantine_list);
static DEFINE_SPINLOCK(quarantine_lock);
static atomic_t quarantine_count = ATOMIC_INIT(0);

static struct sock *nl_sock = NULL;
static struct workqueue_struct *response_wq;

/* Statistics */
static atomic_t total_stops = ATOMIC_INIT(0);
static atomic_t total_kills = ATOMIC_INIT(0);
static atomic_t total_releases = ATOMIC_INIT(0);
static atomic_t false_positives = ATOMIC_INIT(0);

/*
 * Send alert to Ara daemon via netlink
 */
static void send_alert_to_ara(struct quarantine_entry *qe)
{
    struct sk_buff *skb;
    struct nlmsghdr *nlh;
    struct banos_alert_msg *msg;
    int msg_size = sizeof(*msg);

    if (!nl_sock)
        return;

    skb = nlmsg_new(msg_size, GFP_ATOMIC);
    if (!skb) {
        pr_warn("BANOS: Failed to allocate netlink message\n");
        return;
    }

    nlh = nlmsg_put(skb, 0, 0, NLMSG_DONE, msg_size, 0);
    if (!nlh) {
        kfree_skb(skb);
        return;
    }

    msg = nlmsg_data(nlh);
    msg->type = 1;  /* IMMUNE_ALERT */
    msg->pid = qe->pid;
    memcpy(msg->comm, qe->comm, TASK_COMM_LEN);
    msg->anomaly_count = qe->anomaly_count;
    memcpy(msg->ngram, qe->ngram, sizeof(msg->ngram));
    msg->timestamp = qe->timestamp;

    /* Broadcast to all listeners (Ara daemon) */
    NETLINK_CB(skb).dst_group = 1;
    nlmsg_multicast(nl_sock, skb, 0, 1, GFP_ATOMIC);
}

/*
 * Work handler for immune response
 * Does the actual process manipulation
 */
struct response_work {
    struct work_struct work;
    pid_t pid;
    u32 anomaly_count;
    u16 ngram[4];
};

static void immune_response_work(struct work_struct *work)
{
    struct response_work *rw = container_of(work, struct response_work, work);
    struct task_struct *task;
    struct quarantine_entry *qe;
    unsigned long flags;
    bool should_kill = false;

    /* Find the task */
    rcu_read_lock();
    task = find_task_by_vpid(rw->pid);
    if (!task) {
        rcu_read_unlock();
        kfree(rw);
        return;
    }
    get_task_struct(task);
    rcu_read_unlock();

    /* Check if already quarantined */
    spin_lock_irqsave(&quarantine_lock, flags);
    list_for_each_entry(qe, &quarantine_list, list) {
        if (qe->pid == rw->pid) {
            qe->anomaly_count += rw->anomaly_count;
            if (qe->anomaly_count >= KILL_THRESHOLD && qe->state == 0) {
                should_kill = true;
            }
            spin_unlock_irqrestore(&quarantine_lock, flags);
            goto respond;
        }
    }

    /* New quarantine entry */
    if (atomic_read(&quarantine_count) >= MAX_QUARANTINE) {
        spin_unlock_irqrestore(&quarantine_lock, flags);
        pr_warn("BANOS: Quarantine full, cannot detain PID %d\n", rw->pid);
        put_task_struct(task);
        kfree(rw);
        return;
    }

    qe = kzalloc(sizeof(*qe), GFP_ATOMIC);
    if (!qe) {
        spin_unlock_irqrestore(&quarantine_lock, flags);
        put_task_struct(task);
        kfree(rw);
        return;
    }

    qe->pid = rw->pid;
    get_task_comm(qe->comm, task);
    qe->timestamp = ktime_get_real_ns();
    qe->anomaly_count = rw->anomaly_count;
    memcpy(qe->ngram, rw->ngram, sizeof(qe->ngram));
    qe->state = 0;
    qe->reviewed = 0;

    list_add(&qe->list, &quarantine_list);
    atomic_inc(&quarantine_count);
    spin_unlock_irqrestore(&quarantine_lock, flags);

respond:
    /* Take action based on severity */
    if (should_kill) {
        /* Kill the process */
        pr_crit("BANOS IMMUNE: KILLING PID %d (%s) - %d anomalies exceeded threshold\n",
               task->pid, qe->comm, qe->anomaly_count);

        send_sig(SIGKILL, task, 1);
        qe->state = 1;
        atomic_inc(&total_kills);
    } else {
        /* Stop the process */
        pr_warn("BANOS IMMUNE: STOPPING PID %d (%s) - anomalous syscall pattern\n",
               task->pid, qe->comm);
        pr_warn("  Offending N-gram: [%d, %d, %d, %d]\n",
               rw->ngram[0], rw->ngram[1], rw->ngram[2], rw->ngram[3]);

        send_sig(SIGSTOP, task, 1);
        atomic_inc(&total_stops);
    }

    /* Alert Ara daemon */
    send_alert_to_ara(qe);

    put_task_struct(task);
    kfree(rw);
}

/*
 * Trigger immune response (called from self_model.c)
 * This queues the work to avoid blocking in syscall context
 */
void banos_immune_respond(pid_t pid, u32 anomaly_count, const u16 *ngram)
{
    struct response_work *rw;

    rw = kmalloc(sizeof(*rw), GFP_ATOMIC);
    if (!rw)
        return;

    INIT_WORK(&rw->work, immune_response_work);
    rw->pid = pid;
    rw->anomaly_count = anomaly_count;
    if (ngram)
        memcpy(rw->ngram, ngram, sizeof(rw->ngram));

    queue_work(response_wq, &rw->work);
}
EXPORT_SYMBOL_GPL(banos_immune_respond);

/*
 * Release a quarantined process (human approved)
 */
int banos_immune_release(pid_t pid)
{
    struct quarantine_entry *qe;
    struct task_struct *task;
    unsigned long flags;
    int ret = -ESRCH;

    spin_lock_irqsave(&quarantine_lock, flags);
    list_for_each_entry(qe, &quarantine_list, list) {
        if (qe->pid == pid && qe->state == 0) {
            qe->state = 2;  /* Released */
            qe->reviewed = 1;
            spin_unlock_irqrestore(&quarantine_lock, flags);

            /* Send SIGCONT */
            rcu_read_lock();
            task = find_task_by_vpid(pid);
            if (task) {
                get_task_struct(task);
                rcu_read_unlock();
                send_sig(SIGCONT, task, 1);
                put_task_struct(task);
                pr_info("BANOS IMMUNE: Released PID %d (%s)\n", pid, qe->comm);
                atomic_inc(&total_releases);
                ret = 0;
            } else {
                rcu_read_unlock();
            }
            return ret;
        }
    }
    spin_unlock_irqrestore(&quarantine_lock, flags);

    return ret;
}
EXPORT_SYMBOL_GPL(banos_immune_release);

/*
 * Mark as false positive (for learning)
 */
int banos_immune_false_positive(pid_t pid)
{
    struct quarantine_entry *qe;
    unsigned long flags;

    spin_lock_irqsave(&quarantine_lock, flags);
    list_for_each_entry(qe, &quarantine_list, list) {
        if (qe->pid == pid) {
            qe->reviewed = 1;
            atomic_inc(&false_positives);
            spin_unlock_irqrestore(&quarantine_lock, flags);

            pr_info("BANOS IMMUNE: PID %d marked as false positive\n", pid);
            /* TODO: Add N-gram to self-model */
            return 0;
        }
    }
    spin_unlock_irqrestore(&quarantine_lock, flags);

    return -ESRCH;
}
EXPORT_SYMBOL_GPL(banos_immune_false_positive);

/*
 * Kill a quarantined process (human confirmed threat)
 */
int banos_immune_kill(pid_t pid)
{
    struct quarantine_entry *qe;
    struct task_struct *task;
    unsigned long flags;

    spin_lock_irqsave(&quarantine_lock, flags);
    list_for_each_entry(qe, &quarantine_list, list) {
        if (qe->pid == pid && qe->state == 0) {
            qe->state = 1;  /* Killed */
            qe->reviewed = 1;
            spin_unlock_irqrestore(&quarantine_lock, flags);

            rcu_read_lock();
            task = find_task_by_vpid(pid);
            if (task) {
                get_task_struct(task);
                rcu_read_unlock();
                send_sig(SIGKILL, task, 1);
                put_task_struct(task);
                pr_warn("BANOS IMMUNE: Killed PID %d by human order\n", pid);
                atomic_inc(&total_kills);
                return 0;
            }
            rcu_read_unlock();
            return -ESRCH;
        }
    }
    spin_unlock_irqrestore(&quarantine_lock, flags);

    return -ESRCH;
}
EXPORT_SYMBOL_GPL(banos_immune_kill);

/*
 * Procfs interface
 */
static int quarantine_show(struct seq_file *m, void *v)
{
    struct quarantine_entry *qe;
    const char *state_str[] = {"STOPPED", "KILLED", "RELEASED"};

    seq_printf(m, "BANOS Immune System - Quarantine\n");
    seq_printf(m, "================================\n\n");

    seq_printf(m, "Statistics:\n");
    seq_printf(m, "  Total stops:     %d\n", atomic_read(&total_stops));
    seq_printf(m, "  Total kills:     %d\n", atomic_read(&total_kills));
    seq_printf(m, "  Total releases:  %d\n", atomic_read(&total_releases));
    seq_printf(m, "  False positives: %d\n\n", atomic_read(&false_positives));

    seq_printf(m, "Quarantined Processes (%d):\n", atomic_read(&quarantine_count));
    seq_printf(m, "%-8s %-16s %-10s %-8s %-24s %s\n",
              "PID", "COMM", "STATE", "ANOMALIES", "N-GRAM", "REVIEWED");

    rcu_read_lock();
    list_for_each_entry_rcu(qe, &quarantine_list, list) {
        seq_printf(m, "%-8d %-16s %-10s %-8d [%d,%d,%d,%d]        %s\n",
                  qe->pid, qe->comm, state_str[qe->state], qe->anomaly_count,
                  qe->ngram[0], qe->ngram[1], qe->ngram[2], qe->ngram[3],
                  qe->reviewed ? "YES" : "NO");
    }
    rcu_read_unlock();

    return 0;
}

static int quarantine_open(struct inode *inode, struct file *file)
{
    return single_open(file, quarantine_show, NULL);
}

/*
 * Write interface for control:
 * "release <pid>" - release process
 * "kill <pid>"    - kill process
 * "fp <pid>"      - mark as false positive
 */
static ssize_t quarantine_write(struct file *file, const char __user *buf,
                               size_t count, loff_t *ppos)
{
    char kbuf[64];
    char cmd[16];
    pid_t pid;
    int ret;

    if (count >= sizeof(kbuf))
        return -EINVAL;

    if (copy_from_user(kbuf, buf, count))
        return -EFAULT;

    kbuf[count] = '\0';

    if (sscanf(kbuf, "%15s %d", cmd, &pid) != 2)
        return -EINVAL;

    if (strcmp(cmd, "release") == 0) {
        ret = banos_immune_release(pid);
    } else if (strcmp(cmd, "kill") == 0) {
        ret = banos_immune_kill(pid);
    } else if (strcmp(cmd, "fp") == 0) {
        ret = banos_immune_false_positive(pid);
        if (ret == 0)
            ret = banos_immune_release(pid);
    } else {
        return -EINVAL;
    }

    return ret < 0 ? ret : count;
}

static const struct proc_ops quarantine_proc_ops = {
    .proc_open    = quarantine_open,
    .proc_read    = seq_read,
    .proc_write   = quarantine_write,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};

static struct proc_dir_entry *quarantine_proc_entry;

/*
 * Netlink receive handler (for commands from Ara daemon)
 */
static void nl_recv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh;
    char *msg;

    nlh = nlmsg_hdr(skb);
    msg = nlmsg_data(nlh);

    /* Parse command from Ara daemon */
    pr_debug("BANOS: Received netlink message: %s\n", msg);

    /* Commands handled via procfs for now */
}

static struct netlink_kernel_cfg nl_cfg = {
    .input = nl_recv_msg,
};

/*
 * Module init
 */
static int __init banos_negative_selection_init(void)
{
    response_wq = alloc_workqueue("banos_immune", WQ_UNBOUND | WQ_HIGHPRI, 0);
    if (!response_wq) {
        pr_err("BANOS: Failed to create response workqueue\n");
        return -ENOMEM;
    }

    nl_sock = netlink_kernel_create(&init_net, NETLINK_BANOS, &nl_cfg);
    if (!nl_sock) {
        pr_warn("BANOS: Failed to create netlink socket, alerts disabled\n");
    }

    quarantine_proc_entry = proc_create("banos_quarantine", 0644, NULL,
                                        &quarantine_proc_ops);
    if (!quarantine_proc_entry) {
        pr_err("BANOS: Failed to create proc entry\n");
        if (nl_sock)
            netlink_kernel_release(nl_sock);
        destroy_workqueue(response_wq);
        return -ENOMEM;
    }

    pr_info("BANOS Negative Selection initialized\n");
    return 0;
}

/*
 * Module exit
 */
static void __exit banos_negative_selection_exit(void)
{
    struct quarantine_entry *qe, *tmp;

    proc_remove(quarantine_proc_entry);

    if (nl_sock)
        netlink_kernel_release(nl_sock);

    flush_workqueue(response_wq);
    destroy_workqueue(response_wq);

    /* Free quarantine list */
    list_for_each_entry_safe(qe, tmp, &quarantine_list, list) {
        list_del(&qe->list);
        kfree(qe);
    }

    pr_info("BANOS Negative Selection unloaded\n");
}

module_init(banos_negative_selection_init);
module_exit(banos_negative_selection_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("BANOS Project");
MODULE_DESCRIPTION("BANOS Negative Selection - Immune Response System");
MODULE_VERSION("1.0");
