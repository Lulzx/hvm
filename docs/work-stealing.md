# Work Stealing for Concurrent Term Reduction

A beginner-friendly explanation of work stealing infrastructure in HVM4.

## The Problem: Uneven Work Distribution

Imagine you have 4 workers at a restaurant, each with their own queue of orders to cook. Some workers might get easy orders (toast, salad) while others get complex ones (multi-course meals).

**Without work stealing:** Worker A finishes their easy orders and sits idle, while Worker D is drowning in complex orders. Waste of resources!

## The Solution: Work Stealing

**Work stealing** lets idle workers "steal" tasks from busy workers' queues.

```
Worker A: [done] [done] [done] → "I'm free, let me help D!"
Worker D: [complex] [complex] [complex] ← "Thanks, take one from my queue"
```

## How It Works in HVM4

Looking at `src/hvm.zig`, each worker has a **double-ended queue (deque)**:

```
         ← pop (owner takes from here)
[task1] [task2] [task3] [task4]
                              ← steal (thieves take from here)
```

1. **Owner** pops from the **head** (front) — fast, no contention
2. **Thieves** steal from the **tail** (back) — minimizes conflicts

The key is using **atomic operations** (`cmpxchgStrong`) so two workers don't accidentally steal the same task.

## Why It's Useful for Term Reduction

**Term reduction** = simplifying expressions like `(λx.x+1) 5 → 6`

In HVM, you might have:

```
SUP(branch_a, branch_b)  ← a "superposition" with two branches
```

Both branches can be reduced **independently**! So instead of:

```
reduce(branch_a)  // wait...
reduce(branch_b)  // then do this
```

You can:

```
Thread 1: reduce(branch_a)  ⎤
Thread 2: reduce(branch_b)  ⎦ simultaneously!
```

But what if `branch_a` spawns 100 sub-tasks and `branch_b` spawns only 2? Work stealing balances the load automatically.

## Benefits Summary

| Problem | Work Stealing Solution |
|---------|----------------------|
| Idle CPUs | Steal work → no wasted cores |
| Unpredictable workloads | Dynamic load balancing |
| Lock contention | Owner/thief access opposite ends |
| Task creation overhead | Local queue = cheap pushes |

It's essentially **self-balancing parallelism** — you don't need to predict which tasks will be expensive. The system adapts at runtime.
