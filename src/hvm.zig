const std = @import("std");
const print = std.debug.print;

// =============================================================================
// Type Definitions
// =============================================================================

pub const Tag = u8;
pub const Lab = u16;
pub const Loc = u40;
pub const Term = u64;

// =============================================================================
// Tag Constants (Node Types)
// =============================================================================

pub const DP0: Tag = 0x00; // Dup node, 1st projection
pub const DP1: Tag = 0x01; // Dup node, 2nd projection
pub const VAR: Tag = 0x02; // Variable (lambda body pointer)
pub const SUB: Tag = 0x03; // Substitution marker
pub const REF: Tag = 0x04; // Function reference
pub const LET: Tag = 0x05; // Let binding
pub const APP: Tag = 0x06; // Application
pub const ERA: Tag = 0x07; // Erasure
pub const MAT: Tag = 0x08; // Pattern matching
pub const IFL: Tag = 0x09; // If-less conditional
pub const SWI: Tag = 0x0A; // Switch statement
pub const OPX: Tag = 0x0B; // Operator node (1st arg)
pub const OPY: Tag = 0x0C; // Operator node (2nd arg)
pub const LAM: Tag = 0x0E; // Lambda abstraction
pub const SUP: Tag = 0x0F; // Superposition
pub const CTR: Tag = 0x10; // Constructor
pub const W32: Tag = 0x11; // 32-bit word
pub const CHR: Tag = 0x12; // Character
pub const INC: Tag = 0x13; // Increment (reference count)
pub const DEC: Tag = 0x14; // Decrement (reference count)

// VOID term (used for erasure)
pub const VOID: Term = 0;

// =============================================================================
// Operator Constants
// =============================================================================

pub const OP_ADD: Lab = 0x00;
pub const OP_SUB: Lab = 0x01;
pub const OP_MUL: Lab = 0x02;
pub const OP_DIV: Lab = 0x03;
pub const OP_MOD: Lab = 0x04;
pub const OP_EQ: Lab = 0x05;
pub const OP_NE: Lab = 0x06;
pub const OP_LT: Lab = 0x07;
pub const OP_GT: Lab = 0x08;
pub const OP_LTE: Lab = 0x09;
pub const OP_GTE: Lab = 0x0A;
pub const OP_AND: Lab = 0x0B;
pub const OP_OR: Lab = 0x0C;
pub const OP_XOR: Lab = 0x0D;
pub const OP_LSH: Lab = 0x0E;
pub const OP_RSH: Lab = 0x0F;

// Built-in function IDs
pub const SUP_F: u16 = 0x00;
pub const DUP_F: u16 = 0x01;
pub const LOG_F: u16 = 0x02;

// =============================================================================
// Term Bit Layout (64-bit)
// Layout: 1-bit sub | 7-bit tag | 16-bit lab | 40-bit loc
// =============================================================================

const TAG_MASK: u64 = 0x7F;
const SUB_MASK: u64 = 1 << 7;
const LAB_MASK: u64 = 0xFFFF;
const LOC_MASK: u64 = 0xFFFFFFFFFF;
const LAB_SHIFT: u6 = 8;
const LOC_SHIFT: u6 = 24;

pub inline fn term_new(tag: Tag, lab: Lab, loc: u64) Term {
    return @as(Term, tag) |
        (@as(Term, lab) << LAB_SHIFT) |
        (@as(Term, loc) << LOC_SHIFT);
}

pub inline fn term_tag(x: Term) Tag {
    return @truncate(x & TAG_MASK);
}

pub inline fn term_lab(x: Term) Lab {
    return @truncate((x >> LAB_SHIFT) & LAB_MASK);
}

pub inline fn term_loc(x: Term) u64 {
    return (x >> LOC_SHIFT) & LOC_MASK;
}

pub inline fn term_get_bit(x: Term) u1 {
    return @truncate((x >> 7) & 1);
}

pub inline fn term_set_bit(x: Term) Term {
    return x | SUB_MASK;
}

pub inline fn term_rem_bit(x: Term) Term {
    return x & ~SUB_MASK;
}

pub inline fn term_set_loc(x: Term, loc: u64) Term {
    return (x & 0x0000000000FFFFFF) | (@as(Term, loc) << LOC_SHIFT);
}

pub inline fn term_is_atom(t: Term) bool {
    const tg = term_tag(t);
    return tg == ERA or tg == W32 or tg == CHR;
}

// =============================================================================
// HVM State
// =============================================================================

pub const MAX_HEAP_SIZE: u64 = 1 << 30; // 1GB (reduced from 1TB for safety)
pub const MAX_STACK_SIZE: u64 = 1 << 24; // 16MB
pub const MAX_BOOK_SIZE: usize = 65536;

pub const BookFn = *const fn (Term) Term;

/// HVM runtime state - hot fields grouped together for cache efficiency
pub const State = struct {
    // Hot fields (accessed every reduction step) - grouped at start
    heap: []Term, // Heap buffer
    sbuf: []Term, // Stack buffer
    size: u64, // Heap size (next free location)
    spos: u64, // Stack position
    itrs: u64, // Interaction counter
    frsh: u64, // Fresh label generator

    // Cold fields (accessed less frequently)
    book: [MAX_BOOK_SIZE]?BookFn, // Function lookup table
    cari: [MAX_BOOK_SIZE]u16, // Constructor arities
    clen: [MAX_BOOK_SIZE]u16, // Case lengths
    cadt: [MAX_BOOK_SIZE]u16, // ADT identifiers
    fari: [MAX_BOOK_SIZE]u16, // Function arities

    pub fn init(allocator: std.mem.Allocator) !*State {
        const state = try allocator.create(State);
        state.heap = try allocator.alloc(Term, MAX_HEAP_SIZE);
        state.sbuf = try allocator.alloc(Term, MAX_STACK_SIZE);
        state.spos = 0;
        state.size = 1; // Reserve location 0
        state.itrs = 0;
        state.frsh = 0x20;
        state.book = [_]?BookFn{null} ** MAX_BOOK_SIZE;
        state.cari = [_]u16{0} ** MAX_BOOK_SIZE;
        state.clen = [_]u16{0} ** MAX_BOOK_SIZE;
        state.cadt = [_]u16{0} ** MAX_BOOK_SIZE;
        state.fari = [_]u16{0} ** MAX_BOOK_SIZE;

        // Initialize built-in functions
        state.book[SUP_F] = SUP_f;
        state.book[DUP_F] = DUP_f;
        state.book[LOG_F] = LOG_f;

        // Initialize heap to avoid undefined behavior
        @memset(state.heap, 0);
        @memset(state.sbuf, 0);

        return state;
    }

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        allocator.free(self.sbuf);
        allocator.free(self.heap);
        allocator.destroy(self);
    }

    pub fn define(self: *State, fid: u16, func: BookFn) void {
        self.book[fid] = func;
    }

    pub fn set_cari(self: *State, cid: u16, arity: u16) void {
        self.cari[cid] = arity;
    }

    pub fn set_fari(self: *State, fid: u16, arity: u16) void {
        self.fari[fid] = arity;
    }

    pub fn set_clen(self: *State, cid: u16, cases: u16) void {
        self.clen[cid] = cases;
    }

    pub fn set_cadt(self: *State, cid: u16, adt: u16) void {
        self.cadt[cid] = adt;
    }
};

// Thread-local HVM state for parallel execution
threadlocal var HVM: *State = undefined;

pub fn hvm_init(allocator: std.mem.Allocator) !void {
    HVM = try State.init(allocator);
}

pub fn hvm_free(allocator: std.mem.Allocator) void {
    HVM.deinit(allocator);
}

pub fn hvm_get_state() *State {
    return HVM;
}

pub fn hvm_set_state(state: *State) void {
    HVM = state;
}

// =============================================================================
// Parallel Execution
// =============================================================================

/// Number of worker threads (auto-detect CPU cores)
pub const NUM_WORKERS = 12; // M4 Pro has 12 cores

/// Thread-local states for parallel execution
var worker_states: [NUM_WORKERS]?*State = [_]?*State{null} ** NUM_WORKERS;
var worker_allocator: ?std.mem.Allocator = null;

/// Initialize parallel execution with thread-local states
pub fn parallel_init(allocator: std.mem.Allocator) !void {
    worker_allocator = allocator;
    for (0..NUM_WORKERS) |i| {
        worker_states[i] = try State.init(allocator);
    }
}

/// Free parallel execution resources
pub fn parallel_free() void {
    if (worker_allocator) |alloc| {
        for (0..NUM_WORKERS) |i| {
            if (worker_states[i]) |state| {
                state.deinit(alloc);
                worker_states[i] = null;
            }
        }
    }
}

/// Batch reduce multiple terms in parallel
pub fn parallel_reduce_batch(terms: []Term, results: []Term) void {
    const chunk_size = (terms.len + NUM_WORKERS - 1) / NUM_WORKERS;

    var threads: [NUM_WORKERS]?std.Thread = [_]?std.Thread{null} ** NUM_WORKERS;

    for (0..NUM_WORKERS) |i| {
        const start = i * chunk_size;
        if (start >= terms.len) break;
        const end = @min(start + chunk_size, terms.len);

        threads[i] = std.Thread.spawn(.{}, worker_reduce_batch, .{
            @as(u32, @intCast(i)),
            terms[start..end],
            results[start..end]
        }) catch null;
    }

    // Wait for all threads
    for (threads) |maybe_thread| {
        if (maybe_thread) |thread| {
            thread.join();
        }
    }
}

fn worker_reduce_batch(worker_id: u32, terms: []Term, results: []Term) void {
    // Set thread-local state
    if (worker_states[worker_id]) |state| {
        HVM = state;
        for (terms, 0..) |term, i| {
            results[i] = reduce(term);
        }
    }
}

// =============================================================================
// Fast Batch Arithmetic (bypasses full reduction for pure numeric ops)
// =============================================================================

/// Batch add: compute a[i] + b[i] for all i, store in results
/// This bypasses the full reduction machinery for maximum speed
pub fn batch_add(a: []const u32, b: []const u32, results: []u32) void {
    const vec_size = 4;
    const Vec = @Vector(vec_size, u32);

    var i: usize = 0;
    // SIMD loop
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: Vec = a[i..][0..vec_size].*;
        const vb: Vec = b[i..][0..vec_size].*;
        const vr = va +% vb;
        results[i..][0..vec_size].* = vr;
    }
    // Scalar remainder
    while (i < a.len) : (i += 1) {
        results[i] = a[i] +% b[i];
    }
}

/// Batch multiply
pub fn batch_mul(a: []const u32, b: []const u32, results: []u32) void {
    const vec_size = 4;
    const Vec = @Vector(vec_size, u32);

    var i: usize = 0;
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: Vec = a[i..][0..vec_size].*;
        const vb: Vec = b[i..][0..vec_size].*;
        const vr = va *% vb;
        results[i..][0..vec_size].* = vr;
    }
    while (i < a.len) : (i += 1) {
        results[i] = a[i] *% b[i];
    }
}

/// Batch subtract
pub fn batch_sub(a: []const u32, b: []const u32, results: []u32) void {
    const vec_size = 4;
    const Vec = @Vector(vec_size, u32);

    var i: usize = 0;
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: Vec = a[i..][0..vec_size].*;
        const vb: Vec = b[i..][0..vec_size].*;
        const vr = va -% vb;
        results[i..][0..vec_size].* = vr;
    }
    while (i < a.len) : (i += 1) {
        results[i] = a[i] -% b[i];
    }
}

/// Generic batch binary operation with SIMD
pub fn batch_op(comptime op: Lab, a: []const u32, b: []const u32, results: []u32) void {
    switch (op) {
        OP_ADD => batch_add(a, b, results),
        OP_SUB => batch_sub(a, b, results),
        OP_MUL => batch_mul(a, b, results),
        else => {
            // Fallback for other ops
            for (a, b, results) |av, bv, *r| {
                r.* = switch (op) {
                    OP_DIV => if (bv != 0) av / bv else 0,
                    OP_MOD => if (bv != 0) av % bv else 0,
                    OP_EQ => @intFromBool(av == bv),
                    OP_NE => @intFromBool(av != bv),
                    OP_LT => @intFromBool(av < bv),
                    OP_GT => @intFromBool(av > bv),
                    OP_LTE => @intFromBool(av <= bv),
                    OP_GTE => @intFromBool(av >= bv),
                    OP_AND => av & bv,
                    OP_OR => av | bv,
                    OP_XOR => av ^ bv,
                    OP_LSH => av << @truncate(bv),
                    OP_RSH => av >> @truncate(bv),
                    else => 0,
                };
            }
        },
    }
}

// =============================================================================
// Parallel SIMD Batch Operations (multi-threaded + vectorized)
// =============================================================================

/// Parallel SIMD batch add using all CPU cores
pub fn parallel_batch_add(a: []const u32, b: []const u32, results: []u32) void {
    const chunk_size = (a.len + NUM_WORKERS - 1) / NUM_WORKERS;
    var threads: [NUM_WORKERS]?std.Thread = [_]?std.Thread{null} ** NUM_WORKERS;

    for (0..NUM_WORKERS) |i| {
        const start = i * chunk_size;
        if (start >= a.len) break;
        const end = @min(start + chunk_size, a.len);

        threads[i] = std.Thread.spawn(.{}, struct {
            fn work(a_slice: []const u32, b_slice: []const u32, r_slice: []u32) void {
                batch_add(a_slice, b_slice, r_slice);
            }
        }.work, .{ a[start..end], b[start..end], results[start..end] }) catch null;
    }

    for (threads) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

/// Parallel SIMD batch multiply using all CPU cores
pub fn parallel_batch_mul(a: []const u32, b: []const u32, results: []u32) void {
    const chunk_size = (a.len + NUM_WORKERS - 1) / NUM_WORKERS;
    var threads: [NUM_WORKERS]?std.Thread = [_]?std.Thread{null} ** NUM_WORKERS;

    for (0..NUM_WORKERS) |i| {
        const start = i * chunk_size;
        if (start >= a.len) break;
        const end = @min(start + chunk_size, a.len);

        threads[i] = std.Thread.spawn(.{}, struct {
            fn work(a_slice: []const u32, b_slice: []const u32, r_slice: []u32) void {
                batch_mul(a_slice, b_slice, r_slice);
            }
        }.work, .{ a[start..end], b[start..end], results[start..end] }) catch null;
    }

    for (threads) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

// =============================================================================
// Heap Operations
// =============================================================================

pub fn set_len(size: u64) void {
    HVM.size = size;
}

pub fn set_itr(itrs: u64) void {
    HVM.itrs = itrs;
}

pub fn get_len() u64 {
    return HVM.size;
}

pub fn get_itr() u64 {
    return HVM.itrs;
}

pub fn fresh() u64 {
    const val = HVM.frsh;
    HVM.frsh += 1;
    return val;
}

/// Fast heap swap - in release mode, skips validation for speed
pub inline fn swap(loc: u64, term: Term) Term {
    if (comptime std.debug.runtime_safety) {
        const val = HVM.heap[loc];
        HVM.heap[loc] = term;
        if (val == 0) {
            print("SWAP 0 at {x}\n", .{loc});
            std.process.exit(1);
        }
        return val;
    } else {
        const val = HVM.heap[loc];
        HVM.heap[loc] = term;
        return val;
    }
}

/// Fast heap read - in release mode, skips validation for speed
pub inline fn got(loc: u64) Term {
    if (comptime std.debug.runtime_safety) {
        const val = HVM.heap[loc];
        if (val == 0) {
            print("GOT 0 at {x}\n", .{loc});
            std.process.exit(1);
        }
        return val;
    } else {
        return HVM.heap[loc];
    }
}

/// Fast heap write
pub inline fn set(loc: u64, term: Term) void {
    HVM.heap[loc] = term;
}

/// Set with substitution bit
pub inline fn sub(loc: u64, term: Term) void {
    HVM.heap[loc] = term | SUB_MASK; // Inline term_set_bit
}

/// Swap with VOID (take ownership)
pub inline fn take(loc: u64) Term {
    return swap(loc, VOID);
}

/// Bump allocator for heap nodes
pub inline fn alloc_node(arity: u64) u64 {
    if (comptime std.debug.runtime_safety) {
        if (HVM.size + arity > MAX_HEAP_SIZE) {
            print("Heap memory limit exceeded\n", .{});
            std.process.exit(1);
        }
    }
    const old = HVM.size;
    HVM.size += arity;
    return old;
}

/// Increment interaction counter
pub inline fn inc_itr() void {
    HVM.itrs += 1;
}

// =============================================================================
// Stack Operations
// =============================================================================

/// Push term onto evaluation stack
pub inline fn spush(term: Term) void {
    if (comptime std.debug.runtime_safety) {
        if (HVM.spos >= MAX_STACK_SIZE) {
            print("Stack memory limit exceeded\n", .{});
            std.process.exit(1);
        }
    }
    HVM.sbuf[HVM.spos] = term;
    HVM.spos += 1;
}

/// Pop term from evaluation stack
pub inline fn spop() Term {
    HVM.spos -= 1;
    return HVM.sbuf[HVM.spos];
}

// =============================================================================
// Interaction Rules: APP
// =============================================================================

/// Beta reduction: (λx.body arg) -> body[x := arg]
/// This is the most common interaction - highly optimized
inline fn reduce_app_lam(app: Term, lam: Term) Term {
    HVM.itrs += 1;
    const app_loc = term_loc(app);
    const lam_loc = term_loc(lam);
    const bod = HVM.heap[lam_loc];
    const arg = HVM.heap[app_loc + 1];
    HVM.heap[lam_loc] = arg | SUB_MASK; // Direct substitution
    return bod;
}

/// Erasure application: (* arg) -> *
inline fn reduce_app_era(app: Term, era: Term) Term {
    _ = app;
    HVM.itrs += 1;
    return era;
}

// ({a, b} arg) -> {(a arg0), (b arg1)} with !lab{arg0, arg1} = arg
pub fn reduce_app_sup(app: Term, sup: Term) Term {
    inc_itr();
    const app_loc = term_loc(app);
    const sup_loc = term_loc(sup);
    const sup_lab = term_lab(sup);

    const arg = got(app_loc + 1);
    const tm1 = got(sup_loc + 1);

    const loc = alloc_node(3);
    const ap0 = sup_loc;
    const ap1 = loc + 0;
    const su0 = app_loc;
    const dup = loc + 2;

    set(ap0 + 1, term_new(DP0, sup_lab, dup));
    set(ap1 + 0, tm1);
    set(ap1 + 1, term_new(DP1, sup_lab, dup));
    set(su0 + 0, term_new(APP, 0, ap0));
    set(su0 + 1, term_new(APP, 0, ap1));
    set(dup + 0, arg);

    return term_new(SUP, sup_lab, su0);
}

// (CTR arg) -> error
pub fn reduce_app_ctr(app: Term, ctr: Term) Term {
    _ = app;
    print("invalid:app-ctr({d})\n", .{term_lab(ctr)});
    std.process.exit(1);
}

// (W32 arg) -> error
pub fn reduce_app_w32(app: Term, w32: Term) Term {
    _ = app;
    print("invalid:app-w32({d})\n", .{term_loc(w32)});
    std.process.exit(1);
}

// =============================================================================
// Interaction Rules: DUP
// =============================================================================

/// Duplication of erasure: !lab{x, y} = * -> x <- *; y <- *
inline fn reduce_dup_era(dup: Term, era: Term) Term {
    HVM.itrs += 1;
    const dup_loc = term_loc(dup);
    HVM.heap[dup_loc] = era | SUB_MASK;
    return era;
}

// !lab{x, y} = lambda z.body -> x <- lambda a.(body0); y <- lambda b.(body1)
pub fn reduce_dup_lam(dup: Term, lam: Term) Term {
    inc_itr();
    const dup_loc = term_loc(dup);
    const lam_loc = term_loc(lam);
    const dup_lab = term_lab(dup);

    const bod = got(lam_loc + 0);

    const loc = alloc_node(5);
    const lm0 = loc + 0;
    const lm1 = loc + 1;
    const su0 = loc + 2;
    const du0 = loc + 4;

    sub(lam_loc + 0, term_new(SUP, dup_lab, su0));

    set(lm0 + 0, term_new(DP0, dup_lab, du0));
    set(lm1 + 0, term_new(DP1, dup_lab, du0));
    set(su0 + 0, term_new(VAR, 0, lm0));
    set(su0 + 1, term_new(VAR, 0, lm1));
    set(du0 + 0, bod);

    if (term_tag(dup) == DP0) {
        sub(dup_loc + 0, term_new(LAM, 0, lm1));
        return term_new(LAM, 0, lm0);
    } else {
        sub(dup_loc + 0, term_new(LAM, 0, lm0));
        return term_new(LAM, 0, lm1);
    }
}

/// DUP+SUP interaction: annihilation (same label) or commutation (different labels)
/// This is critical for optimal sharing - highly optimized
inline fn reduce_dup_sup(dup: Term, sup: Term) Term {
    HVM.itrs += 1;
    const dup_loc = term_loc(dup);
    const dup_lab = term_lab(dup);
    const sup_lab = term_lab(sup);
    const sup_loc = term_loc(sup);

    if (dup_lab == sup_lab) {
        // Annihilation: same labels - O(1) cancellation
        const tm0 = HVM.heap[sup_loc];
        const tm1 = HVM.heap[sup_loc + 1];
        if (term_tag(dup) == DP0) {
            HVM.heap[dup_loc] = tm1 | SUB_MASK;
            return tm0;
        } else {
            HVM.heap[dup_loc] = tm0 | SUB_MASK;
            return tm1;
        }
    } else {
        // Commutation: different labels - creates new nodes
        const loc = alloc_node(4);
        const du0 = sup_loc;
        const du1 = sup_loc + 1;
        const su0 = loc;
        const su1 = loc + 2;

        HVM.heap[su0] = term_new(DP0, dup_lab, du0);
        HVM.heap[su0 + 1] = term_new(DP0, dup_lab, du1);
        HVM.heap[su1] = term_new(DP1, dup_lab, du0);
        HVM.heap[su1 + 1] = term_new(DP1, dup_lab, du1);

        if (term_tag(dup) == DP0) {
            HVM.heap[dup_loc] = term_new(SUP, sup_lab, su1) | SUB_MASK;
            return term_new(SUP, sup_lab, su0);
        } else {
            HVM.heap[dup_loc] = term_new(SUP, sup_lab, su0) | SUB_MASK;
            return term_new(SUP, sup_lab, su1);
        }
    }
}

// !lab{x, y} = CTR{...} -> duplicate constructor
pub fn reduce_dup_ctr(dup: Term, ctr: Term) Term {
    inc_itr();
    const dup_loc = term_loc(dup);
    const dup_lab = term_lab(dup);
    const ctr_loc = term_loc(ctr);
    const ctr_lab = term_lab(ctr);
    const ctr_ari = HVM.cari[ctr_lab];

    const loc = alloc_node(ctr_ari * 2);
    const ctr0 = ctr_loc;
    const ctr1 = loc + 0;

    var i: u64 = 0;
    while (i < ctr_ari) : (i += 1) {
        const du0 = loc + ctr_ari + i;
        set(du0 + 0, got(ctr_loc + i));
        set(ctr0 + i, term_new(DP0, dup_lab, du0));
        set(ctr1 + i, term_new(DP1, dup_lab, du0));
    }

    if (term_tag(dup) == DP0) {
        sub(dup_loc + 0, term_new(CTR, ctr_lab, ctr1));
        return term_new(CTR, ctr_lab, ctr0);
    } else {
        sub(dup_loc + 0, term_new(CTR, ctr_lab, ctr0));
        return term_new(CTR, ctr_lab, ctr1);
    }
}

/// Duplication of number: !lab{x, y} = W32 -> x <- W32; y <- W32
/// Numbers are cheap to duplicate (just copy the value)
inline fn reduce_dup_w32(dup: Term, w32: Term) Term {
    HVM.itrs += 1;
    const dup_loc = term_loc(dup);
    HVM.heap[dup_loc] = w32 | SUB_MASK;
    return w32;
}

// !lab{x, y} = @ref -> expand reference first
pub fn reduce_dup_ref(dup: Term, ref: Term) Term {
    inc_itr();
    const dup_loc = term_loc(dup);
    const dup_lab = term_lab(dup);
    const ref_loc = term_loc(ref);
    const ref_lab = term_lab(ref);
    const ref_ari = HVM.fari[ref_lab];

    const loc = alloc_node(ref_ari * 2 + 2);
    const ref0 = ref_loc;
    const ref1 = loc;
    const dup0 = loc + ref_ari;

    var i: u64 = 0;
    while (i < ref_ari) : (i += 1) {
        const arg = got(ref_loc + i);
        const du0 = dup0 + i;
        set(du0, arg);
        set(ref0 + i, term_new(DP0, dup_lab, du0));
        set(ref1 + i, term_new(DP1, dup_lab, du0));
    }

    if (term_tag(dup) == DP0) {
        sub(dup_loc + 0, term_new(REF, ref_lab, ref1));
        return term_new(REF, ref_lab, ref0);
    } else {
        sub(dup_loc + 0, term_new(REF, ref_lab, ref0));
        return term_new(REF, ref_lab, ref1);
    }
}

// =============================================================================
// Interaction Rules: MAT (Pattern Matching)
// =============================================================================

pub fn reduce_mat_era(mat: Term, era: Term) Term {
    _ = mat;
    inc_itr();
    return era;
}

pub fn reduce_mat_lam(mat: Term, lam: Term) Term {
    _ = mat;
    _ = lam;
    print("invalid:mat-lam\n", .{});
    std.process.exit(1);
}

pub fn reduce_mat_sup(mat: Term, sup: Term) Term {
    inc_itr();
    const mat_tag = term_tag(mat);
    const mat_loc = term_loc(mat);
    const mat_lab = term_lab(mat);
    const sup_loc = term_loc(sup);
    const sup_lab = term_lab(sup);

    const tm0 = got(sup_loc + 0);
    const tm1 = got(sup_loc + 1);

    if (mat_tag == IFL) {
        const loc = alloc_node(6);
        const mat0 = sup_loc;
        const mat1 = loc + 0;
        const sup0 = mat_loc;
        const du0 = loc + 3;
        const du1 = loc + 4;
        const du2 = loc + 5;

        set(mat0 + 0, tm0);
        set(mat0 + 1, term_new(DP0, sup_lab, du0));
        set(mat0 + 2, term_new(DP0, sup_lab, du1));
        set(mat1 + 0, tm1);
        set(mat1 + 1, term_new(DP1, sup_lab, du0));
        set(mat1 + 2, term_new(DP1, sup_lab, du1));
        set(sup0 + 0, term_new(IFL, mat_lab, mat0));
        set(sup0 + 1, term_new(IFL, mat_lab, mat1));
        set(du0 + 0, got(mat_loc + 1));
        set(du1 + 0, got(mat_loc + 2));
        _ = du2;

        return term_new(SUP, sup_lab, sup0);
    } else if (mat_tag == SWI) {
        const mat_len = mat_lab;
        const loc = alloc_node(2 + mat_len + mat_len + 1);
        const mat0 = sup_loc;
        const mat1 = loc + 0;
        const sup0 = mat_loc;
        const dup0 = loc + 2 + mat_len;

        set(mat0 + 0, tm0);
        set(mat1 + 0, tm1);

        var i: u64 = 0;
        while (i <= mat_len) : (i += 1) {
            const dup_loc = dup0 + i;
            set(dup_loc + 0, got(mat_loc + 1 + i));
            set(mat0 + 1 + i, term_new(DP0, sup_lab, dup_loc));
            set(mat1 + 1 + i, term_new(DP1, sup_lab, dup_loc));
        }

        set(sup0 + 0, term_new(SWI, mat_lab, mat0));
        set(sup0 + 1, term_new(SWI, mat_lab, mat1));

        return term_new(SUP, sup_lab, sup0);
    } else {
        // MAT
        const mat_ctr = mat_lab;
        const clen = HVM.clen[mat_ctr];
        const loc = alloc_node(2 + clen + clen);
        const mat0 = sup_loc;
        const mat1 = loc + 0;
        const sup0 = mat_loc;
        const dup0 = loc + 2 + clen;

        set(mat0 + 0, tm0);
        set(mat1 + 0, tm1);

        var i: u64 = 0;
        while (i < clen) : (i += 1) {
            const dup_loc = dup0 + i;
            set(dup_loc + 0, got(mat_loc + 1 + i));
            set(mat0 + 1 + i, term_new(DP0, sup_lab, dup_loc));
            set(mat1 + 1 + i, term_new(DP1, sup_lab, dup_loc));
        }

        set(sup0 + 0, term_new(MAT, mat_lab, mat0));
        set(sup0 + 1, term_new(MAT, mat_lab, mat1));

        return term_new(SUP, sup_lab, sup0);
    }
}

pub fn reduce_mat_ctr(mat: Term, ctr: Term) Term {
    inc_itr();
    const mat_tag = term_tag(mat);
    const mat_loc = term_loc(mat);
    const mat_lab = term_lab(mat);

    // If-Let
    if (mat_tag == IFL) {
        const ctr_loc = term_loc(ctr);
        const ctr_lab = term_lab(ctr);
        const mat_ctr = mat_lab;
        const ctr_num = ctr_lab;
        const ctr_ari = HVM.cari[ctr_num];

        if (mat_ctr == ctr_num) {
            var app_term = got(mat_loc + 1);
            const loc = alloc_node(ctr_ari * 2);
            var i: u64 = 0;
            while (i < ctr_ari) : (i += 1) {
                const new_app = loc + i * 2;
                set(new_app + 0, app_term);
                set(new_app + 1, got(ctr_loc + i));
                app_term = term_new(APP, 0, new_app);
            }
            return app_term;
        } else {
            var app_term = got(mat_loc + 2);
            const new_app = mat_loc;
            set(new_app + 0, app_term);
            set(new_app + 1, ctr);
            app_term = term_new(APP, 0, new_app);
            return app_term;
        }
    } else {
        // Match
        const ctr_loc = term_loc(ctr);
        const ctr_lab = term_lab(ctr);
        const ctr_num = ctr_lab;
        const ctr_ari = HVM.cari[ctr_num];
        const mat_ctr = mat_lab;
        const cadt = HVM.cadt[mat_ctr];
        const clen = HVM.clen[mat_ctr];

        if (ctr_num < cadt or ctr_num >= cadt + clen) {
            print("invalid:mat-ctr({d}, {d})\n", .{ ctr_num, cadt });
            std.process.exit(1);
        }

        const cse_idx = ctr_num - mat_ctr;
        var app_term = got(mat_loc + 1 + cse_idx);
        const loc = alloc_node(ctr_ari * 2);
        var i: u64 = 0;
        while (i < ctr_ari) : (i += 1) {
            const new_app = loc + i * 2;
            set(new_app + 0, app_term);
            set(new_app + 1, got(ctr_loc + i));
            app_term = term_new(APP, 0, new_app);
        }
        return app_term;
    }
}

pub fn reduce_mat_w32(mat: Term, w32: Term) Term {
    inc_itr();
    const mat_loc = term_loc(mat);
    const mat_lab = term_lab(mat);
    const mat_len = mat_lab;
    const w32_val = term_loc(w32);

    if (w32_val < mat_len - 1) {
        return got(mat_loc + 1 + w32_val);
    } else {
        const fn_term = got(mat_loc + mat_len);
        const app_loc = mat_loc;
        set(app_loc + 0, fn_term);
        set(app_loc + 1, term_new(W32, 0, w32_val - (mat_len - 1)));
        return term_new(APP, 0, app_loc);
    }
}

// =============================================================================
// Interaction Rules: OPX (Operator, 1st argument)
// =============================================================================

pub fn reduce_opx_era(opx: Term, era: Term) Term {
    _ = opx;
    inc_itr();
    return era;
}

pub fn reduce_opx_lam(opx: Term, lam: Term) Term {
    _ = opx;
    _ = lam;
    print("invalid:opx-lam\n", .{});
    std.process.exit(1);
}

pub fn reduce_opx_sup(opx: Term, sup: Term) Term {
    inc_itr();
    const opx_loc = term_loc(opx);
    const opx_lab = term_lab(opx);
    const sup_loc = term_loc(sup);
    const sup_lab = term_lab(sup);

    const nmy = got(opx_loc + 1);
    const tm0 = got(sup_loc + 0);
    const tm1 = got(sup_loc + 1);

    const loc = alloc_node(3);
    const op0 = sup_loc;
    const op1 = loc + 0;
    const su0 = opx_loc;
    const dup = loc + 2;

    set(op0 + 0, tm0);
    set(op0 + 1, term_new(DP0, sup_lab, dup));
    set(op1 + 0, tm1);
    set(op1 + 1, term_new(DP1, sup_lab, dup));
    set(su0 + 0, term_new(OPX, opx_lab, op0));
    set(su0 + 1, term_new(OPX, opx_lab, op1));
    set(dup + 0, nmy);

    return term_new(SUP, sup_lab, su0);
}

pub fn reduce_opx_ctr(opx: Term, ctr: Term) Term {
    _ = opx;
    _ = ctr;
    print("invalid:opx-ctr\n", .{});
    std.process.exit(1);
}

/// First operand ready: convert OPX to OPY
inline fn reduce_opx_w32(opx: Term, nmx: Term) Term {
    HVM.itrs += 1;
    const opx_lab = term_lab(opx);
    const opx_loc = term_loc(opx);
    const nmy = HVM.heap[opx_loc + 1];
    HVM.heap[opx_loc] = nmy;
    HVM.heap[opx_loc + 1] = nmx;
    return term_new(OPY, opx_lab, opx_loc);
}

// =============================================================================
// Interaction Rules: OPY (Operator, 2nd argument)
// =============================================================================

pub fn reduce_opy_era(opy: Term, era: Term) Term {
    _ = opy;
    inc_itr();
    return era;
}

pub fn reduce_opy_lam(opy: Term, lam: Term) Term {
    _ = opy;
    _ = lam;
    print("invalid:opy-lam\n", .{});
    std.process.exit(1);
}

pub fn reduce_opy_sup(opy: Term, sup: Term) Term {
    inc_itr();
    const opy_loc = term_loc(opy);
    const opy_lab = term_lab(opy);
    const sup_loc = term_loc(sup);
    const sup_lab = term_lab(sup);

    const nmx = got(opy_loc + 1);
    const tm0 = got(sup_loc + 0);
    const tm1 = got(sup_loc + 1);

    const loc = alloc_node(3);
    const op0 = sup_loc;
    const op1 = loc + 0;
    const su0 = opy_loc;
    const dup = loc + 2;

    set(op0 + 0, tm0);
    set(op0 + 1, term_new(DP0, sup_lab, dup));
    set(op1 + 0, tm1);
    set(op1 + 1, term_new(DP1, sup_lab, dup));
    set(su0 + 0, term_new(OPY, opy_lab, op0));
    set(su0 + 1, term_new(OPY, opy_lab, op1));
    set(dup + 0, nmx);

    return term_new(SUP, sup_lab, su0);
}

pub fn reduce_opy_ctr(opy: Term, ctr: Term) Term {
    _ = opy;
    _ = ctr;
    print("invalid:opy-ctr\n", .{});
    std.process.exit(1);
}

/// Arithmetic evaluation: compute binary operation on two W32 values
/// Highly optimized with direct heap access
inline fn reduce_opy_w32(opy: Term, w32: Term) Term {
    HVM.itrs += 1;
    const opy_loc = term_loc(opy);
    const t = term_tag(w32);
    const x: u32 = @truncate(HVM.heap[opy_loc + 1] >> LOC_SHIFT);
    const y: u32 = @truncate(term_loc(w32));
    const op = term_lab(opy);

    const result: u32 = switch (op) {
        OP_ADD => x +% y,
        OP_SUB => x -% y,
        OP_MUL => x *% y,
        OP_DIV => if (y != 0) x / y else 0,
        OP_MOD => if (y != 0) x % y else 0,
        OP_EQ => @intFromBool(x == y),
        OP_NE => @intFromBool(x != y),
        OP_LT => @intFromBool(x < y),
        OP_GT => @intFromBool(x > y),
        OP_LTE => @intFromBool(x <= y),
        OP_GTE => @intFromBool(x >= y),
        OP_AND => x & y,
        OP_OR => x | y,
        OP_XOR => x ^ y,
        OP_LSH => x << @truncate(y),
        OP_RSH => x >> @truncate(y),
        else => {
            print("invalid:opy-w32\n", .{});
            std.process.exit(1);
        },
    };

    return term_new(t, 0, result);
}

// =============================================================================
// Reference Expansion
// =============================================================================

/// Expand function reference to its body
inline fn reduce_ref(ref: Term) Term {
    HVM.itrs += 1;
    const lab = term_lab(ref);
    if (HVM.book[lab]) |func| {
        return func(ref);
    } else {
        print("undefined function: {d}\n", .{lab});
        std.process.exit(1);
    }
}

// =============================================================================
// Let Binding
// =============================================================================

/// Let binding: let x = val in body -> body[x := val]
inline fn reduce_let(let_term: Term, val: Term) Term {
    HVM.itrs += 1;
    const let_loc = term_loc(let_term);
    const bod = HVM.heap[let_loc + 1];
    HVM.heap[let_loc] = val | SUB_MASK;
    return bod;
}

// =============================================================================
// Primitives
// =============================================================================

// Dynamic SUP: @SUP(lab tm0 tm1)
pub fn SUP_f(ref: Term) Term {
    const ref_loc = term_loc(ref);
    const lab = reduce(got(ref_loc + 0));
    const lab_val = term_loc(lab);

    if (term_tag(lab) != W32) {
        print("ERROR:non-numeric-sup-label\n", .{});
        std.process.exit(1);
    }
    if (lab_val > 0xFFFF) {
        print("ERROR:sup-label-too-large\n", .{});
        std.process.exit(1);
    }

    const tm0 = got(ref_loc + 1);
    const tm1 = got(ref_loc + 2);
    const sup_loc = alloc_node(2);
    const ret = term_new(SUP, @truncate(lab_val), sup_loc);
    set(sup_loc + 0, tm0);
    set(sup_loc + 1, tm1);
    HVM.itrs += 1;
    return ret;
}

// Dynamic DUP: @DUP(lab val lambda dp0. lambda dp1. bod)
pub fn DUP_f(ref: Term) Term {
    const ref_loc = term_loc(ref);
    const lab = reduce(got(ref_loc + 0));
    const lab_val = term_loc(lab);

    if (term_tag(lab) != W32) {
        print("ERROR:non-numeric-dup-label\n", .{});
        std.process.exit(1);
    }
    if (lab_val > 0xFFFF) {
        print("ERROR:dup-label-too-large\n", .{});
        std.process.exit(1);
    }

    const val = got(ref_loc + 1);
    const bod = got(ref_loc + 2);
    const dup_loc = alloc_node(1);
    set(dup_loc + 0, val);

    if (term_tag(bod) == LAM) {
        const lam0 = term_loc(bod);
        const bod0 = got(lam0 + 0);
        if (term_tag(bod0) == LAM) {
            const lam1 = term_loc(bod0);
            const bod1 = got(lam1 + 0);
            sub(lam0 + 0, term_new(DP0, @truncate(lab_val), dup_loc));
            sub(lam1 + 0, term_new(DP1, @truncate(lab_val), dup_loc));
            HVM.itrs += 3;
            return bod1;
        }
    }

    const app0 = alloc_node(2);
    set(app0 + 0, bod);
    set(app0 + 1, term_new(DP0, @truncate(lab_val), dup_loc));
    const app1 = alloc_node(2);
    set(app1 + 0, term_new(APP, 0, app0));
    set(app1 + 1, term_new(DP1, @truncate(lab_val), dup_loc));
    HVM.itrs += 1;
    return term_new(APP, 0, app1);
}

// LOG primitive (placeholder)
pub fn LOG_f(ref: Term) Term {
    _ = ref;
    print("TODO: LOG_f\n", .{});
    std.process.exit(1);
}

// =============================================================================
// Reduce (WHNF Evaluation) - Heavily optimized following VictorTaelin's IC
// =============================================================================

/// WHNF reduction with cached pointers and inlined hot paths
/// Based on optimizations from https://gist.github.com/VictorTaelin/4f55a8a07be9bd9f6d828227675fa9ac
pub noinline fn reduce(term: Term) Term {
    // Cache heap/stack pointers in locals (register allocation hint)
    const heap = HVM.heap;
    const stack = HVM.sbuf;
    const stop = HVM.spos;
    var spos = stop;
    var itrs = HVM.itrs;
    var next: Term = term;

    while (true) {
        // Handle substitution bit (variable was bound)
        if ((next & SUB_MASK) != 0) {
            @branchHint(.likely);
            const loc = (next >> LOC_SHIFT) & LOC_MASK;
            next = heap[loc];
            continue;
        }

        const tag: Tag = @truncate(next & TAG_MASK);
        const loc = (next >> LOC_SHIFT) & LOC_MASK;

        switch (tag) {
            VAR => {
                next = heap[loc];
                if ((next & SUB_MASK) != 0) {
                    @branchHint(.likely);
                    next = next & ~SUB_MASK;
                    continue;
                }
                if (spos > stop) {
                    spos -= 1;
                    const prev = stack[spos];
                    heap[(prev >> LOC_SHIFT) & LOC_MASK] = next;
                    next = prev;
                } else {
                    break;
                }
            },

            DP0, DP1 => {
                const val = heap[loc];
                if ((val & SUB_MASK) != 0) {
                    @branchHint(.likely);
                    // Inlined reduce_dup_sub
                    if (tag == DP0) {
                        next = val & ~SUB_MASK;
                    } else {
                        const stored = heap[loc];
                        if ((stored & SUB_MASK) != 0) {
                            next = stored & ~SUB_MASK;
                        } else {
                            heap[loc] = (val & ~SUB_MASK) | SUB_MASK;
                            next = val & ~SUB_MASK;
                        }
                    }
                    continue;
                }
                stack[spos] = next;
                spos += 1;
                next = val;
            },

            REF => {
                HVM.spos = spos;
                HVM.itrs = itrs;
                next = reduce_ref(next);
                spos = HVM.spos;
                itrs = HVM.itrs;
            },

            APP, MAT, IFL, SWI, OPX, OPY, LET => {
                stack[spos] = next;
                spos += 1;
                next = heap[loc];
            },

            LAM, SUP, CTR, ERA, W32, CHR => {
                if (spos <= stop) {
                    @branchHint(.unlikely);
                    break;
                }

                spos -= 1;
                const prev = stack[spos];
                const p_tag: Tag = @truncate(prev & TAG_MASK);
                const p_loc = (prev >> LOC_SHIFT) & LOC_MASK;

                heap[p_loc] = next;

                // APP-LAM: Most common interaction - fully inlined
                if (p_tag == APP and tag == LAM) {
                    @branchHint(.likely);
                    itrs += 1;
                    const arg = heap[p_loc + 1];
                    const bod = heap[loc];
                    heap[loc] = arg | SUB_MASK;
                    next = bod;
                    continue;
                }

                // DUP-SUP annihilation: Critical for optimal sharing - fully inlined
                if ((p_tag == DP0 or p_tag == DP1) and tag == SUP) {
                    const dup_lab: Lab = @truncate((prev >> LAB_SHIFT) & LAB_MASK);
                    const sup_lab: Lab = @truncate((next >> LAB_SHIFT) & LAB_MASK);

                    if (dup_lab == sup_lab) {
                        @branchHint(.likely);
                        itrs += 1;
                        const tm0 = heap[loc];
                        const tm1 = heap[loc + 1];
                        const dup_loc = (prev >> LOC_SHIFT) & LOC_MASK;

                        if (p_tag == DP0) {
                            heap[dup_loc] = tm1 | SUB_MASK;
                            next = tm0;
                        } else {
                            heap[dup_loc] = tm0 | SUB_MASK;
                            next = tm1;
                        }
                        continue;
                    }
                }

                // DUP-W32: Common case - inlined
                if ((p_tag == DP0 or p_tag == DP1) and (tag == W32 or tag == CHR)) {
                    @branchHint(.likely);
                    itrs += 1;
                    const dup_loc = (prev >> LOC_SHIFT) & LOC_MASK;
                    heap[dup_loc] = next | SUB_MASK;
                    continue;
                }

                // DUP-ERA: inlined
                if ((p_tag == DP0 or p_tag == DP1) and tag == ERA) {
                    itrs += 1;
                    const dup_loc = (prev >> LOC_SHIFT) & LOC_MASK;
                    heap[dup_loc] = next | SUB_MASK;
                    continue;
                }

                // APP-ERA: inlined
                if (p_tag == APP and tag == ERA) {
                    itrs += 1;
                    continue;
                }

                // OPX-W32: First operand ready, convert to OPY - inlined
                if (p_tag == OPX and (tag == W32 or tag == CHR)) {
                    @branchHint(.likely);
                    itrs += 1;
                    const opx_lab: Lab = @truncate((prev >> LAB_SHIFT) & LAB_MASK);
                    const nmy = heap[p_loc + 1];
                    heap[p_loc] = nmy;
                    heap[p_loc + 1] = next;
                    next = term_new(OPY, opx_lab, p_loc);
                    continue;
                }

                // OPY-W32: Both operands ready, compute result - inlined
                if (p_tag == OPY and (tag == W32 or tag == CHR)) {
                    @branchHint(.likely);
                    itrs += 1;
                    const op: Lab = @truncate((prev >> LAB_SHIFT) & LAB_MASK);
                    const t = tag;
                    const x: u32 = @truncate(heap[p_loc + 1] >> LOC_SHIFT);
                    const y: u32 = @truncate(loc);

                    const result: u32 = switch (op) {
                        OP_ADD => x +% y,
                        OP_SUB => x -% y,
                        OP_MUL => x *% y,
                        OP_DIV => if (y != 0) x / y else 0,
                        OP_MOD => if (y != 0) x % y else 0,
                        OP_EQ => @intFromBool(x == y),
                        OP_NE => @intFromBool(x != y),
                        OP_LT => @intFromBool(x < y),
                        OP_GT => @intFromBool(x > y),
                        OP_LTE => @intFromBool(x <= y),
                        OP_GTE => @intFromBool(x >= y),
                        OP_AND => x & y,
                        OP_OR => x | y,
                        OP_XOR => x ^ y,
                        OP_LSH => x << @truncate(y),
                        OP_RSH => x >> @truncate(y),
                        else => 0,
                    };
                    next = term_new(t, 0, result);
                    continue;
                }

                // Restore state for non-inlined interactions
                HVM.spos = spos;
                HVM.itrs = itrs;

                switch (p_tag) {
                    APP => {
                        switch (tag) {
                            SUP => next = reduce_app_sup(prev, next),
                            CTR => next = reduce_app_ctr(prev, next),
                            W32 => next = reduce_app_w32(prev, next),
                            else => {
                                HVM.spos = spos;
                                HVM.itrs = itrs;
                                return next;
                            },
                        }
                    },
                    DP0, DP1 => {
                        switch (tag) {
                            LAM => next = reduce_dup_lam(prev, next),
                            SUP => next = reduce_dup_sup(prev, next),
                            CTR => next = reduce_dup_ctr(prev, next),
                            else => {
                                HVM.spos = spos;
                                HVM.itrs = itrs;
                                return next;
                            },
                        }
                    },
                    MAT, IFL, SWI => {
                        switch (tag) {
                            ERA => next = reduce_mat_era(prev, next),
                            LAM => next = reduce_mat_lam(prev, next),
                            SUP => next = reduce_mat_sup(prev, next),
                            CTR => next = reduce_mat_ctr(prev, next),
                            W32, CHR => next = reduce_mat_w32(prev, next),
                            else => {
                                HVM.spos = spos;
                                HVM.itrs = itrs;
                                return next;
                            },
                        }
                    },
                    OPX => {
                        switch (tag) {
                            ERA => next = reduce_opx_era(prev, next),
                            LAM => next = reduce_opx_lam(prev, next),
                            SUP => next = reduce_opx_sup(prev, next),
                            CTR => next = reduce_opx_ctr(prev, next),
                            W32, CHR => next = reduce_opx_w32(prev, next),
                            else => {
                                HVM.spos = spos;
                                HVM.itrs = itrs;
                                return next;
                            },
                        }
                    },
                    OPY => {
                        switch (tag) {
                            ERA => next = reduce_opy_era(prev, next),
                            LAM => next = reduce_opy_lam(prev, next),
                            SUP => next = reduce_opy_sup(prev, next),
                            CTR => next = reduce_opy_ctr(prev, next),
                            W32, CHR => next = reduce_opy_w32(prev, next),
                            else => {
                                HVM.spos = spos;
                                HVM.itrs = itrs;
                                return next;
                            },
                        }
                    },
                    LET => {
                        next = reduce_let(prev, next);
                    },
                    else => {
                        HVM.spos = spos;
                        HVM.itrs = itrs;
                        return next;
                    },
                }

                spos = HVM.spos;
                itrs = HVM.itrs;
            },

            else => break,
        }
    }

    HVM.spos = spos;
    HVM.itrs = itrs;
    return next;
}

// Handle substituted dup nodes
fn reduce_dup_sub(dup: Term, val: Term) Term {
    const dup_loc = term_loc(dup);
    const dup_tag = term_tag(dup);

    if (dup_tag == DP0) {
        return val;
    } else {
        // DP1 - need to check if other projection was taken
        const stored = got(dup_loc);
        if (term_get_bit(stored) == 1) {
            return term_rem_bit(stored);
        } else {
            sub(dup_loc, val);
            return val;
        }
    }
}

// Reduce at a specific location
pub fn reduce_at(host: u64) Term {
    const term = got(host);
    const whnf = reduce(term);
    set(host, whnf);
    return whnf;
}

// =============================================================================
// Normal Form Evaluation
// =============================================================================

pub fn normal(term: Term) Term {
    const nf = reduce(term);
    const tag = term_tag(nf);
    const loc = term_loc(nf);

    switch (tag) {
        LAM => {
            const bod = normal(got(loc + 0));
            set(loc + 0, bod);
        },
        APP => {
            const fun = normal(got(loc + 0));
            const arg = normal(got(loc + 1));
            set(loc + 0, fun);
            set(loc + 1, arg);
        },
        SUP => {
            const tm0 = normal(got(loc + 0));
            const tm1 = normal(got(loc + 1));
            set(loc + 0, tm0);
            set(loc + 1, tm1);
        },
        DP0, DP1 => {
            const val = normal(got(loc + 0));
            set(loc + 0, val);
        },
        CTR => {
            const lab = term_lab(nf);
            const ari = HVM.cari[lab];
            var i: u64 = 0;
            while (i < ari) : (i += 1) {
                const arg = normal(got(loc + i));
                set(loc + i, arg);
            }
        },
        MAT, IFL, SWI => {
            // Should be reduced already
        },
        else => {},
    }

    return nf;
}

// =============================================================================
// Debugging / Printing
// =============================================================================

pub fn show_term(term: Term) void {
    const tag = term_tag(term);
    const lab = term_lab(term);
    const loc = term_loc(term);

    switch (tag) {
        DP0 => print("DP0({d},{d})", .{ lab, loc }),
        DP1 => print("DP1({d},{d})", .{ lab, loc }),
        VAR => print("VAR({d})", .{loc}),
        SUB => print("SUB({d})", .{loc}),
        REF => print("REF({d},{d})", .{ lab, loc }),
        LET => print("LET({d})", .{loc}),
        APP => print("APP({d})", .{loc}),
        ERA => print("*", .{}),
        MAT => print("MAT({d},{d})", .{ lab, loc }),
        IFL => print("IFL({d},{d})", .{ lab, loc }),
        SWI => print("SWI({d},{d})", .{ lab, loc }),
        OPX => print("OPX({d},{d})", .{ lab, loc }),
        OPY => print("OPY({d},{d})", .{ lab, loc }),
        LAM => print("LAM({d})", .{loc}),
        SUP => print("SUP({d},{d})", .{ lab, loc }),
        CTR => print("CTR({d},{d})", .{ lab, loc }),
        W32 => print("{d}", .{loc}),
        CHR => print("'{c}'", .{@as(u8, @truncate(loc))}),
        else => print("???({d},{d},{d})", .{ tag, lab, loc }),
    }
}

pub fn show_stats() void {
    print("Interactions: {d}\n", .{HVM.itrs});
    print("Heap size: {d}\n", .{HVM.size});
}
