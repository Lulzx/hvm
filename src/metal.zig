// =============================================================================
// HVM4 Metal GPU Acceleration - Pure Zig Implementation
// Actual Metal compute dispatch using Objective-C runtime
// =============================================================================
//
// GPU-RESIDENT REDUCTION ENGINE
// =============================
// This module implements a fully GPU-resident HVM reduction engine with:
// 1. GPU-resident heap - data stays on GPU across reduction steps
// 2. Full reduction loop - reduces to normal form without CPU round-trips
// 3. Heterogeneous batch interactions - single dispatch handles all interaction types
// 4. Async pipelining - triple buffering overlaps upload/compute/download
//
// =============================================================================

const std = @import("std");
const builtin = @import("builtin");

pub const is_supported = builtin.os.tag == .macos or builtin.os.tag == .ios;

// =============================================================================
// Objective-C Runtime Types and Functions
// =============================================================================

pub const id = ?*anyopaque;
pub const SEL = ?*anyopaque;
pub const Class = ?*anyopaque;
pub const NSUInteger = usize;
pub const NSInteger = isize;

// MTLSize struct (3 x NSUInteger)
pub const MTLSize = extern struct {
    width: NSUInteger,
    height: NSUInteger,
    depth: NSUInteger,
};

extern "objc" fn sel_registerName(name: [*:0]const u8) SEL;
extern "objc" fn objc_getClass(name: [*:0]const u8) Class;
extern "objc" fn objc_msgSend() void;
extern "objc" fn objc_msgSend_stret() void;

extern "Metal" fn MTLCreateSystemDefaultDevice() id;

// =============================================================================
// Message Send Helpers
// =============================================================================

// Return id
fn msg0(target: id, sel: SEL) id {
    const f: *const fn (id, SEL) callconv(.c) id = @ptrCast(&objc_msgSend);
    return f(target, sel);
}

fn msg1(target: id, sel: SEL, a1: anytype) id {
    const f: *const fn (id, SEL, @TypeOf(a1)) callconv(.c) id = @ptrCast(&objc_msgSend);
    return f(target, sel, a1);
}

fn msg2(target: id, sel: SEL, a1: anytype, a2: anytype) id {
    const f: *const fn (id, SEL, @TypeOf(a1), @TypeOf(a2)) callconv(.c) id = @ptrCast(&objc_msgSend);
    return f(target, sel, a1, a2);
}

fn msg3(target: id, sel: SEL, a1: anytype, a2: anytype, a3: anytype) id {
    const f: *const fn (id, SEL, @TypeOf(a1), @TypeOf(a2), @TypeOf(a3)) callconv(.c) id = @ptrCast(&objc_msgSend);
    return f(target, sel, a1, a2, a3);
}

// Return void
fn msgV0(target: id, sel: SEL) void {
    const f: *const fn (id, SEL) callconv(.c) void = @ptrCast(&objc_msgSend);
    f(target, sel);
}

fn msgV2(target: id, sel: SEL, a1: anytype, a2: anytype) void {
    const f: *const fn (id, SEL, @TypeOf(a1), @TypeOf(a2)) callconv(.c) void = @ptrCast(&objc_msgSend);
    f(target, sel, a1, a2);
}

fn msgV3(target: id, sel: SEL, a1: anytype, a2: anytype, a3: anytype) void {
    const f: *const fn (id, SEL, @TypeOf(a1), @TypeOf(a2), @TypeOf(a3)) callconv(.c) void = @ptrCast(&objc_msgSend);
    f(target, sel, a1, a2, a3);
}

// Return NSUInteger
fn msgU0(target: id, sel: SEL) NSUInteger {
    const f: *const fn (id, SEL) callconv(.c) NSUInteger = @ptrCast(&objc_msgSend);
    return f(target, sel);
}

// Dispatch threadgroups (takes two MTLSize by value)
fn msgDispatch(target: id, sel: SEL, tg: MTLSize, tpt: MTLSize) void {
    const f: *const fn (id, SEL, MTLSize, MTLSize) callconv(.c) void = @ptrCast(&objc_msgSend);
    f(target, sel, tg, tpt);
}

// =============================================================================
// Cached Selectors
// =============================================================================

const Selectors = struct {
    // Device
    newCommandQueue: SEL = null,
    newBufferWithLength_options: SEL = null,
    newBufferWithBytes_length_options: SEL = null,
    newLibraryWithSource_options_error: SEL = null,
    newFunctionWithName: SEL = null,
    newComputePipelineStateWithFunction_error: SEL = null,
    name: SEL = null,
    maxTotalThreadsPerThreadgroup: SEL = null,

    // Buffer
    contents: SEL = null,

    // Queue
    commandBuffer: SEL = null,

    // Command Buffer
    computeCommandEncoder: SEL = null,
    commit: SEL = null,
    waitUntilCompleted: SEL = null,

    // Compute Encoder
    setComputePipelineState: SEL = null,
    setBuffer_offset_atIndex: SEL = null,
    dispatchThreadgroups_threadsPerThreadgroup: SEL = null,
    endEncoding: SEL = null,

    // NSString
    stringWithUTF8String: SEL = null,
    UTF8String: SEL = null,
    localizedDescription: SEL = null,

    // Memory
    release: SEL = null,

    fn init() Selectors {
        return .{
            .newCommandQueue = sel_registerName("newCommandQueue"),
            .newBufferWithLength_options = sel_registerName("newBufferWithLength:options:"),
            .newBufferWithBytes_length_options = sel_registerName("newBufferWithBytes:length:options:"),
            .newLibraryWithSource_options_error = sel_registerName("newLibraryWithSource:options:error:"),
            .newFunctionWithName = sel_registerName("newFunctionWithName:"),
            .newComputePipelineStateWithFunction_error = sel_registerName("newComputePipelineStateWithFunction:error:"),
            .name = sel_registerName("name"),
            .maxTotalThreadsPerThreadgroup = sel_registerName("maxTotalThreadsPerThreadgroup"),
            .contents = sel_registerName("contents"),
            .commandBuffer = sel_registerName("commandBuffer"),
            .computeCommandEncoder = sel_registerName("computeCommandEncoder"),
            .commit = sel_registerName("commit"),
            .waitUntilCompleted = sel_registerName("waitUntilCompleted"),
            .setComputePipelineState = sel_registerName("setComputePipelineState:"),
            .setBuffer_offset_atIndex = sel_registerName("setBuffer:offset:atIndex:"),
            .dispatchThreadgroups_threadsPerThreadgroup = sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"),
            .endEncoding = sel_registerName("endEncoding"),
            .stringWithUTF8String = sel_registerName("stringWithUTF8String:"),
            .UTF8String = sel_registerName("UTF8String"),
            .localizedDescription = sel_registerName("localizedDescription"),
            .release = sel_registerName("release"),
        };
    }
};

var sels: Selectors = .{};

// =============================================================================
// Metal Shader Source (embedded)
// =============================================================================

const shader_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\// HVM4 term layout: [8-bit tag][24-bit ext][32-bit val]
    \\#define TAG_MASK  0xFF00000000000000ULL
    \\#define EXT_MASK  0x00FFFFFF00000000ULL
    \\#define VAL_MASK  0x00000000FFFFFFFFULL
    \\#define SUB_BIT   0x8000000000000000ULL
    \\#define TERM_CLR_SUB(t) ((t) & ~SUB_BIT)
    \\#define TERM_IS_SUB(t) ((t) & SUB_BIT)
    \\#define GET_TAG(t) ((TERM_CLR_SUB(t) >> 56) & 0xFF)
    \\#define GET_EXT(t) ((TERM_CLR_SUB(t) >> 32) & 0xFFFFFF)
    \\#define GET_VAL(t) ((t) & 0xFFFFFFFF)
    \\#define MAKE_TERM(tag, ext, val) (((ulong)(tag) << 56) | ((ulong)(ext) << 32) | (ulong)(val))
    \\
    \\// Term tags (HVM4)
    \\#define TAG_APP 0x00
    \\#define TAG_VAR 0x01
    \\#define TAG_LAM 0x02
    \\#define TAG_CO0 0x03
    \\#define TAG_CO1 0x04
    \\#define TAG_SUP 0x05
    \\#define TAG_DUP 0x06
    \\#define TAG_REF 0x07
    \\#define TAG_ERA 0x08
    \\#define TAG_RED 0x09
    \\#define TAG_LET 0x0A
    \\#define TAG_USE 0x0B
    \\#define TAG_EQL 0x0C
    \\#define TAG_C00 0x10
    \\#define TAG_C15 0x1F
    \\#define TAG_MAT 0x20
    \\#define TAG_SWI 0x21
    \\#define TAG_NUM 0x22
    \\#define TAG_P00 0x30
    \\#define TAG_P02 0x32
    \\#define TAG_P15 0x3F
    \\
    \\inline bool is_ctr_tag(uint tag) {
    \\    return tag >= TAG_C00 && tag <= TAG_C15;
    \\}
    \\
    \\inline bool is_value_tag(uint tag) {
    \\    return tag == TAG_LAM || tag == TAG_SUP || tag == TAG_ERA || tag == TAG_NUM || is_ctr_tag(tag);
    \\}
    \\
    \\inline bool is_redex_tag(uint tag) {
    \\    return tag == TAG_REF || tag == TAG_VAR || tag == TAG_APP || tag == TAG_CO0 || tag == TAG_CO1 ||
    \\        tag == TAG_MAT || tag == TAG_SWI || tag == TAG_LET || tag == TAG_USE || tag == TAG_RED ||
    \\        tag == TAG_P02 || tag == TAG_EQL;
    \\}
    \\
    \\inline void push_redex(
    \\    device uint2* out,
    \\    device atomic_uint* out_count,
    \\    device atomic_uint* stats,
    \\    uint loc_a,
    \\    uint loc_b
    \\) {
    \\    uint cap = atomic_load_explicit(&stats[3], memory_order_relaxed);
    \\    uint idx = atomic_fetch_add_explicit(out_count, 1, memory_order_relaxed);
    \\    if (idx < cap) {
    \\        out[idx] = uint2(loc_a, loc_b);
    \\    }
    \\}
    \\
    \\inline void enqueue_loc(
    \\    device ulong* heap,
    \\    device uint2* out,
    \\    device atomic_uint* out_count,
    \\    device atomic_uint* stats,
    \\    uint loc
    \\) {
    \\    ulong term = TERM_CLR_SUB(heap[loc]);
    \\    uint tag = GET_TAG(term);
    \\    if (!is_redex_tag(tag)) return;
    \\    uint val = GET_VAL(term);
    \\    uint loc_b = (tag == TAG_REF) ? 0xFFFFFFFFu : val;
    \\    push_redex(out, out_count, stats, loc, loc_b);
    \\}
    \\
    \\inline bool try_lock(device atomic_uint* locks, uint loc) {
    \\    return atomic_exchange_explicit(&locks[loc], 1, memory_order_relaxed) == 0;
    \\}
    \\
    \\inline void unlock(device atomic_uint* locks, uint loc) {
    \\    atomic_store_explicit(&locks[loc], 0, memory_order_relaxed);
    \\}
    \\
    \\inline bool tag_has_ptr(uint tag) {
    \\    if (tag == TAG_LAM || tag == TAG_APP || tag == TAG_SUP || tag == TAG_CO0 || tag == TAG_CO1) return true;
    \\    if (tag == TAG_VAR || tag == TAG_MAT || tag == TAG_SWI || tag == TAG_LET || tag == TAG_USE || tag == TAG_RED) return true;
    \\    if (tag == TAG_P02 || tag == TAG_EQL) return true;
    \\    if (tag >= TAG_C00 && tag <= TAG_C15 && tag != TAG_C00) return true;
    \\    return false;
    \\}
    \\
    \\inline ulong fixup_term(ulong term, uint base_delta) {
    \\    bool absolute = TERM_IS_SUB(term);
    \\    term = TERM_CLR_SUB(term);
    \\    uint tag = GET_TAG(term);
    \\    uint ext = GET_EXT(term);
    \\    uint val = GET_VAL(term);
    \\    if (tag_has_ptr(tag) && !absolute) {
    \\        val += base_delta;
    \\    }
    \\    return MAKE_TERM(tag, ext, val);
    \\}
    \\
    \\// Simple kernels
    \\kernel void batch_add(
    \\    device const uint* a [[buffer(0)]],
    \\    device const uint* b [[buffer(1)]],
    \\    device uint* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    result[id] = a[id] + b[id];
    \\}
    \\
    \\kernel void batch_mul(
    \\    device const uint* a [[buffer(0)]],
    \\    device const uint* b [[buffer(1)]],
    \\    device uint* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    result[id] = a[id] * b[id];
    \\}
    \\
    \\kernel void batch_add_vec4(
    \\    device const uint4* a [[buffer(0)]],
    \\    device const uint4* b [[buffer(1)]],
    \\    device uint4* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    result[id] = a[id] + b[id];
    \\}
    \\
    \\kernel void batch_mul_vec4(
    \\    device const uint4* a [[buffer(0)]],
    \\    device const uint4* b [[buffer(1)]],
    \\    device uint4* result [[buffer(2)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    result[id] = a[id] * b[id];
    \\}
    \\
    \\// ==========================================================================
    \\// HVM Interaction Rules - Complex GPU Kernels
    \\// ==========================================================================
    \\
    \\// Parallel beta reduction: APP-LAM interaction
    \\// Each thread processes one redex (application of lambda)
    \\kernel void interact_app_lam(
    \\    device const ulong* apps [[buffer(0)]],    // APP terms
    \\    device const ulong* lams [[buffer(1)]],    // LAM terms
    \\    device ulong* results [[buffer(2)]],       // Reduced results
    \\    device const ulong* heap [[buffer(3)]],    // Heap for lookups
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    ulong app = apps[id];
    \\    ulong lam = lams[id];
    \\
    \\    // Extract components
    \\    uint app_val = GET_VAL(app);  // Points to (func, arg) pair
    \\    uint lam_val = GET_VAL(lam);  // Points to body
    \\
    \\    // Simulate substitution: result is the lambda body
    \\    // In real HVM, this would involve variable substitution
    \\    ulong body = heap[lam_val];
    \\    ulong arg = heap[app_val + 1];
    \\
    \\    // Combine body tag with arg value (simplified interaction)
    \\    uint body_tag = GET_TAG(body);
    \\    uint arg_val = GET_VAL(arg);
    \\    results[id] = MAKE_TERM(body_tag, 0, arg_val);
    \\}
    \\
    \\// Parallel SUP-CO0 annihilation (same label)
    \\// Optimal sharing: when labels match, collapse without duplication
    \\kernel void interact_sup_co0(
    \\    device const ulong* sups [[buffer(0)]],    // SUP terms
    \\    device const ulong* co0s [[buffer(1)]],    // CO0 terms
    \\    device ulong* results [[buffer(2)]],       // First element of pair
    \\    device const ulong* heap [[buffer(3)]],    // Heap
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    ulong sup = sups[id];
    \\    ulong co0 = co0s[id];
    \\
    \\    uint sup_label = GET_EXT(sup);
    \\    uint co0_label = GET_EXT(co0);
    \\    uint sup_val = GET_VAL(sup);
    \\
    \\    if (sup_label == co0_label) {
    \\        // Annihilation: return first element
    \\        results[id] = heap[sup_val];
    \\    } else {
    \\        // Commutation: create new superposition (simplified)
    \\        results[id] = MAKE_TERM(TAG_SUP, co0_label, sup_val);
    \\    }
    \\}
    \\
    \\// Parallel DUP-LAM interaction (lambda duplication)
    \\// Creates two copies of lambda with fresh variables
    \\kernel void interact_dup_lam(
    \\    device const ulong* dups [[buffer(0)]],    // DUP terms
    \\    device const ulong* lams [[buffer(1)]],    // LAM terms
    \\    device ulong* result0 [[buffer(2)]],       // First copy
    \\    device ulong* result1 [[buffer(3)]],       // Second copy
    \\    device const ulong* heap [[buffer(4)]],    // Heap
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    ulong dup = dups[id];
    \\    ulong lam = lams[id];
    \\
    \\    uint dup_label = GET_EXT(dup);
    \\    uint lam_val = GET_VAL(lam);
    \\
    \\    // Create two lambda copies with different labels
    \\    ulong body = heap[lam_val];
    \\    result0[id] = MAKE_TERM(TAG_LAM, dup_label, GET_VAL(body));
    \\    result1[id] = MAKE_TERM(TAG_LAM, dup_label + 1, GET_VAL(body) + 1);
    \\}
    \\
    \\// Parallel numeric operations (P02-NUM interaction)
    \\// Applies binary operation to two numbers
    \\kernel void interact_op2_num(
    \\    device const ulong* ops [[buffer(0)]],     // Operation terms (ext = opcode)
    \\    device const ulong* nums [[buffer(1)]],    // NUM terms
    \\    device ulong* results [[buffer(2)]],       // Result NUMs
    \\    device const ulong* heap [[buffer(3)]],    // Heap for second operand
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    ulong op = ops[id];
    \\    ulong num = nums[id];
    \\
    \\    uint opcode = GET_EXT(op);
    \\    uint op_val = GET_VAL(op);
    \\    uint a = GET_VAL(num);
    \\    uint b = GET_VAL(heap[op_val]);  // Second operand from heap
    \\
    \\    uint result;
    \\    switch (opcode) {
    \\        case 0: result = a + b; break;  // ADD
    \\        case 1: result = a - b; break;  // SUB
    \\        case 2: result = a * b; break;  // MUL
    \\        case 3: result = b != 0 ? a / b : 0; break;  // DIV
    \\        case 4: result = b != 0 ? a % b : 0; break;  // MOD
    \\        case 5: result = a & b; break;  // AND
    \\        case 6: result = a | b; break;  // OR
    \\        case 7: result = a ^ b; break;  // XOR
    \\        case 8: result = a << (b & 31); break;  // LSH
    \\        case 9: result = a >> (b & 31); break;  // RSH
    \\        case 10: result = a == b ? 1 : 0; break; // EQ
    \\        case 11: result = a != b ? 1 : 0; break; // NE
    \\        case 12: result = a < b ? 1 : 0; break;  // LT
    \\        case 13: result = a <= b ? 1 : 0; break; // LE
    \\        case 14: result = a > b ? 1 : 0; break;  // GT
    \\        case 15: result = a >= b ? 1 : 0; break; // GE
    \\        default: result = 0; break;
    \\    }
    \\    results[id] = MAKE_TERM(TAG_NUM, 0, result);
    \\}
    \\
    \\// Parallel pattern matching (MAT-CTR interaction)
    \\// Selects branch based on constructor tag
    \\kernel void interact_mat_ctr(
    \\    device const ulong* mats [[buffer(0)]],    // MAT terms
    \\    device const ulong* ctrs [[buffer(1)]],    // Constructor terms
    \\    device ulong* results [[buffer(2)]],       // Selected branches
    \\    device const ulong* branches [[buffer(3)]], // Branch table
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    ulong mat = mats[id];
    \\    ulong ctr = ctrs[id];
    \\
    \\    uint mat_val = GET_VAL(mat);    // Points to branch table
    \\    uint ctr_tag = GET_TAG(ctr);    // Constructor index (C00-C15)
    \\    uint ctr_idx = ctr_tag & 0x0F;  // Extract 0-15
    \\
    \\    // Select branch and apply constructor's value
    \\    ulong branch = branches[mat_val + ctr_idx];
    \\    uint ctr_val = GET_VAL(ctr);
    \\    results[id] = MAKE_TERM(GET_TAG(branch), GET_EXT(branch), ctr_val);
    \\}
    \\
    \\// ==========================================================================
    \\// Parallel Reduction Kernels
    \\// ==========================================================================
    \\
    \\// Parallel sum reduction (for benchmarking reduction patterns)
    \\kernel void reduce_sum(
    \\    device const uint* input [[buffer(0)]],
    \\    device uint* output [[buffer(1)]],
    \\    threadgroup uint* shared [[threadgroup(0)]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint gid [[thread_position_in_grid]],
    \\    uint blockDim [[threads_per_threadgroup]]
    \\) {
    \\    // Load to shared memory
    \\    shared[tid] = input[gid];
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Parallel reduction in shared memory
    \\    for (uint s = blockDim / 2; s > 0; s >>= 1) {
    \\        if (tid < s) {
    \\            shared[tid] += shared[tid + s];
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\
    \\    // Write result
    \\    if (tid == 0) {
    \\        output[gid / blockDim] = shared[0];
    \\    }
    \\}
    \\
    \\// Interaction net step: process all active pairs in parallel
    \\// This is the core HVM reduction step
    \\kernel void interact_step(
    \\    device ulong* heap [[buffer(0)]],          // Mutable heap
    \\    device const uint* active_pairs [[buffer(1)]], // Pairs of indices to interact
    \\    device uint* interactions [[buffer(2)]],   // Interaction count output
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    uint pair_idx = id * 2;
    \\    uint idx_a = active_pairs[pair_idx];
    \\    uint idx_b = active_pairs[pair_idx + 1];
    \\
    \\    ulong term_a = heap[idx_a];
    \\    ulong term_b = heap[idx_b];
    \\
    \\    uint tag_a = GET_TAG(term_a);
    \\    uint tag_b = GET_TAG(term_b);
    \\
    \\    // Dispatch based on tag combination
    \\    ulong result = 0;
    \\    uint did_interact = 0;
    \\
    \\    if (tag_a == TAG_APP && tag_b == TAG_LAM) {
    \\        // Beta reduction
    \\        uint body_idx = GET_VAL(term_b);
    \\        result = heap[body_idx];
    \\        did_interact = 1;
    \\    } else if (tag_a == TAG_CO0 && tag_b == TAG_SUP) {
    \\        uint label_a = GET_EXT(term_a);
    \\        uint label_b = GET_EXT(term_b);
    \\        if (label_a == label_b) {
    \\            // Annihilation
    \\            result = heap[GET_VAL(term_b)];
    \\            did_interact = 1;
    \\        }
    \\    } else if (tag_a == TAG_ERA) {
    \\        // Erasure
    \\        result = MAKE_TERM(TAG_ERA, 0, 0);
    \\        did_interact = 1;
    \\    }
    \\
    \\    if (did_interact) {
    \\        heap[idx_a] = result;
    \\        heap[idx_b] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        atomic_fetch_add_explicit((device atomic_uint*)interactions, 1, memory_order_relaxed);
    \\    }
    \\}
    \\
    \\// ==========================================================================
    \\// GPU-RESIDENT REDUCTION ENGINE
    \\// ==========================================================================
    \\
    \\// Unified heterogeneous interaction kernel
    \\// Processes mixed redex types in a single dispatch - no sorting needed
    \\kernel void interact_heterogeneous(
    \\    device ulong* heap [[buffer(0)]],           // GPU-resident heap (read/write)
    \\    device const uint2* redexes [[buffer(1)]],  // Redex pairs (loc_a, loc_b)
    \\    device atomic_uint* alloc_ptr [[buffer(2)]],// Atomic heap allocation pointer
    \\    device uint2* redex_out [[buffer(3)]],      // Output redex queue (new redexes)
    \\    device atomic_uint* out_count [[buffer(4)]], // Number of output redexes
    \\    device atomic_uint* stats [[buffer(5)]],    // [0]=interactions, [1]=allocations
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    uint2 redex = redexes[id];
    \\    uint loc_a = redex.x;
    \\    uint loc_b = redex.y;
    \\
    \\    ulong term_a = heap[loc_a];
    \\    ulong term_b = heap[loc_b];
    \\
    \\    uint tag_a = GET_TAG(term_a);
    \\    uint tag_b = GET_TAG(term_b);
    \\    uint ext_a = GET_EXT(term_a);
    \\    uint ext_b = GET_EXT(term_b);
    \\    uint val_a = GET_VAL(term_a);
    \\    uint val_b = GET_VAL(term_b);
    \\
    \\    // APP-LAM: Beta reduction
    \\    if (tag_a == TAG_APP && tag_b == TAG_LAM) {
    \\        ulong body = heap[val_b];
    \\        ulong arg = heap[val_a + 1];
    \\        heap[loc_a] = body;
    \\        heap[loc_b] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        if (GET_TAG(body) == TAG_VAR && GET_VAL(body) == val_b) {
    \\            heap[loc_a] = arg;
    \\        }
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        return;
    \\    }
    \\
    \\    // CO0-SUP or CO1-SUP: Collapse/Commutation
    \\    if ((tag_a == TAG_CO0 || tag_a == TAG_CO1) && tag_b == TAG_SUP) {
    \\        if (ext_a == ext_b) {
    \\            uint offset = (tag_a == TAG_CO0) ? 0 : 1;
    \\            heap[loc_a] = heap[val_b + offset];
    \\            heap[loc_b] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        } else {
    \\            uint new_loc = atomic_fetch_add_explicit(alloc_ptr, 4, memory_order_relaxed);
    \\            atomic_fetch_add_explicit(&stats[1], 4, memory_order_relaxed);
    \\            ulong fst = heap[val_b];
    \\            ulong snd = heap[val_b + 1];
    \\            heap[new_loc] = MAKE_TERM(tag_a, ext_a, GET_VAL(fst));
    \\            heap[new_loc + 1] = MAKE_TERM(tag_a, ext_a, GET_VAL(snd));
    \\            heap[new_loc + 2] = MAKE_TERM(tag_a == TAG_CO0 ? TAG_CO1 : TAG_CO0, ext_a, GET_VAL(fst));
    \\            heap[new_loc + 3] = MAKE_TERM(tag_a == TAG_CO0 ? TAG_CO1 : TAG_CO0, ext_a, GET_VAL(snd));
    \\            heap[loc_a] = MAKE_TERM(TAG_SUP, ext_b, new_loc);
    \\            heap[loc_b] = MAKE_TERM(TAG_ERA, 0, 0);
    \\            uint out_idx = atomic_fetch_add_explicit(out_count, 2, memory_order_relaxed);
    \\            redex_out[out_idx] = uint2(new_loc, GET_VAL(fst));
    \\            redex_out[out_idx + 1] = uint2(new_loc + 2, GET_VAL(snd));
    \\        }
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        return;
    \\    }
    \\
    \\    // Numeric operations (P02-NUM)
    \\    if (tag_a == TAG_P02 && tag_b == TAG_NUM) {
    \\        uint opcode = ext_a;
    \\        uint a = val_b;
    \\        uint b = GET_VAL(heap[val_a]);
    \\        uint result;
    \\        switch (opcode) {
    \\            case 0: result = a + b; break;
    \\            case 1: result = a - b; break;
    \\            case 2: result = a * b; break;
    \\            case 3: result = b != 0 ? a / b : 0; break;
    \\            case 4: result = b != 0 ? a % b : 0; break;
    \\            case 5: result = a & b; break;
    \\            case 6: result = a | b; break;
    \\            case 7: result = a ^ b; break;
    \\            case 8: result = a << (b & 31); break;
    \\            case 9: result = a >> (b & 31); break;
    \\            case 10: result = a == b ? 1 : 0; break;
    \\            case 11: result = a != b ? 1 : 0; break;
    \\            case 12: result = a < b ? 1 : 0; break;
    \\            case 13: result = a <= b ? 1 : 0; break;
    \\            case 14: result = a > b ? 1 : 0; break;
    \\            case 15: result = a >= b ? 1 : 0; break;
    \\            default: result = 0; break;
    \\        }
    \\        heap[loc_a] = MAKE_TERM(TAG_NUM, 0, result);
    \\        heap[loc_b] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        return;
    \\    }
    \\
    \\    // ERA-ANY: Erasure propagates
    \\    if (tag_a == TAG_ERA || tag_b == TAG_ERA) {
    \\        heap[loc_a] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        heap[loc_b] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        return;
    \\    }
    \\}
    \\
    \\// GPU-resident reduction loop - runs N steps without CPU round-trip
    \\kernel void reduce_step(
    \\    device ulong* heap [[buffer(0)]],
    \\    device uint2* redex_in [[buffer(1)]],
    \\    device uint2* redex_out [[buffer(2)]],
    \\    device atomic_uint* alloc_ptr [[buffer(3)]],
    \\    device atomic_uint* in_count [[buffer(4)]],
    \\    device atomic_uint* out_count [[buffer(5)]],
    \\    device atomic_uint* stats [[buffer(6)]],
    \\    device const ulong* tmpl_terms [[buffer(7)]],
    \\    device const uint* tmpl_info [[buffer(8)]],
    \\    device atomic_uint* locks [[buffer(9)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    uint count = atomic_load_explicit(in_count, memory_order_relaxed);
    \\    if (id >= count) return;
    \\
    \\    uint2 redex = redex_in[id];
    \\    uint loc_a = redex.x;
    \\    if (!try_lock(locks, loc_a)) {
    \\        push_redex(redex_out, out_count, stats, loc_a, 0xFFFFFFFFu);
    \\        return;
    \\    }
    \\
    \\    ulong term_a = heap[loc_a];
    \\    uint tag_a = GET_TAG(term_a);
    \\    uint ext_a = GET_EXT(term_a);
    \\    uint val_a = GET_VAL(term_a);
    \\
    \\    if (!is_redex_tag(tag_a)) {
    \\        unlock(locks, loc_a);
    \\        return;
    \\    }
    \\
    \\    uint loc_b = 0xFFFFFFFFu;
    \\    bool locked_b = false;
    \\    if (tag_a != TAG_REF) {
    \\        loc_b = val_a;
    \\        if (loc_b != loc_a) {
    \\            locked_b = try_lock(locks, loc_b);
    \\            if (!locked_b) {
    \\                unlock(locks, loc_a);
    \\                push_redex(redex_out, out_count, stats, loc_a, 0xFFFFFFFFu);
    \\                return;
    \\            }
    \\        }
    \\    }
    \\
    \\
    \\#define REDUCE_STEP_RETURN() { if (locked_b) { unlock(locks, loc_b); } unlock(locks, loc_a); return; }
    \\    ulong term_b = 0;
    \\    uint tag_b = 0xFF;
    \\    uint ext_b = 0;
    \\    uint val_b = 0;
    \\    if (loc_b != 0xFFFFFFFFu) {
    \\        term_b = heap[loc_b];
    \\        tag_b = GET_TAG(term_b);
    \\        ext_b = GET_EXT(term_b);
    \\        val_b = GET_VAL(term_b);
    \\    }
    \\
    \\    // REF expansion (template instantiation)
    \\    if (tag_a == TAG_REF) {
    \\        uint info_idx = ext_a * 3;
    \\        uint tmpl_off = tmpl_info[info_idx];
    \\        uint tmpl_len = tmpl_info[info_idx + 1];
    \\        uint tmpl_root = tmpl_info[info_idx + 2];
    \\        if (tmpl_root == 0xFFFFFFFFu) {
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        if (tmpl_len == 0) {
    \\            ulong root_term = tmpl_terms[tmpl_off + tmpl_root];
    \\            heap[loc_a] = fixup_term(root_term, 0);
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        uint base = atomic_fetch_add_explicit(alloc_ptr, tmpl_len, memory_order_relaxed);
    \\        atomic_fetch_add_explicit(&stats[1], tmpl_len, memory_order_relaxed);
    \\        uint base_delta = base - tmpl_off;
    \\        for (uint i = 0; i < tmpl_len; i++) {
    \\            ulong t = tmpl_terms[tmpl_off + i];
    \\            heap[base + i] = fixup_term(t, base_delta);
    \\        }
    \\        ulong root_term = tmpl_terms[tmpl_off + tmpl_root];
    \\        heap[loc_a] = fixup_term(root_term, base_delta);
    \\        enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // VAR substitution / path compression
    \\    if (tag_a == TAG_VAR) {
    \\        ulong bound = heap[val_a];
    \\        ulong clean = TERM_CLR_SUB(bound);
    \\        uint b_tag = GET_TAG(clean);
    \\        if (b_tag == TAG_VAR) {
    \\            heap[loc_a] = clean;
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        } else if (is_value_tag(b_tag)) {
    \\            heap[val_a] = clean | SUB_BIT;
    \\            heap[loc_a] = clean;
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        } else {
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\            push_redex(redex_out, out_count, stats, loc_a, val_a);
    \\        }
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // APP interactions
    \\    if (tag_a == TAG_APP) {
    \\        if (tag_b == TAG_LAM) {
    \\            uint lam_loc = val_b;
    \\            ulong arg = heap[val_a + 1];
    \\            ulong body = heap[lam_loc];
    \\            heap[lam_loc] = arg | SUB_BIT;
    \\            heap[loc_a] = body;
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        } else if (tag_b == TAG_ERA) {
    \\            heap[loc_a] = MAKE_TERM(TAG_ERA, 0, 0);
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        } else if (tag_b == TAG_SUP) {
    \\            uint sup_loc = val_b;
    \\            uint sup_lab = ext_b;
    \\            ulong arg = heap[val_a + 1];
    \\            ulong lft = heap[sup_loc];
    \\            ulong rgt = heap[sup_loc + 1];
    \\            uint base = atomic_fetch_add_explicit(alloc_ptr, 7, memory_order_relaxed);
    \\            atomic_fetch_add_explicit(&stats[1], 7, memory_order_relaxed);
    \\            heap[base] = arg;
    \\            heap[base + 1] = lft;
    \\            heap[base + 2] = MAKE_TERM(TAG_CO0, sup_lab, base);
    \\            heap[base + 3] = rgt;
    \\            heap[base + 4] = MAKE_TERM(TAG_CO1, sup_lab, base);
    \\            heap[base + 5] = MAKE_TERM(TAG_APP, 0, base + 1);
    \\            heap[base + 6] = MAKE_TERM(TAG_APP, 0, base + 3);
    \\            heap[loc_a] = MAKE_TERM(TAG_SUP, sup_lab, base + 5);
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\        push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // CO0/CO1 interactions
    \\    if (tag_a == TAG_CO0 || tag_a == TAG_CO1) {
    \\        uint dup_loc = val_a;
    \\        ulong target = heap[dup_loc];
    \\        if (TERM_IS_SUB(target)) {
    \\            heap[loc_a] = TERM_CLR_SUB(target);
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        uint t_tag = GET_TAG(target);
    \\        if (t_tag == TAG_SUP) {
    \\            uint sup_loc = GET_VAL(target);
    \\            uint sup_lab = GET_EXT(target);
    \\            uint dup_lab = ext_a;
    \\            if (dup_lab == sup_lab) {
    \\                ulong lft = heap[sup_loc];
    \\                ulong rgt = heap[sup_loc + 1];
    \\                if (tag_a == TAG_CO0) {
    \\                    heap[dup_loc] = TERM_CLR_SUB(rgt) | SUB_BIT;
    \\                    heap[loc_a] = TERM_CLR_SUB(lft);
    \\                } else {
    \\                    heap[dup_loc] = TERM_CLR_SUB(lft) | SUB_BIT;
    \\                    heap[loc_a] = TERM_CLR_SUB(rgt);
    \\                }
    \\            } else {
    \\                uint base = atomic_fetch_add_explicit(alloc_ptr, 4, memory_order_relaxed);
    \\                atomic_fetch_add_explicit(&stats[1], 4, memory_order_relaxed);
    \\                heap[base] = MAKE_TERM(TAG_CO0, dup_lab, sup_loc);
    \\                heap[base + 1] = MAKE_TERM(TAG_CO0, dup_lab, sup_loc + 1);
    \\                heap[base + 2] = MAKE_TERM(TAG_CO1, dup_lab, sup_loc);
    \\                heap[base + 3] = MAKE_TERM(TAG_CO1, dup_lab, sup_loc + 1);
    \\                if (tag_a == TAG_CO0) {
    \\                    heap[dup_loc] = MAKE_TERM(TAG_SUP, sup_lab, base + 2) | SUB_BIT;
    \\                    heap[loc_a] = MAKE_TERM(TAG_SUP, sup_lab, base);
    \\                } else {
    \\                    heap[dup_loc] = MAKE_TERM(TAG_SUP, sup_lab, base) | SUB_BIT;
    \\                    heap[loc_a] = MAKE_TERM(TAG_SUP, sup_lab, base + 2);
    \\                }
    \\            }
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        if (t_tag == TAG_NUM || t_tag == TAG_ERA) {
    \\            ulong clean = TERM_CLR_SUB(target);
    \\            heap[dup_loc] = clean | SUB_BIT;
    \\            heap[loc_a] = clean;
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        if (t_tag == TAG_LAM) {
    \\            uint dup_lab = ext_a;
    \\            uint lam_loc = GET_VAL(target);
    \\            ulong bod = heap[lam_loc];
    \\            uint base = atomic_fetch_add_explicit(alloc_ptr, 3, memory_order_relaxed);
    \\            atomic_fetch_add_explicit(&stats[1], 3, memory_order_relaxed);
    \\            heap[base] = bod;
    \\            heap[base + 1] = MAKE_TERM(TAG_CO0, dup_lab, base);
    \\            heap[base + 2] = MAKE_TERM(TAG_CO1, dup_lab, base);
    \\            if (tag_a == TAG_CO0) {
    \\                heap[dup_loc] = MAKE_TERM(TAG_LAM, 0, base + 2) | SUB_BIT;
    \\                heap[loc_a] = MAKE_TERM(TAG_LAM, 0, base + 1);
    \\            } else {
    \\                heap[dup_loc] = MAKE_TERM(TAG_LAM, 0, base + 1) | SUB_BIT;
    \\                heap[loc_a] = MAKE_TERM(TAG_LAM, 0, base + 2);
    \\            }
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        if (t_tag == TAG_APP) {
    \\            uint dup_lab = ext_a;
    \\            uint app_loc = GET_VAL(target);
    \\            uint base = atomic_fetch_add_explicit(alloc_ptr, 6, memory_order_relaxed);
    \\            atomic_fetch_add_explicit(&stats[1], 6, memory_order_relaxed);
    \\            heap[base] = heap[app_loc];
    \\            heap[base + 1] = heap[app_loc + 1];
    \\            heap[base + 2] = MAKE_TERM(TAG_CO0, dup_lab, base);
    \\            heap[base + 3] = MAKE_TERM(TAG_CO0, dup_lab, base + 1);
    \\            heap[base + 4] = MAKE_TERM(TAG_CO1, dup_lab, base);
    \\            heap[base + 5] = MAKE_TERM(TAG_CO1, dup_lab, base + 1);
    \\            if (tag_a == TAG_CO0) {
    \\                heap[dup_loc] = MAKE_TERM(TAG_APP, 0, base + 4) | SUB_BIT;
    \\                heap[loc_a] = MAKE_TERM(TAG_APP, 0, base + 2);
    \\            } else {
    \\                heap[dup_loc] = MAKE_TERM(TAG_APP, 0, base + 2) | SUB_BIT;
    \\                heap[loc_a] = MAKE_TERM(TAG_APP, 0, base + 4);
    \\            }
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        if (is_ctr_tag(t_tag)) {
    \\            uint arity = t_tag - TAG_C00;
    \\            if (arity == 0) {
    \\                ulong clean = TERM_CLR_SUB(target);
    \\                heap[dup_loc] = clean | SUB_BIT;
    \\                heap[loc_a] = clean;
    \\            } else {
    \\                uint dup_lab = ext_a;
    \\                uint ctr_loc = GET_VAL(target);
    \\                uint base = atomic_fetch_add_explicit(alloc_ptr, 3 * arity, memory_order_relaxed);
    \\                atomic_fetch_add_explicit(&stats[1], 3 * arity, memory_order_relaxed);
    \\                for (uint i = 0; i < arity; i++) {
    \\                    heap[base + i] = heap[ctr_loc + i];
    \\                }
    \\                uint ctr0 = base + arity;
    \\                uint ctr1 = base + arity + arity;
    \\                for (uint i = 0; i < arity; i++) {
    \\                    heap[ctr0 + i] = MAKE_TERM(TAG_CO0, dup_lab, base + i);
    \\                    heap[ctr1 + i] = MAKE_TERM(TAG_CO1, dup_lab, base + i);
    \\                }
    \\                if (tag_a == TAG_CO0) {
    \\                    heap[dup_loc] = MAKE_TERM(t_tag, ext_b, ctr1) | SUB_BIT;
    \\                    heap[loc_a] = MAKE_TERM(t_tag, ext_b, ctr0);
    \\                } else {
    \\                    heap[dup_loc] = MAKE_TERM(t_tag, ext_b, ctr0) | SUB_BIT;
    \\                    heap[loc_a] = MAKE_TERM(t_tag, ext_b, ctr1);
    \\                }
    \\            }
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        enqueue_loc(heap, redex_out, out_count, stats, dup_loc);
    \\        push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // MAT + CTR
    \\    if (tag_a == TAG_MAT) {
    \\        if (is_ctr_tag(tag_b)) {
    \\            uint ctr_loc = val_b;
    \\            uint case_idx = ext_b;
    \\            uint arity = tag_b - TAG_C00;
    \\            ulong branch = heap[val_a + 1 + case_idx];
    \\            for (uint i = 0; i < arity; i++) {
    \\                uint app_loc = atomic_fetch_add_explicit(alloc_ptr, 2, memory_order_relaxed);
    \\                atomic_fetch_add_explicit(&stats[1], 2, memory_order_relaxed);
    \\                heap[app_loc] = branch;
    \\                heap[app_loc + 1] = heap[ctr_loc + i];
    \\                branch = MAKE_TERM(TAG_APP, 0, app_loc);
    \\            }
    \\            heap[loc_a] = branch;
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\        push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // SWI + NUM
    \\    if (tag_a == TAG_SWI) {
    \\        if (tag_b == TAG_NUM) {
    \\            uint num_val = val_b;
    \\            if (num_val == 0) {
    \\                heap[loc_a] = heap[val_a + 1];
    \\            } else {
    \\                uint app_loc = atomic_fetch_add_explicit(alloc_ptr, 2, memory_order_relaxed);
    \\                atomic_fetch_add_explicit(&stats[1], 2, memory_order_relaxed);
    \\                heap[app_loc] = heap[val_a + 2];
    \\                heap[app_loc + 1] = MAKE_TERM(TAG_NUM, 0, num_val - 1);
    \\                heap[loc_a] = MAKE_TERM(TAG_APP, 0, app_loc);
    \\            }
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\        push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // LET/USE/RED
    \\    if (tag_a == TAG_LET) {
    \\        ulong bound = heap[val_a];
    \\        if (is_value_tag(GET_TAG(bound))) {
    \\            ulong clean = TERM_CLR_SUB(bound);
    \\            heap[val_a] = clean | SUB_BIT;
    \\            heap[loc_a] = heap[val_a + 1];
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        } else {
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\            push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        }
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\    if (tag_a == TAG_USE) {
    \\        ulong bound = heap[val_a];
    \\        if (is_value_tag(GET_TAG(bound))) {
    \\            ulong clean = TERM_CLR_SUB(bound);
    \\            heap[val_a] = clean;
    \\            heap[loc_a] = clean;
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        } else {
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\            push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        }
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\    if (tag_a == TAG_RED) {
    \\        ulong bound = heap[val_a];
    \\        if (is_value_tag(GET_TAG(bound))) {
    \\            heap[loc_a] = TERM_CLR_SUB(bound);
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        } else {
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\            push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        }
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // P02 numeric ops
    \\    if (tag_a == TAG_P02) {
    \\        ulong left = heap[val_a];
    \\        ulong right = heap[val_a + 1];
    \\        if (GET_TAG(left) == TAG_NUM && GET_TAG(right) == TAG_NUM) {
    \\            uint opcode = ext_a;
    \\            uint a = GET_VAL(left);
    \\            uint b = GET_VAL(right);
    \\            uint result;
    \\            switch (opcode) {
    \\                case 0: result = a + b; break;
    \\                case 1: result = a - b; break;
    \\                case 2: result = a * b; break;
    \\                case 3: result = b != 0 ? a / b : 0; break;
    \\                case 4: result = b != 0 ? a % b : 0; break;
    \\                case 5: result = a & b; break;
    \\                case 6: result = a | b; break;
    \\                case 7: result = a ^ b; break;
    \\                case 8: result = a << (b & 31); break;
    \\                case 9: result = a >> (b & 31); break;
    \\                case 10: result = a == b ? 1 : 0; break;
    \\                case 11: result = a != b ? 1 : 0; break;
    \\                case 12: result = a < b ? 1 : 0; break;
    \\                case 13: result = a <= b ? 1 : 0; break;
    \\                case 14: result = a > b ? 1 : 0; break;
    \\                case 15: result = a >= b ? 1 : 0; break;
    \\                default: result = 0; break;
    \\            }
    \\            heap[loc_a] = MAKE_TERM(TAG_NUM, 0, result);
    \\            enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\            atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        } else {
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a + 1);
    \\            push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\        }
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // EQL (basic)
    \\    if (tag_a == TAG_EQL) {
    \\        ulong left = heap[val_a];
    \\        ulong right = heap[val_a + 1];
    \\        uint lt = GET_TAG(left);
    \\        uint rt = GET_TAG(right);
    \\        if (!is_value_tag(lt) || !is_value_tag(rt)) {
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a);
    \\            enqueue_loc(heap, redex_out, out_count, stats, val_a + 1);
    \\            push_redex(redex_out, out_count, stats, loc_a, loc_b);
    \\            REDUCE_STEP_RETURN();
    \\        }
    \\        uint eq = 0;
    \\        if (lt == rt) {
    \\            if (lt == TAG_NUM) {
    \\                eq = (GET_VAL(left) == GET_VAL(right)) ? 1 : 0;
    \\            } else if (lt == TAG_ERA) {
    \\                eq = 1;
    \\            } else if (is_ctr_tag(lt) && lt == TAG_C00 && rt == TAG_C00) {
    \\                eq = 1;
    \\            }
    \\        }
    \\        heap[loc_a] = MAKE_TERM(TAG_NUM, 0, eq);
    \\        enqueue_loc(heap, redex_out, out_count, stats, loc_a);
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        REDUCE_STEP_RETURN();
    \\    }
    \\
    \\    // Erasure fallback
    \\    if (tag_a == TAG_ERA || tag_b == TAG_ERA) {
    \\        heap[loc_a] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        if (loc_b != 0) {
    \\            heap[loc_b] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        }
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\    }
    \\    REDUCE_STEP_RETURN();
    \\}
    \\#undef REDUCE_STEP_RETURN
    \\
    \\// Scan heap for active redexes
    \\kernel void scan_redexes(
    \\    device const ulong* heap [[buffer(0)]],
    \\    device uint2* redexes [[buffer(1)]],
    \\    device atomic_uint* redex_count [[buffer(2)]],
    \\    constant uint& heap_size [[buffer(3)]],
    \\    constant uint& redex_cap [[buffer(4)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    if (id >= heap_size) return;
    \\    ulong term = heap[id];
    \\    uint tag = GET_TAG(term);
    \\    uint val = GET_VAL(term);
    \\    bool is_active = false;
    \\    uint loc_b = val;
    \\
    \\    if (tag == TAG_REF) {
    \\        is_active = true;
    \\        loc_b = 0xFFFFFFFFu;
    \\    } else if (tag == TAG_VAR) {
    \\        ulong bound = heap[val];
    \\        ulong clean = TERM_CLR_SUB(bound);
    \\        uint b_tag = GET_TAG(clean);
    \\        if (b_tag == TAG_VAR || is_value_tag(b_tag)) is_active = true;
    \\    } else if (tag == TAG_APP) {
    \\        ulong func = heap[val];
    \\        uint f_tag = GET_TAG(func);
    \\        if (f_tag == TAG_LAM || f_tag == TAG_SUP || f_tag == TAG_ERA) is_active = true;
    \\    } else if (tag == TAG_CO0 || tag == TAG_CO1) {
    \\        ulong target = heap[val];
    \\        uint t_tag = GET_TAG(target);
    \\        if (t_tag == TAG_SUP || t_tag == TAG_NUM || t_tag == TAG_ERA || t_tag == TAG_LAM || t_tag == TAG_APP || is_ctr_tag(t_tag)) is_active = true;
    \\    } else if (tag == TAG_MAT) {
    \\        ulong scrut = heap[val];
    \\        if (is_ctr_tag(GET_TAG(scrut))) is_active = true;
    \\    } else if (tag == TAG_SWI) {
    \\        ulong scrut = heap[val];
    \\        if (GET_TAG(scrut) == TAG_NUM) is_active = true;
    \\    } else if (tag == TAG_LET || tag == TAG_USE || tag == TAG_RED) {
    \\        ulong bound = heap[val];
    \\        if (is_value_tag(GET_TAG(bound))) is_active = true;
    \\    } else if (tag == TAG_P02) {
    \\        ulong left = heap[val];
    \\        ulong right = heap[val + 1];
    \\        if (GET_TAG(left) == TAG_NUM && GET_TAG(right) == TAG_NUM) is_active = true;
    \\    } else if (tag == TAG_EQL) {
    \\        ulong left = heap[val];
    \\        ulong right = heap[val + 1];
    \\        if (is_value_tag(GET_TAG(left)) && is_value_tag(GET_TAG(right))) is_active = true;
    \\    }
    \\
    \\    if (is_active) {
    \\        uint idx = atomic_fetch_add_explicit(redex_count, 1, memory_order_relaxed);
    \\        if (idx < redex_cap) {
    \\            redexes[idx] = uint2(id, loc_b);
    \\        }
    \\    }
    \\}
    \\
    \\// Batch beta chain - multiple reductions per thread
    \\kernel void batch_beta_chain(
    \\    device ulong* heap [[buffer(0)]],
    \\    device const uint* app_locs [[buffer(1)]],
    \\    device atomic_uint* stats [[buffer(2)]],
    \\    constant uint& chain_depth [[buffer(3)]],
    \\    uint id [[thread_position_in_grid]]
    \\) {
    \\    uint loc = app_locs[id];
    \\    uint depth = 0;
    \\    while (depth < chain_depth) {
    \\        ulong app = heap[loc];
    \\        if (GET_TAG(app) != TAG_APP) break;
    \\        uint app_val = GET_VAL(app);
    \\        ulong func = heap[app_val];
    \\        if (GET_TAG(func) != TAG_LAM) break;
    \\        uint lam_val = GET_VAL(func);
    \\        ulong body = heap[lam_val];
    \\        ulong arg = heap[app_val + 1];
    \\        if (GET_TAG(body) == TAG_VAR && GET_VAL(body) == lam_val) {
    \\            heap[loc] = arg;
    \\        } else {
    \\            heap[loc] = body;
    \\        }
    \\        heap[app_val] = MAKE_TERM(TAG_ERA, 0, 0);
    \\        atomic_fetch_add_explicit(&stats[0], 1, memory_order_relaxed);
    \\        depth++;
    \\    }
    \\}
;

// =============================================================================
// Metal GPU Context
// =============================================================================

pub const MetalError = error{
    NotSupported,
    DeviceNotAvailable,
    InitFailed,
    ShaderCompileFailed,
    PipelineCreateFailed,
    BufferAllocFailed,
    CommandFailed,
};

// Scratch buffer size (64MB = 16M u32s - fits 10M benchmark)
const SCRATCH_SIZE: usize = 64 * 1024 * 1024;
const SCRATCH_ELEMENTS: usize = SCRATCH_SIZE / @sizeOf(u32);

pub const MetalGPU = struct {
    device: id = null,
    queue: id = null,
    library: id = null,

    // Compute pipelines - basic
    add_pipeline: id = null,
    mul_pipeline: id = null,
    add_vec4_pipeline: id = null,
    mul_vec4_pipeline: id = null,

    // Compute pipelines - HVM interactions
    interact_app_lam_pipeline: id = null,
    interact_sup_co0_pipeline: id = null,
    interact_dup_lam_pipeline: id = null,
    interact_op2_num_pipeline: id = null,
    interact_mat_ctr_pipeline: id = null,
    interact_step_pipeline: id = null,

    // Compute pipelines - GPU-resident reduction
    interact_heterogeneous_pipeline: id = null,
    reduce_step_pipeline: id = null,
    scan_redexes_pipeline: id = null,
    batch_beta_chain_pipeline: id = null,

    // GPU heap
    heap_buffer: id = null,
    heap_size: usize = 0,

    // GPU-resident reduction state
    gpu_heap: id = null,              // Permanent GPU heap
    gpu_heap_ptr: ?[*]u64 = null,     // CPU pointer to GPU heap
    gpu_heap_capacity: usize = 0,     // Max terms in GPU heap
    heap_locks: id = null,            // Per-node locks for GPU reduction
    heap_locks_ptr: ?[*]u32 = null,

    // Redex queues (double-buffered for ping-pong)
    redex_queue_a: id = null,         // Redex queue A
    redex_queue_b: id = null,         // Redex queue B
    redex_ptr_a: ?[*][2]u32 = null,
    redex_ptr_b: ?[*][2]u32 = null,
    redex_capacity: usize = 0,        // Max redexes per queue

    // Control buffers
    alloc_ptr_buf: id = null,         // Atomic allocation pointer
    redex_count_buf: id = null,       // Atomic redex count
    stats_buf: id = null,             // Statistics buffer
    ctrl_ptr: ?[*]u32 = null,         // CPU pointer to control data
    alloc_ptr_ptr: ?*u32 = null,
    redex_count_ptr: ?*u32 = null,
    stats_ptr: ?[*]u32 = null,

    // Template buffers for REF expansion
    tmpl_terms_buf: id = null,
    tmpl_info_buf: id = null,
    tmpl_terms_len: usize = 0,
    tmpl_info_len: usize = 0,

    // Scan constants (heap size, redex cap)
    scan_const_buf: id = null,
    scan_const_ptr: ?[*]u32 = null,

    // Async pipelining (triple buffering)
    async_cmd_bufs: [3]id = .{ null, null, null },
    async_current: usize = 0,
    async_pending: usize = 0,

    // Pre-allocated scratch buffers (shared memory for zero-copy)
    scratch_a: id = null,
    scratch_b: id = null,
    scratch_r: id = null,
    scratch_capacity: usize = 0,

    // Cached buffer pointers (avoid repeated contents calls)
    ptr_a: ?[*]u32 = null,
    ptr_b: ?[*]u32 = null,
    ptr_r: ?[*]u32 = null,

    // HVM interaction scratch buffers (u64 terms)
    hvm_buf0: id = null,
    hvm_buf1: id = null,
    hvm_buf2: id = null,
    hvm_buf3: id = null,
    hvm_ptr0: ?[*]u64 = null,
    hvm_ptr1: ?[*]u64 = null,
    hvm_ptr2: ?[*]u64 = null,
    hvm_ptr3: ?[*]u64 = null,
    hvm_capacity: usize = 0,

    device_name: [256]u8 = undefined,
    max_threads: NSUInteger = 1024,

    const Self = @This();

    pub fn init() MetalError!Self {
        if (!is_supported) return MetalError.NotSupported;

        // Initialize selectors
        sels = Selectors.init();

        var self = Self{};

        // Get device
        self.device = MTLCreateSystemDefaultDevice();
        if (self.device == null) return MetalError.DeviceNotAvailable;

        // Create command queue
        self.queue = msg0(self.device, sels.newCommandQueue);
        if (self.queue == null) return MetalError.InitFailed;

        // Get device name
        const name_obj = msg0(self.device, sels.name);
        if (name_obj != null) {
            const cstr_ptr = msg0(name_obj, sels.UTF8String);
            if (cstr_ptr != null) {
                const cstr: [*:0]const u8 = @ptrCast(cstr_ptr);
                const len = std.mem.len(cstr);
                const copy_len = @min(len, self.device_name.len - 1);
                @memcpy(self.device_name[0..copy_len], cstr[0..copy_len]);
                self.device_name[copy_len] = 0;
            }
        }

        // Compile shaders
        try self.compileShaders();

        // Allocate scratch buffers for zero-copy operations
        try self.allocScratchBuffers();

        return self;
    }

    fn allocScratchBuffers(self: *Self) MetalError!void {
        // Allocate shared memory buffers (storageModeShared = 0)
        self.scratch_a = msg2(self.device, sels.newBufferWithLength_options, SCRATCH_SIZE, @as(u64, 0));
        self.scratch_b = msg2(self.device, sels.newBufferWithLength_options, SCRATCH_SIZE, @as(u64, 0));
        self.scratch_r = msg2(self.device, sels.newBufferWithLength_options, SCRATCH_SIZE, @as(u64, 0));

        if (self.scratch_a == null or self.scratch_b == null or self.scratch_r == null) {
            return MetalError.BufferAllocFailed;
        }

        // Cache the buffer contents pointers
        self.ptr_a = @ptrCast(@alignCast(msg0(self.scratch_a, sels.contents)));
        self.ptr_b = @ptrCast(@alignCast(msg0(self.scratch_b, sels.contents)));
        self.ptr_r = @ptrCast(@alignCast(msg0(self.scratch_r, sels.contents)));

        self.scratch_capacity = SCRATCH_ELEMENTS;

        // HVM interaction buffers (128MB each = 16M u64 terms)
        const HVM_BUF_SIZE: usize = 128 * 1024 * 1024;
        self.hvm_buf0 = msg2(self.device, sels.newBufferWithLength_options, HVM_BUF_SIZE, @as(u64, 0));
        self.hvm_buf1 = msg2(self.device, sels.newBufferWithLength_options, HVM_BUF_SIZE, @as(u64, 0));
        self.hvm_buf2 = msg2(self.device, sels.newBufferWithLength_options, HVM_BUF_SIZE, @as(u64, 0));
        self.hvm_buf3 = msg2(self.device, sels.newBufferWithLength_options, HVM_BUF_SIZE, @as(u64, 0));

        if (self.hvm_buf0 != null and self.hvm_buf1 != null and self.hvm_buf2 != null and self.hvm_buf3 != null) {
            self.hvm_ptr0 = @ptrCast(@alignCast(msg0(self.hvm_buf0, sels.contents)));
            self.hvm_ptr1 = @ptrCast(@alignCast(msg0(self.hvm_buf1, sels.contents)));
            self.hvm_ptr2 = @ptrCast(@alignCast(msg0(self.hvm_buf2, sels.contents)));
            self.hvm_ptr3 = @ptrCast(@alignCast(msg0(self.hvm_buf3, sels.contents)));
            self.hvm_capacity = HVM_BUF_SIZE / @sizeOf(u64);
        }
    }

    fn compileShaders(self: *Self) MetalError!void {
        // Create NSString from shader source
        const NSString = objc_getClass("NSString");
        const source_str = msg1(NSString, sels.stringWithUTF8String, @as([*:0]const u8, shader_source.ptr));
        if (source_str == null) return MetalError.ShaderCompileFailed;

        // Compile library
        var err: id = null;
        self.library = msg3(self.device, sels.newLibraryWithSource_options_error, source_str, @as(id, null), &err);
        if (self.library == null) {
            if (err != null) {
                const desc_obj = msg0(err, sels.localizedDescription);
                if (desc_obj != null) {
                    const cstr_ptr = msg0(desc_obj, sels.UTF8String);
                    if (cstr_ptr != null) {
                        const cstr: [*:0]const u8 = @ptrCast(cstr_ptr);
                        std.debug.print("Metal shader compile error: {s}\n", .{cstr});
                    }
                }
            }
            return MetalError.ShaderCompileFailed;
        }

        // Create pipeline for batch_add
        const add_name = msg1(NSString, sels.stringWithUTF8String, @as([*:0]const u8, "batch_add"));
        const add_fn = msg1(self.library, sels.newFunctionWithName, add_name);
        if (add_fn != null) {
            self.add_pipeline = msg2(self.device, sels.newComputePipelineStateWithFunction_error, add_fn, @as(id, null));
            if (self.add_pipeline != null) {
                self.max_threads = msgU0(self.add_pipeline, sels.maxTotalThreadsPerThreadgroup);
            }
        }

        // Create pipeline for batch_mul
        const mul_name = msg1(NSString, sels.stringWithUTF8String, @as([*:0]const u8, "batch_mul"));
        const mul_fn = msg1(self.library, sels.newFunctionWithName, mul_name);
        if (mul_fn != null) {
            self.mul_pipeline = msg2(self.device, sels.newComputePipelineStateWithFunction_error, mul_fn, @as(id, null));
        }

        // Create pipeline for batch_add_vec4
        const add4_name = msg1(NSString, sels.stringWithUTF8String, @as([*:0]const u8, "batch_add_vec4"));
        const add4_fn = msg1(self.library, sels.newFunctionWithName, add4_name);
        if (add4_fn != null) {
            self.add_vec4_pipeline = msg2(self.device, sels.newComputePipelineStateWithFunction_error, add4_fn, @as(id, null));
        }

        // Create pipeline for batch_mul_vec4
        const mul4_name = msg1(NSString, sels.stringWithUTF8String, @as([*:0]const u8, "batch_mul_vec4"));
        const mul4_fn = msg1(self.library, sels.newFunctionWithName, mul4_name);
        if (mul4_fn != null) {
            self.mul_vec4_pipeline = msg2(self.device, sels.newComputePipelineStateWithFunction_error, mul4_fn, @as(id, null));
        }

        // HVM interaction pipelines
        const kernel_names = [_][*:0]const u8{
            "interact_app_lam",
            "interact_sup_co0",
            "interact_dup_lam",
            "interact_op2_num",
            "interact_mat_ctr",
            "interact_step",
            // GPU-resident reduction kernels
            "interact_heterogeneous",
            "reduce_step",
            "scan_redexes",
            "batch_beta_chain",
        };
        const pipelines = [_]*id{
            &self.interact_app_lam_pipeline,
            &self.interact_sup_co0_pipeline,
            &self.interact_dup_lam_pipeline,
            &self.interact_op2_num_pipeline,
            &self.interact_mat_ctr_pipeline,
            &self.interact_step_pipeline,
            // GPU-resident reduction pipelines
            &self.interact_heterogeneous_pipeline,
            &self.reduce_step_pipeline,
            &self.scan_redexes_pipeline,
            &self.batch_beta_chain_pipeline,
        };

        for (kernel_names, pipelines) |name, pipeline| {
            const fn_name = msg1(NSString, sels.stringWithUTF8String, name);
            const func = msg1(self.library, sels.newFunctionWithName, fn_name);
            if (func != null) {
                pipeline.* = msg2(self.device, sels.newComputePipelineStateWithFunction_error, func, @as(id, null));
            }
        }
    }

    pub fn deinit(self: *Self) void {
        if (!is_supported) return;

        // Release scratch buffers
        if (self.scratch_a != null) msgV0(self.scratch_a, sels.release);
        if (self.scratch_b != null) msgV0(self.scratch_b, sels.release);
        if (self.scratch_r != null) msgV0(self.scratch_r, sels.release);

        // Release HVM interaction buffers
        if (self.hvm_buf0 != null) msgV0(self.hvm_buf0, sels.release);
        if (self.hvm_buf1 != null) msgV0(self.hvm_buf1, sels.release);
        if (self.hvm_buf2 != null) msgV0(self.hvm_buf2, sels.release);
        if (self.hvm_buf3 != null) msgV0(self.hvm_buf3, sels.release);

        // Release GPU-resident state
        if (self.gpu_heap != null) msgV0(self.gpu_heap, sels.release);
        if (self.heap_locks != null) msgV0(self.heap_locks, sels.release);
        if (self.redex_queue_a != null) msgV0(self.redex_queue_a, sels.release);
        if (self.redex_queue_b != null) msgV0(self.redex_queue_b, sels.release);
        if (self.alloc_ptr_buf != null) msgV0(self.alloc_ptr_buf, sels.release);
        if (self.redex_count_buf != null) msgV0(self.redex_count_buf, sels.release);
        if (self.stats_buf != null) msgV0(self.stats_buf, sels.release);
        if (self.tmpl_terms_buf != null) msgV0(self.tmpl_terms_buf, sels.release);
        if (self.tmpl_info_buf != null) msgV0(self.tmpl_info_buf, sels.release);
        if (self.scan_const_buf != null) msgV0(self.scan_const_buf, sels.release);

        if (self.heap_buffer != null) msgV0(self.heap_buffer, sels.release);
        if (self.add_pipeline != null) msgV0(self.add_pipeline, sels.release);
        if (self.mul_pipeline != null) msgV0(self.mul_pipeline, sels.release);
        if (self.add_vec4_pipeline != null) msgV0(self.add_vec4_pipeline, sels.release);
        if (self.mul_vec4_pipeline != null) msgV0(self.mul_vec4_pipeline, sels.release);

        // Release HVM pipelines
        if (self.interact_app_lam_pipeline != null) msgV0(self.interact_app_lam_pipeline, sels.release);
        if (self.interact_sup_co0_pipeline != null) msgV0(self.interact_sup_co0_pipeline, sels.release);
        if (self.interact_dup_lam_pipeline != null) msgV0(self.interact_dup_lam_pipeline, sels.release);
        if (self.interact_op2_num_pipeline != null) msgV0(self.interact_op2_num_pipeline, sels.release);
        if (self.interact_mat_ctr_pipeline != null) msgV0(self.interact_mat_ctr_pipeline, sels.release);
        if (self.interact_step_pipeline != null) msgV0(self.interact_step_pipeline, sels.release);

        if (self.library != null) msgV0(self.library, sels.release);
        if (self.queue != null) msgV0(self.queue, sels.release);

        self.* = .{};
    }

    pub fn getDeviceName(self: *const Self) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.device_name, 0) orelse self.device_name.len;
        return self.device_name[0..len];
    }

    // =========================================================================
    // Buffer Management
    // =========================================================================

    pub fn allocBuffer(self: *Self, size: usize) MetalError!id {
        if (self.device == null) return MetalError.DeviceNotAvailable;
        const buffer = msg2(self.device, sels.newBufferWithLength_options, size, @as(u64, 0));
        if (buffer == null) return MetalError.BufferAllocFailed;
        return buffer;
    }

    pub fn allocBufferWithData(self: *Self, data: []const u8) MetalError!id {
        if (self.device == null) return MetalError.DeviceNotAvailable;
        const buffer = msg3(self.device, sels.newBufferWithBytes_length_options, data.ptr, data.len, @as(u64, 0));
        if (buffer == null) return MetalError.BufferAllocFailed;
        return buffer;
    }

    pub fn getBufferContents(_: *Self, buffer: id) ?*anyopaque {
        if (buffer == null) return null;
        return msg0(buffer, sels.contents);
    }

    pub fn allocHeap(self: *Self, term_count: usize) MetalError!void {
        const byte_size = term_count * @sizeOf(u64);
        self.heap_buffer = try self.allocBuffer(byte_size);
        self.heap_size = term_count;
    }

    pub fn freeHeap(self: *Self) void {
        if (self.heap_buffer != null) {
            msgV0(self.heap_buffer, sels.release);
            self.heap_buffer = null;
            self.heap_size = 0;
        }
    }

    pub fn uploadHeap(self: *Self, data: []const u64) MetalError!void {
        if (self.heap_buffer == null) return MetalError.BufferAllocFailed;
        const contents = self.getBufferContents(self.heap_buffer) orelse return MetalError.BufferAllocFailed;
        const dest: [*]u64 = @ptrCast(@alignCast(contents));
        @memcpy(dest[0..data.len], data);
    }

    pub fn downloadHeap(self: *Self, data: []u64) MetalError!void {
        if (self.heap_buffer == null) return MetalError.BufferAllocFailed;
        const contents = self.getBufferContents(self.heap_buffer) orelse return MetalError.BufferAllocFailed;
        const src: [*]const u64 = @ptrCast(@alignCast(contents));
        @memcpy(data, src[0..data.len]);
    }

    // =========================================================================
    // GPU Compute Dispatch (Zero-Copy Optimized)
    // =========================================================================

    /// Execute GPU batch add using pre-allocated scratch buffers
    /// This eliminates per-call buffer allocation overhead
    pub fn gpuBatchAdd(self: *Self, a: []const u32, b: []const u32, results: []u32) MetalError!void {
        if (self.add_pipeline == null) return MetalError.PipelineCreateFailed;

        const count = @min(a.len, @min(b.len, results.len));

        // Use scratch buffers for data that fits
        if (count <= self.scratch_capacity and self.ptr_a != null and self.ptr_b != null and self.ptr_r != null) {
            // Zero-copy path: write directly to GPU-visible memory
            @memcpy(self.ptr_a.?[0..count], a[0..count]);
            @memcpy(self.ptr_b.?[0..count], b[0..count]);

            // Use vec4 pipeline for large, aligned batches (4x fewer threads = less dispatch overhead)
            if (count >= 4096 and count % 4 == 0 and self.add_vec4_pipeline != null) {
                try self.dispatchComputeVec4(self.add_vec4_pipeline, self.scratch_a, self.scratch_b, self.scratch_r, count);
            } else {
                try self.dispatchCompute(self.add_pipeline, self.scratch_a, self.scratch_b, self.scratch_r, count);
            }

            // Read results directly from GPU-visible memory
            @memcpy(results[0..count], self.ptr_r.?[0..count]);
        } else {
            // Fallback: allocate temporary buffers for large data
            try self.gpuBatchAddSlow(a, b, results);
        }
    }

    /// Execute GPU batch multiply using pre-allocated scratch buffers
    pub fn gpuBatchMul(self: *Self, a: []const u32, b: []const u32, results: []u32) MetalError!void {
        if (self.mul_pipeline == null) return MetalError.PipelineCreateFailed;

        const count = @min(a.len, @min(b.len, results.len));

        if (count <= self.scratch_capacity and self.ptr_a != null and self.ptr_b != null and self.ptr_r != null) {
            @memcpy(self.ptr_a.?[0..count], a[0..count]);
            @memcpy(self.ptr_b.?[0..count], b[0..count]);

            // Use vec4 pipeline for large, aligned batches
            if (count >= 4096 and count % 4 == 0 and self.mul_vec4_pipeline != null) {
                try self.dispatchComputeVec4(self.mul_vec4_pipeline, self.scratch_a, self.scratch_b, self.scratch_r, count);
            } else {
                try self.dispatchCompute(self.mul_pipeline, self.scratch_a, self.scratch_b, self.scratch_r, count);
            }

            @memcpy(results[0..count], self.ptr_r.?[0..count]);
        } else {
            try self.gpuBatchMulSlow(a, b, results);
        }
    }

    /// Shared compute dispatch logic (no buffer allocation)
    fn dispatchCompute(self: *Self, pipeline: id, buf_a: id, buf_b: id, buf_r: id, count: usize) MetalError!void {
        const cmd_buf = msg0(self.queue, sels.commandBuffer);
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, pipeline);

        msgV3(encoder, sels.setBuffer_offset_atIndex, buf_a, @as(NSUInteger, 0), @as(NSUInteger, 0));
        msgV3(encoder, sels.setBuffer_offset_atIndex, buf_b, @as(NSUInteger, 0), @as(NSUInteger, 1));
        msgV3(encoder, sels.setBuffer_offset_atIndex, buf_r, @as(NSUInteger, 0), @as(NSUInteger, 2));

        const threads_per_group = @min(self.max_threads, count);
        const num_groups = (count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
        msgV0(cmd_buf, sels.commit);
        msgV0(cmd_buf, sels.waitUntilCompleted);
    }

    /// Dispatch using vec4 pipeline (4 elements per thread) - more efficient for large batches
    fn dispatchComputeVec4(self: *Self, pipeline: id, buf_a: id, buf_b: id, buf_r: id, count: usize) MetalError!void {
        const cmd_buf = msg0(self.queue, sels.commandBuffer);
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, pipeline);

        msgV3(encoder, sels.setBuffer_offset_atIndex, buf_a, @as(NSUInteger, 0), @as(NSUInteger, 0));
        msgV3(encoder, sels.setBuffer_offset_atIndex, buf_b, @as(NSUInteger, 0), @as(NSUInteger, 1));
        msgV3(encoder, sels.setBuffer_offset_atIndex, buf_r, @as(NSUInteger, 0), @as(NSUInteger, 2));

        // For vec4, we process 4 elements per thread
        const vec4_count = count / 4;
        const threads_per_group = @min(self.max_threads, vec4_count);
        const num_groups = (vec4_count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
        msgV0(cmd_buf, sels.commit);
        msgV0(cmd_buf, sels.waitUntilCompleted);
    }

    /// Slow path with buffer allocation for oversized data
    fn gpuBatchAddSlow(self: *Self, a: []const u32, b: []const u32, results: []u32) MetalError!void {
        const count = @min(a.len, @min(b.len, results.len));
        const byte_size = count * @sizeOf(u32);

        const buf_a = msg3(self.device, sels.newBufferWithBytes_length_options, a.ptr, byte_size, @as(u64, 0));
        const buf_b = msg3(self.device, sels.newBufferWithBytes_length_options, b.ptr, byte_size, @as(u64, 0));
        const buf_r = msg2(self.device, sels.newBufferWithLength_options, byte_size, @as(u64, 0));

        if (buf_a == null or buf_b == null or buf_r == null) return MetalError.BufferAllocFailed;
        defer {
            msgV0(buf_a, sels.release);
            msgV0(buf_b, sels.release);
            msgV0(buf_r, sels.release);
        }

        try self.dispatchCompute(self.add_pipeline, buf_a, buf_b, buf_r, count);

        const result_ptr = self.getBufferContents(buf_r) orelse return MetalError.CommandFailed;
        const src: [*]const u32 = @ptrCast(@alignCast(result_ptr));
        @memcpy(results[0..count], src[0..count]);
    }

    fn gpuBatchMulSlow(self: *Self, a: []const u32, b: []const u32, results: []u32) MetalError!void {
        const count = @min(a.len, @min(b.len, results.len));
        const byte_size = count * @sizeOf(u32);

        const buf_a = msg3(self.device, sels.newBufferWithBytes_length_options, a.ptr, byte_size, @as(u64, 0));
        const buf_b = msg3(self.device, sels.newBufferWithBytes_length_options, b.ptr, byte_size, @as(u64, 0));
        const buf_r = msg2(self.device, sels.newBufferWithLength_options, byte_size, @as(u64, 0));

        if (buf_a == null or buf_b == null or buf_r == null) return MetalError.BufferAllocFailed;
        defer {
            msgV0(buf_a, sels.release);
            msgV0(buf_b, sels.release);
            msgV0(buf_r, sels.release);
        }

        try self.dispatchCompute(self.mul_pipeline, buf_a, buf_b, buf_r, count);

        const result_ptr = self.getBufferContents(buf_r) orelse return MetalError.CommandFailed;
        const src: [*]const u32 = @ptrCast(@alignCast(result_ptr));
        @memcpy(results[0..count], src[0..count]);
    }

    // =========================================================================
    // HVM Interaction Kernels (GPU-accelerated)
    // =========================================================================

    /// Dispatch 4-buffer compute kernel (for HVM interactions with heap access)
    fn dispatch4(self: *Self, pipeline: id, buf0: id, buf1: id, buf2: id, buf3: id, count: usize) MetalError!void {
        const cmd_buf = msg0(self.queue, sels.commandBuffer);
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, pipeline);

        msgV3(encoder, sels.setBuffer_offset_atIndex, buf0, @as(NSUInteger, 0), @as(NSUInteger, 0));
        msgV3(encoder, sels.setBuffer_offset_atIndex, buf1, @as(NSUInteger, 0), @as(NSUInteger, 1));
        msgV3(encoder, sels.setBuffer_offset_atIndex, buf2, @as(NSUInteger, 0), @as(NSUInteger, 2));
        msgV3(encoder, sels.setBuffer_offset_atIndex, buf3, @as(NSUInteger, 0), @as(NSUInteger, 3));

        const threads_per_group = @min(self.max_threads, count);
        const num_groups = (count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
        msgV0(cmd_buf, sels.commit);
        msgV0(cmd_buf, sels.waitUntilCompleted);
    }

    /// GPU APP-LAM interaction (beta reduction)
    pub fn gpuInteractAppLam(self: *Self, apps: []const u64, lams: []const u64, results: []u64, heap: []const u64) MetalError!void {
        if (self.interact_app_lam_pipeline == null) return MetalError.PipelineCreateFailed;

        const count = @min(apps.len, @min(lams.len, results.len));

        // Use pre-allocated buffers if data fits
        if (count <= self.hvm_capacity and heap.len <= self.hvm_capacity and
            self.hvm_ptr0 != null and self.hvm_ptr1 != null and self.hvm_ptr2 != null and self.hvm_ptr3 != null)
        {
            // Zero-copy: write directly to GPU-visible memory
            @memcpy(self.hvm_ptr0.?[0..count], apps[0..count]);
            @memcpy(self.hvm_ptr1.?[0..count], lams[0..count]);
            @memcpy(self.hvm_ptr3.?[0..heap.len], heap);

            try self.dispatch4(self.interact_app_lam_pipeline, self.hvm_buf0, self.hvm_buf1, self.hvm_buf2, self.hvm_buf3, count);

            @memcpy(results[0..count], self.hvm_ptr2.?[0..count]);
        } else {
            // Fallback: allocate temporary buffers
            const term_size = count * @sizeOf(u64);
            const heap_size = heap.len * @sizeOf(u64);

            const buf_apps = msg3(self.device, sels.newBufferWithBytes_length_options, apps.ptr, term_size, @as(u64, 0));
            const buf_lams = msg3(self.device, sels.newBufferWithBytes_length_options, lams.ptr, term_size, @as(u64, 0));
            const buf_results = msg2(self.device, sels.newBufferWithLength_options, term_size, @as(u64, 0));
            const buf_heap = msg3(self.device, sels.newBufferWithBytes_length_options, heap.ptr, heap_size, @as(u64, 0));

            if (buf_apps == null or buf_lams == null or buf_results == null or buf_heap == null) return MetalError.BufferAllocFailed;
            defer {
                msgV0(buf_apps, sels.release);
                msgV0(buf_lams, sels.release);
                msgV0(buf_results, sels.release);
                msgV0(buf_heap, sels.release);
            }

            try self.dispatch4(self.interact_app_lam_pipeline, buf_apps, buf_lams, buf_results, buf_heap, count);

            const result_ptr = self.getBufferContents(buf_results) orelse return MetalError.CommandFailed;
            const src: [*]const u64 = @ptrCast(@alignCast(result_ptr));
            @memcpy(results[0..count], src[0..count]);
        }
    }

    /// GPU SUP-CO0 interaction (annihilation/commutation)
    pub fn gpuInteractSupCo0(self: *Self, sups: []const u64, co0s: []const u64, results: []u64, heap: []const u64) MetalError!void {
        if (self.interact_sup_co0_pipeline == null) return MetalError.PipelineCreateFailed;

        const count = @min(sups.len, @min(co0s.len, results.len));

        if (count <= self.hvm_capacity and heap.len <= self.hvm_capacity and
            self.hvm_ptr0 != null and self.hvm_ptr1 != null and self.hvm_ptr2 != null and self.hvm_ptr3 != null)
        {
            @memcpy(self.hvm_ptr0.?[0..count], sups[0..count]);
            @memcpy(self.hvm_ptr1.?[0..count], co0s[0..count]);
            @memcpy(self.hvm_ptr3.?[0..heap.len], heap);

            try self.dispatch4(self.interact_sup_co0_pipeline, self.hvm_buf0, self.hvm_buf1, self.hvm_buf2, self.hvm_buf3, count);

            @memcpy(results[0..count], self.hvm_ptr2.?[0..count]);
        } else {
            const term_size = count * @sizeOf(u64);
            const heap_size = heap.len * @sizeOf(u64);

            const buf_sups = msg3(self.device, sels.newBufferWithBytes_length_options, sups.ptr, term_size, @as(u64, 0));
            const buf_co0s = msg3(self.device, sels.newBufferWithBytes_length_options, co0s.ptr, term_size, @as(u64, 0));
            const buf_results = msg2(self.device, sels.newBufferWithLength_options, term_size, @as(u64, 0));
            const buf_heap = msg3(self.device, sels.newBufferWithBytes_length_options, heap.ptr, heap_size, @as(u64, 0));

            if (buf_sups == null or buf_co0s == null or buf_results == null or buf_heap == null) return MetalError.BufferAllocFailed;
            defer {
                msgV0(buf_sups, sels.release);
                msgV0(buf_co0s, sels.release);
                msgV0(buf_results, sels.release);
                msgV0(buf_heap, sels.release);
            }

            try self.dispatch4(self.interact_sup_co0_pipeline, buf_sups, buf_co0s, buf_results, buf_heap, count);

            const result_ptr = self.getBufferContents(buf_results) orelse return MetalError.CommandFailed;
            const src: [*]const u64 = @ptrCast(@alignCast(result_ptr));
            @memcpy(results[0..count], src[0..count]);
        }
    }

    /// GPU OP2-NUM interaction (arithmetic operations)
    pub fn gpuInteractOp2Num(self: *Self, ops: []const u64, nums: []const u64, results: []u64, heap: []const u64) MetalError!void {
        if (self.interact_op2_num_pipeline == null) return MetalError.PipelineCreateFailed;

        const count = @min(ops.len, @min(nums.len, results.len));

        if (count <= self.hvm_capacity and heap.len <= self.hvm_capacity and
            self.hvm_ptr0 != null and self.hvm_ptr1 != null and self.hvm_ptr2 != null and self.hvm_ptr3 != null)
        {
            @memcpy(self.hvm_ptr0.?[0..count], ops[0..count]);
            @memcpy(self.hvm_ptr1.?[0..count], nums[0..count]);
            @memcpy(self.hvm_ptr3.?[0..heap.len], heap);

            try self.dispatch4(self.interact_op2_num_pipeline, self.hvm_buf0, self.hvm_buf1, self.hvm_buf2, self.hvm_buf3, count);

            @memcpy(results[0..count], self.hvm_ptr2.?[0..count]);
        } else {
            const term_size = count * @sizeOf(u64);
            const heap_size = heap.len * @sizeOf(u64);

            const buf_ops = msg3(self.device, sels.newBufferWithBytes_length_options, ops.ptr, term_size, @as(u64, 0));
            const buf_nums = msg3(self.device, sels.newBufferWithBytes_length_options, nums.ptr, term_size, @as(u64, 0));
            const buf_results = msg2(self.device, sels.newBufferWithLength_options, term_size, @as(u64, 0));
            const buf_heap = msg3(self.device, sels.newBufferWithBytes_length_options, heap.ptr, heap_size, @as(u64, 0));

            if (buf_ops == null or buf_nums == null or buf_results == null or buf_heap == null) return MetalError.BufferAllocFailed;
            defer {
                msgV0(buf_ops, sels.release);
                msgV0(buf_nums, sels.release);
                msgV0(buf_results, sels.release);
                msgV0(buf_heap, sels.release);
            }

            try self.dispatch4(self.interact_op2_num_pipeline, buf_ops, buf_nums, buf_results, buf_heap, count);

            const result_ptr = self.getBufferContents(buf_results) orelse return MetalError.CommandFailed;
            const src: [*]const u64 = @ptrCast(@alignCast(result_ptr));
            @memcpy(results[0..count], src[0..count]);
        }
    }

    /// GPU MAT-CTR interaction (pattern matching)
    pub fn gpuInteractMatCtr(self: *Self, mats: []const u64, ctrs: []const u64, results: []u64, branches: []const u64) MetalError!void {
        if (self.interact_mat_ctr_pipeline == null) return MetalError.PipelineCreateFailed;

        const count = @min(mats.len, @min(ctrs.len, results.len));
        const term_size = count * @sizeOf(u64);
        const branch_size = branches.len * @sizeOf(u64);

        const buf_mats = msg3(self.device, sels.newBufferWithBytes_length_options, mats.ptr, term_size, @as(u64, 0));
        const buf_ctrs = msg3(self.device, sels.newBufferWithBytes_length_options, ctrs.ptr, term_size, @as(u64, 0));
        const buf_results = msg2(self.device, sels.newBufferWithLength_options, term_size, @as(u64, 0));
        const buf_branches = msg3(self.device, sels.newBufferWithBytes_length_options, branches.ptr, branch_size, @as(u64, 0));

        if (buf_mats == null or buf_ctrs == null or buf_results == null or buf_branches == null) return MetalError.BufferAllocFailed;
        defer {
            msgV0(buf_mats, sels.release);
            msgV0(buf_ctrs, sels.release);
            msgV0(buf_results, sels.release);
            msgV0(buf_branches, sels.release);
        }

        try self.dispatch4(self.interact_mat_ctr_pipeline, buf_mats, buf_ctrs, buf_results, buf_branches, count);

        const result_ptr = self.getBufferContents(buf_results) orelse return MetalError.CommandFailed;
        const src: [*]const u64 = @ptrCast(@alignCast(result_ptr));
        @memcpy(results[0..count], src[0..count]);
    }

    // =========================================================================
    // GPU-RESIDENT REDUCTION ENGINE
    // =========================================================================

    /// Allocate GPU-resident state for reduction without CPU round-trips
    /// heap_capacity: maximum number of u64 terms in heap
    /// redex_capacity: maximum number of redexes per queue
    pub fn allocGpuResidentState(self: *Self, heap_capacity: usize, redex_cap: usize) MetalError!void {
        const heap_bytes = heap_capacity * @sizeOf(u64);
        const redex_bytes = redex_cap * 2 * @sizeOf(u32); // uint2 = 2 x u32
        const ctrl_bytes: usize = 16 * @sizeOf(u32); // Control data

        // Allocate GPU heap (storageModeShared = 0 for CPU access)
        self.gpu_heap = msg2(self.device, sels.newBufferWithLength_options, heap_bytes, @as(u64, 0));
        if (self.gpu_heap == null) return MetalError.BufferAllocFailed;
        self.gpu_heap_ptr = @ptrCast(@alignCast(msg0(self.gpu_heap, sels.contents)));
        self.gpu_heap_capacity = heap_capacity;

        const lock_bytes = heap_capacity * @sizeOf(u32);
        self.heap_locks = msg2(self.device, sels.newBufferWithLength_options, lock_bytes, @as(u64, 0));
        if (self.heap_locks == null) return MetalError.BufferAllocFailed;
        self.heap_locks_ptr = @ptrCast(@alignCast(msg0(self.heap_locks, sels.contents)));
        if (self.heap_locks_ptr) |ptr| {
            @memset(ptr[0..heap_capacity], 0);
        } else {
            return MetalError.BufferAllocFailed;
        }

        // Allocate double-buffered redex queues
        self.redex_queue_a = msg2(self.device, sels.newBufferWithLength_options, redex_bytes, @as(u64, 0));
        self.redex_queue_b = msg2(self.device, sels.newBufferWithLength_options, redex_bytes, @as(u64, 0));
        if (self.redex_queue_a == null or self.redex_queue_b == null) return MetalError.BufferAllocFailed;
        self.redex_ptr_a = @ptrCast(@alignCast(msg0(self.redex_queue_a, sels.contents)));
        self.redex_ptr_b = @ptrCast(@alignCast(msg0(self.redex_queue_b, sels.contents)));
        self.redex_capacity = redex_cap;

        // Allocate control buffers
        self.alloc_ptr_buf = msg2(self.device, sels.newBufferWithLength_options, @as(usize, 4), @as(u64, 0));
        self.redex_count_buf = msg2(self.device, sels.newBufferWithLength_options, @as(usize, 8), @as(u64, 0));
        self.stats_buf = msg2(self.device, sels.newBufferWithLength_options, ctrl_bytes, @as(u64, 0));
        if (self.alloc_ptr_buf == null or self.redex_count_buf == null or self.stats_buf == null) {
            return MetalError.BufferAllocFailed;
        }
        self.ctrl_ptr = @ptrCast(@alignCast(msg0(self.stats_buf, sels.contents)));
        self.alloc_ptr_ptr = @ptrCast(@alignCast(msg0(self.alloc_ptr_buf, sels.contents)));
        self.redex_count_ptr = @ptrCast(@alignCast(msg0(self.redex_count_buf, sels.contents)));
        self.stats_ptr = self.ctrl_ptr;

        self.scan_const_buf = msg2(self.device, sels.newBufferWithLength_options, @as(usize, 8), @as(u64, 0));
        if (self.scan_const_buf == null) return MetalError.BufferAllocFailed;
        self.scan_const_ptr = @ptrCast(@alignCast(msg0(self.scan_const_buf, sels.contents)));
    }

    pub fn uploadTemplates(self: *Self, terms: []const u64, info: []const u32) MetalError!void {
        const terms_len = if (terms.len == 0) 1 else terms.len;
        const info_len = if (info.len == 0) 1 else info.len;
        const terms_bytes = terms_len * @sizeOf(u64);
        const info_bytes = info_len * @sizeOf(u32);

        if (self.tmpl_terms_buf != null) msgV0(self.tmpl_terms_buf, sels.release);
        if (self.tmpl_info_buf != null) msgV0(self.tmpl_info_buf, sels.release);

        self.tmpl_terms_buf = msg2(self.device, sels.newBufferWithLength_options, terms_bytes, @as(u64, 0));
        self.tmpl_info_buf = msg2(self.device, sels.newBufferWithLength_options, info_bytes, @as(u64, 0));
        if (self.tmpl_terms_buf == null or self.tmpl_info_buf == null) return MetalError.BufferAllocFailed;

        const terms_ptr = msg0(self.tmpl_terms_buf, sels.contents);
        const info_ptr = msg0(self.tmpl_info_buf, sels.contents);
        if (terms_ptr == null or info_ptr == null) return MetalError.BufferAllocFailed;

        const terms_dst: [*]u64 = @ptrCast(@alignCast(terms_ptr));
        const info_dst: [*]u32 = @ptrCast(@alignCast(info_ptr));
        if (terms.len > 0) {
            @memcpy(terms_dst[0..terms.len], terms);
        } else {
            terms_dst[0] = 0;
        }
        if (info.len > 0) {
            @memcpy(info_dst[0..info.len], info);
        } else {
            info_dst[0] = 0;
        }

        self.tmpl_terms_len = terms.len;
        self.tmpl_info_len = info.len;
    }

    /// Dispatch multi-buffer compute kernel (6 buffers for GPU-resident reduction)
    fn dispatch6(self: *Self, pipeline: id, bufs: [6]id, count: usize) MetalError!void {
        const cmd_buf = msg0(self.queue, sels.commandBuffer);
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, pipeline);

        for (bufs, 0..) |buf, i| {
            msgV3(encoder, sels.setBuffer_offset_atIndex, buf, @as(NSUInteger, 0), @as(NSUInteger, i));
        }

        const threads_per_group = @min(self.max_threads, count);
        const num_groups = (count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
        msgV0(cmd_buf, sels.commit);
        msgV0(cmd_buf, sels.waitUntilCompleted);
    }

    /// Dispatch 7-buffer compute kernel (for reduce_step)
    fn dispatch7(self: *Self, pipeline: id, bufs: [7]id, count: usize) MetalError!void {
        const cmd_buf = msg0(self.queue, sels.commandBuffer);
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, pipeline);

        for (bufs, 0..) |buf, i| {
            msgV3(encoder, sels.setBuffer_offset_atIndex, buf, @as(NSUInteger, 0), @as(NSUInteger, i));
        }

        const threads_per_group = @min(self.max_threads, count);
        const num_groups = (count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
        msgV0(cmd_buf, sels.commit);
        msgV0(cmd_buf, sels.waitUntilCompleted);
    }

    pub fn scanRedexes(self: *Self, heap_len: u32) MetalError!u32 {
        if (self.scan_redexes_pipeline == null) return MetalError.PipelineCreateFailed;
        if (self.gpu_heap == null) return MetalError.BufferAllocFailed;
        if (self.redex_queue_a == null or self.redex_count_buf == null) return MetalError.BufferAllocFailed;
        if (self.scan_const_buf == null or self.scan_const_ptr == null) return MetalError.BufferAllocFailed;

        if (heap_len == 0) return 0;

        if (self.redex_count_ptr) |ptr| {
            ptr.* = 0;
        } else {
            return MetalError.BufferAllocFailed;
        }

        const scan_ptr = self.scan_const_ptr.?;
        scan_ptr[0] = heap_len;
        scan_ptr[1] = @intCast(self.redex_capacity);

        const cmd_buf = msg0(self.queue, sels.commandBuffer);
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, self.scan_redexes_pipeline);

        msgV3(encoder, sels.setBuffer_offset_atIndex, self.gpu_heap, @as(NSUInteger, 0), @as(NSUInteger, 0));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_queue_a, @as(NSUInteger, 0), @as(NSUInteger, 1));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_count_buf, @as(NSUInteger, 0), @as(NSUInteger, 2));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.scan_const_buf, @as(NSUInteger, 0), @as(NSUInteger, 3));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.scan_const_buf, @as(NSUInteger, 4), @as(NSUInteger, 4));

        const count: usize = heap_len;
        const threads_per_group = @min(self.max_threads, count);
        const num_groups = (count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
        msgV0(cmd_buf, sels.commit);
        msgV0(cmd_buf, sels.waitUntilCompleted);

        return self.redex_count_ptr.?.*;
    }

    pub fn gpuReduceStep(self: *Self, in_count: u32) MetalError!GpuReductionStats {
        if (self.reduce_step_pipeline == null) return MetalError.PipelineCreateFailed;
        if (self.gpu_heap == null) return MetalError.BufferAllocFailed;
        if (self.heap_locks == null) return MetalError.BufferAllocFailed;
        if (self.tmpl_terms_buf == null or self.tmpl_info_buf == null) return MetalError.BufferAllocFailed;
        if (self.redex_queue_a == null or self.redex_queue_b == null) return MetalError.BufferAllocFailed;
        if (in_count == 0) return .{ .interactions = 0, .allocations = 0, .new_redexes = 0 };

        if (self.redex_count_ptr == null or self.stats_ptr == null) return MetalError.BufferAllocFailed;

        self.redex_count_ptr.?.* = in_count;
        const stats_ptr = self.stats_ptr.?;
        stats_ptr[0] = 0;
        stats_ptr[1] = 0;
        stats_ptr[2] = 0;
        stats_ptr[3] = @intCast(self.redex_capacity);

        const cmd_buf = msg0(self.queue, sels.commandBuffer);
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, self.reduce_step_pipeline);

        msgV3(encoder, sels.setBuffer_offset_atIndex, self.gpu_heap, @as(NSUInteger, 0), @as(NSUInteger, 0));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_queue_a, @as(NSUInteger, 0), @as(NSUInteger, 1));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_queue_b, @as(NSUInteger, 0), @as(NSUInteger, 2));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.alloc_ptr_buf, @as(NSUInteger, 0), @as(NSUInteger, 3));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_count_buf, @as(NSUInteger, 0), @as(NSUInteger, 4));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.stats_buf, @as(NSUInteger, 8), @as(NSUInteger, 5));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.stats_buf, @as(NSUInteger, 0), @as(NSUInteger, 6));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.tmpl_terms_buf, @as(NSUInteger, 0), @as(NSUInteger, 7));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.tmpl_info_buf, @as(NSUInteger, 0), @as(NSUInteger, 8));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.heap_locks, @as(NSUInteger, 0), @as(NSUInteger, 9));

        const count: usize = in_count;
        const threads_per_group = @min(self.max_threads, count);
        const num_groups = (count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
        msgV0(cmd_buf, sels.commit);
        msgV0(cmd_buf, sels.waitUntilCompleted);

        return .{
            .interactions = stats_ptr[0],
            .allocations = stats_ptr[1],
            .new_redexes = stats_ptr[2],
        };
    }

    /// GPU-resident heterogeneous batch interaction
    /// Processes mixed redex types (APP-LAM, CO0-SUP, OP2-NUM, etc.) in single dispatch
    /// Returns: number of interactions performed
    pub fn gpuInteractHeterogeneous(
        self: *Self,
        redexes: []const [2]u32,
        initial_alloc: u32,
    ) MetalError!GpuReductionStats {
        if (self.interact_heterogeneous_pipeline == null) return MetalError.PipelineCreateFailed;
        if (self.gpu_heap == null) return MetalError.BufferAllocFailed;

        const count = redexes.len;
        if (count == 0) return .{ .interactions = 0, .allocations = 0, .new_redexes = 0 };

        // Copy redexes to queue A
        const redex_ptr = self.redex_ptr_a orelse return MetalError.BufferAllocFailed;
        @memcpy(redex_ptr[0..count], redexes);

        // Initialize control values
        const alloc_ptr: *u32 = @ptrCast(@alignCast(msg0(self.alloc_ptr_buf, sels.contents)));
        alloc_ptr.* = initial_alloc;

        const out_count_ptr: *u32 = @ptrCast(@alignCast(msg0(self.redex_count_buf, sels.contents)));
        out_count_ptr.* = 0;

        const stats_ptr: [*]u32 = @ptrCast(@alignCast(msg0(self.stats_buf, sels.contents)));
        stats_ptr[0] = 0; // interactions
        stats_ptr[1] = 0; // allocations

        // Dispatch heterogeneous kernel
        try self.dispatch6(self.interact_heterogeneous_pipeline, .{
            self.gpu_heap,
            self.redex_queue_a,
            self.alloc_ptr_buf,
            self.redex_queue_b,
            self.redex_count_buf,
            self.stats_buf,
        }, count);

        // Read results
        return .{
            .interactions = stats_ptr[0],
            .allocations = stats_ptr[1],
            .new_redexes = out_count_ptr.*,
        };
    }

    /// GPU-resident reduction loop - runs multiple steps without CPU round-trip
    /// max_steps: maximum reduction steps to run
    /// Returns: total statistics from all steps
    pub fn gpuReduceResident(
        self: *Self,
        initial_redexes: []const [2]u32,
        initial_alloc: u32,
        max_steps: u32,
    ) MetalError!GpuReductionStats {
        if (self.reduce_step_pipeline == null) return MetalError.PipelineCreateFailed;
        if (self.gpu_heap == null) return MetalError.BufferAllocFailed;
        if (self.heap_locks == null) return MetalError.BufferAllocFailed;
        if (self.tmpl_terms_buf == null or self.tmpl_info_buf == null) return MetalError.BufferAllocFailed;

        var total_stats = GpuReductionStats{
            .interactions = 0,
            .allocations = 0,
            .new_redexes = 0,
        };

        // Copy initial redexes
        var current_count = initial_redexes.len;
        if (current_count == 0) return total_stats;

        const redex_ptr_a = self.redex_ptr_a orelse return MetalError.BufferAllocFailed;
        @memcpy(redex_ptr_a[0..current_count], initial_redexes);

        // Initialize allocation pointer
        const alloc_ptr: *u32 = @ptrCast(@alignCast(msg0(self.alloc_ptr_buf, sels.contents)));
        alloc_ptr.* = initial_alloc;

        // Get control pointers
        const in_count_ptr: *u32 = @ptrCast(@alignCast(msg0(self.redex_count_buf, sels.contents)));
        const stats_ptr: [*]u32 = @ptrCast(@alignCast(msg0(self.stats_buf, sels.contents)));

        // We need a separate buffer for out_count - use stats[2]
        var current_in_queue = self.redex_queue_a;
        var current_out_queue = self.redex_queue_b;

        var step: u32 = 0;
        while (step < max_steps and current_count > 0) : (step += 1) {
            // Set input count
            in_count_ptr.* = @intCast(current_count);
            stats_ptr[0] = 0; // interactions
            stats_ptr[1] = 0; // allocations
            stats_ptr[2] = 0; // out_count
            stats_ptr[3] = @intCast(self.redex_capacity);

            // Create out_count buffer pointing to stats[2]
            // Dispatch reduce_step kernel
            const cmd_buf = msg0(self.queue, sels.commandBuffer);
            if (cmd_buf == null) return MetalError.CommandFailed;

            const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
            if (encoder == null) return MetalError.CommandFailed;

            const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
            setPipeline(encoder, sels.setComputePipelineState, self.reduce_step_pipeline);

            msgV3(encoder, sels.setBuffer_offset_atIndex, self.gpu_heap, @as(NSUInteger, 0), @as(NSUInteger, 0));
            msgV3(encoder, sels.setBuffer_offset_atIndex, current_in_queue, @as(NSUInteger, 0), @as(NSUInteger, 1));
            msgV3(encoder, sels.setBuffer_offset_atIndex, current_out_queue, @as(NSUInteger, 0), @as(NSUInteger, 2));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.alloc_ptr_buf, @as(NSUInteger, 0), @as(NSUInteger, 3));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_count_buf, @as(NSUInteger, 0), @as(NSUInteger, 4));
            // Use stats_buf offset 8 for out_count (stats[2])
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.stats_buf, @as(NSUInteger, 8), @as(NSUInteger, 5));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.stats_buf, @as(NSUInteger, 0), @as(NSUInteger, 6));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.tmpl_terms_buf, @as(NSUInteger, 0), @as(NSUInteger, 7));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.tmpl_info_buf, @as(NSUInteger, 0), @as(NSUInteger, 8));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.heap_locks, @as(NSUInteger, 0), @as(NSUInteger, 9));

            const threads_per_group = @min(self.max_threads, current_count);
            const num_groups = (current_count + threads_per_group - 1) / threads_per_group;

            msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

            msgV0(encoder, sels.endEncoding);
            msgV0(cmd_buf, sels.commit);
            msgV0(cmd_buf, sels.waitUntilCompleted);

            // Accumulate stats
            total_stats.interactions += stats_ptr[0];
            total_stats.allocations += stats_ptr[1];

            // Get new redex count and swap buffers
            current_count = stats_ptr[2];
            const tmp = current_in_queue;
            current_in_queue = current_out_queue;
            current_out_queue = tmp;
        }

        total_stats.new_redexes = @intCast(current_count);
        return total_stats;
    }

    /// GPU-resident reduction loop starting from an existing queue (redex_queue_a)
    pub fn gpuReduceResidentQueued(
        self: *Self,
        initial_count: u32,
        initial_alloc: u32,
        max_steps: u32,
    ) MetalError!GpuReductionStats {
        if (self.reduce_step_pipeline == null) return MetalError.PipelineCreateFailed;
        if (self.gpu_heap == null) return MetalError.BufferAllocFailed;
        if (self.heap_locks == null) return MetalError.BufferAllocFailed;
        if (self.tmpl_terms_buf == null or self.tmpl_info_buf == null) return MetalError.BufferAllocFailed;

        var total_stats = GpuReductionStats{
            .interactions = 0,
            .allocations = 0,
            .new_redexes = 0,
        };

        var current_count: u32 = initial_count;
        if (current_count == 0) return total_stats;

        // Initialize allocation pointer
        const alloc_ptr: *u32 = @ptrCast(@alignCast(msg0(self.alloc_ptr_buf, sels.contents)));
        alloc_ptr.* = initial_alloc;

        // Get control pointers
        const in_count_ptr: *u32 = @ptrCast(@alignCast(msg0(self.redex_count_buf, sels.contents)));
        const stats_ptr: [*]u32 = @ptrCast(@alignCast(msg0(self.stats_buf, sels.contents)));

        var current_in_queue = self.redex_queue_a;
        var current_out_queue = self.redex_queue_b;

        var step: u32 = 0;
        while (step < max_steps and current_count > 0) : (step += 1) {
            in_count_ptr.* = current_count;
            stats_ptr[0] = 0; // interactions
            stats_ptr[1] = 0; // allocations
            stats_ptr[2] = 0; // out_count
            stats_ptr[3] = @intCast(self.redex_capacity);

            const cmd_buf = msg0(self.queue, sels.commandBuffer);
            if (cmd_buf == null) return MetalError.CommandFailed;

            const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
            if (encoder == null) return MetalError.CommandFailed;

            const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
            setPipeline(encoder, sels.setComputePipelineState, self.reduce_step_pipeline);

            msgV3(encoder, sels.setBuffer_offset_atIndex, self.gpu_heap, @as(NSUInteger, 0), @as(NSUInteger, 0));
            msgV3(encoder, sels.setBuffer_offset_atIndex, current_in_queue, @as(NSUInteger, 0), @as(NSUInteger, 1));
            msgV3(encoder, sels.setBuffer_offset_atIndex, current_out_queue, @as(NSUInteger, 0), @as(NSUInteger, 2));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.alloc_ptr_buf, @as(NSUInteger, 0), @as(NSUInteger, 3));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_count_buf, @as(NSUInteger, 0), @as(NSUInteger, 4));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.stats_buf, @as(NSUInteger, 8), @as(NSUInteger, 5));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.stats_buf, @as(NSUInteger, 0), @as(NSUInteger, 6));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.tmpl_terms_buf, @as(NSUInteger, 0), @as(NSUInteger, 7));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.tmpl_info_buf, @as(NSUInteger, 0), @as(NSUInteger, 8));
            msgV3(encoder, sels.setBuffer_offset_atIndex, self.heap_locks, @as(NSUInteger, 0), @as(NSUInteger, 9));

            const threads_per_group = @min(self.max_threads, current_count);
            const num_groups = (current_count + threads_per_group - 1) / threads_per_group;

            msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

            msgV0(encoder, sels.endEncoding);
            msgV0(cmd_buf, sels.commit);
            msgV0(cmd_buf, sels.waitUntilCompleted);

            total_stats.interactions += stats_ptr[0];
            total_stats.allocations += stats_ptr[1];

            current_count = stats_ptr[2];
            const tmp = current_in_queue;
            current_in_queue = current_out_queue;
            current_out_queue = tmp;
        }

        total_stats.new_redexes = current_count;
        return total_stats;
    }

    /// Upload heap data to GPU-resident buffer
    pub fn uploadGpuHeap(self: *Self, data: []const u64) MetalError!void {
        if (self.gpu_heap_ptr == null) return MetalError.BufferAllocFailed;
        if (data.len > self.gpu_heap_capacity) return MetalError.BufferAllocFailed;
        @memcpy(self.gpu_heap_ptr.?[0..data.len], data);
    }

    /// Download heap data from GPU-resident buffer
    pub fn downloadGpuHeap(self: *Self, data: []u64) MetalError!void {
        if (self.gpu_heap_ptr == null) return MetalError.BufferAllocFailed;
        if (data.len > self.gpu_heap_capacity) return MetalError.BufferAllocFailed;
        @memcpy(data, self.gpu_heap_ptr.?[0..data.len]);
    }

    /// Get current GPU heap pointer for direct access
    pub fn getGpuHeapPtr(self: *Self) ?[*]u64 {
        return self.gpu_heap_ptr;
    }

    // =========================================================================
    // ASYNC PIPELINING (Triple Buffering)
    // =========================================================================

    /// Begin async batch - returns command buffer index
    pub fn beginAsyncBatch(self: *Self) MetalError!usize {
        const idx = self.async_current;
        self.async_cmd_bufs[idx] = msg0(self.queue, sels.commandBuffer);
        if (self.async_cmd_bufs[idx] == null) return MetalError.CommandFailed;
        return idx;
    }

    /// Encode async heterogeneous interaction (non-blocking)
    pub fn encodeAsyncInteract(
        self: *Self,
        cmd_idx: usize,
        redex_count: usize,
    ) MetalError!void {
        if (self.interact_heterogeneous_pipeline == null) return MetalError.PipelineCreateFailed;

        const cmd_buf = self.async_cmd_bufs[cmd_idx];
        if (cmd_buf == null) return MetalError.CommandFailed;

        const encoder = msg0(cmd_buf, sels.computeCommandEncoder);
        if (encoder == null) return MetalError.CommandFailed;

        const setPipeline: *const fn (id, SEL, id) callconv(.c) void = @ptrCast(&objc_msgSend);
        setPipeline(encoder, sels.setComputePipelineState, self.interact_heterogeneous_pipeline);

        msgV3(encoder, sels.setBuffer_offset_atIndex, self.gpu_heap, @as(NSUInteger, 0), @as(NSUInteger, 0));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_queue_a, @as(NSUInteger, 0), @as(NSUInteger, 1));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.alloc_ptr_buf, @as(NSUInteger, 0), @as(NSUInteger, 2));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_queue_b, @as(NSUInteger, 0), @as(NSUInteger, 3));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.redex_count_buf, @as(NSUInteger, 0), @as(NSUInteger, 4));
        msgV3(encoder, sels.setBuffer_offset_atIndex, self.stats_buf, @as(NSUInteger, 0), @as(NSUInteger, 5));

        const threads_per_group = @min(self.max_threads, redex_count);
        const num_groups = (redex_count + threads_per_group - 1) / threads_per_group;

        msgDispatch(encoder, sels.dispatchThreadgroups_threadsPerThreadgroup, .{ .width = num_groups, .height = 1, .depth = 1 }, .{ .width = threads_per_group, .height = 1, .depth = 1 });

        msgV0(encoder, sels.endEncoding);
    }

    /// Commit async batch (non-blocking)
    pub fn commitAsyncBatch(self: *Self, cmd_idx: usize) void {
        const cmd_buf = self.async_cmd_bufs[cmd_idx];
        if (cmd_buf != null) {
            msgV0(cmd_buf, sels.commit);
            self.async_pending += 1;
        }
        self.async_current = (self.async_current + 1) % 3;
    }

    /// Wait for all async batches to complete
    pub fn waitAsyncBatches(self: *Self) void {
        for (&self.async_cmd_bufs) |cmd_buf| {
            if (cmd_buf != null) {
                msgV0(cmd_buf, sels.waitUntilCompleted);
            }
        }
        self.async_pending = 0;
    }

    /// Async pipelined reduction - overlaps upload/compute/download
    /// Uses triple buffering for maximum throughput
    pub fn gpuReduceAsyncPipelined(
        self: *Self,
        batches: []const []const [2]u32,
        initial_alloc: u32,
    ) MetalError!GpuReductionStats {
        if (batches.len == 0) return .{ .interactions = 0, .allocations = 0, .new_redexes = 0 };

        var total_stats = GpuReductionStats{
            .interactions = 0,
            .allocations = 0,
            .new_redexes = 0,
        };

        // Initialize allocation pointer
        const alloc_ptr: *u32 = @ptrCast(@alignCast(msg0(self.alloc_ptr_buf, sels.contents)));
        alloc_ptr.* = initial_alloc;

        // Triple-buffer pipelining
        for (batches, 0..) |batch, i| {
            // Upload current batch to redex queue
            const redex_ptr = self.redex_ptr_a orelse return MetalError.BufferAllocFailed;
            @memcpy(redex_ptr[0..batch.len], batch);

            // Initialize stats
            const stats_ptr: [*]u32 = @ptrCast(@alignCast(msg0(self.stats_buf, sels.contents)));
            stats_ptr[0] = 0;
            stats_ptr[1] = 0;

            const out_count_ptr: *u32 = @ptrCast(@alignCast(msg0(self.redex_count_buf, sels.contents)));
            out_count_ptr.* = 0;

            // Begin and encode async batch
            const cmd_idx = try self.beginAsyncBatch();
            try self.encodeAsyncInteract(cmd_idx, batch.len);
            self.commitAsyncBatch(cmd_idx);

            // For pipelining, wait every 3rd batch (or at end)
            if ((i + 1) % 3 == 0 or i == batches.len - 1) {
                self.waitAsyncBatches();

                // Accumulate stats
                total_stats.interactions += stats_ptr[0];
                total_stats.allocations += stats_ptr[1];
                total_stats.new_redexes = out_count_ptr.*;
            }
        }

        return total_stats;
    }
};

/// Statistics from GPU reduction
pub const GpuReductionStats = struct {
    interactions: u32,
    allocations: u32,
    new_redexes: u32,
};

// =============================================================================
// Global GPU Instance
// =============================================================================

var global_gpu: ?MetalGPU = null;

pub fn isAvailable() bool {
    if (!is_supported) return false;
    return MTLCreateSystemDefaultDevice() != null;
}

pub fn init() MetalError!void {
    if (global_gpu != null) return;
    global_gpu = try MetalGPU.init();
}

pub fn deinit() void {
    if (global_gpu) |*gpu| {
        gpu.deinit();
        global_gpu = null;
    }
}

pub fn getGPU() ?*MetalGPU {
    return if (global_gpu) |*gpu| gpu else null;
}

// =============================================================================
// CPU SIMD Fallback
// =============================================================================

pub fn cpuBatchAdd(a: []const u32, b: []const u32, results: []u32) void {
    const Vec8 = @Vector(8, u32);
    const len = @min(a.len, @min(b.len, results.len));

    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: Vec8 = a[i..][0..8].*;
        const vb: Vec8 = b[i..][0..8].*;
        results[i..][0..8].* = va + vb;
    }
    while (i < len) : (i += 1) {
        results[i] = a[i] + b[i];
    }
}

pub fn cpuBatchMul(a: []const u32, b: []const u32, results: []u32) void {
    const Vec8 = @Vector(8, u32);
    const len = @min(a.len, @min(b.len, results.len));

    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: Vec8 = a[i..][0..8].*;
        const vb: Vec8 = b[i..][0..8].*;
        results[i..][0..8].* = va * vb;
    }
    while (i < len) : (i += 1) {
        results[i] = a[i] * b[i];
    }
}

// =============================================================================
// High-Level API (GPU with CPU fallback)
// =============================================================================

pub fn batchAdd(a: []const u32, b: []const u32, results: []u32) void {
    if (global_gpu) |*gpu| {
        gpu.gpuBatchAdd(a, b, results) catch {
            cpuBatchAdd(a, b, results);
            return;
        };
    } else {
        cpuBatchAdd(a, b, results);
    }
}

pub fn batchMul(a: []const u32, b: []const u32, results: []u32) void {
    if (global_gpu) |*gpu| {
        gpu.gpuBatchMul(a, b, results) catch {
            cpuBatchMul(a, b, results);
            return;
        };
    } else {
        cpuBatchMul(a, b, results);
    }
}

// =============================================================================
// Benchmark
// =============================================================================

pub const BenchResult = struct {
    ops: u64,
    ns: u64,
};

pub fn benchGPUInit() BenchResult {
    if (!is_supported) return .{ .ops = 0, .ns = 1 };

    var timer = std.time.Timer.start() catch return .{ .ops = 0, .ns = 1 };
    const device = MTLCreateSystemDefaultDevice();
    if (device == null) return .{ .ops = 0, .ns = 1 };

    return .{ .ops = 1, .ns = timer.read() };
}

// =============================================================================
// Tests
// =============================================================================

test "cpu batch add" {
    var a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var results: [8]u32 = undefined;

    cpuBatchAdd(&a, &b, &results);

    for (0..8) |i| {
        try std.testing.expectEqual(a[i] + b[i], results[i]);
    }
}

test "cpu batch mul" {
    var a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b = [_]u32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var results: [8]u32 = undefined;

    cpuBatchMul(&a, &b, &results);

    for (0..8) |i| {
        try std.testing.expectEqual(a[i] * b[i], results[i]);
    }
}
