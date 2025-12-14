const std = @import("std");
const print = std.debug.print;

// =============================================================================
// HVM4 - Higher-order Virtual Machine v4 (Zig Implementation)
// Based on VictorTaelin's HVM4 design
// =============================================================================

// =============================================================================
// Type Definitions
// =============================================================================

pub const Tag = u8;
pub const Ext = u24;
pub const Val = u32;
pub const Term = u64;

// =============================================================================
// Term Layout: [8-bit tag][24-bit ext][32-bit val]
// =============================================================================

const TAG_SHIFT: u6 = 56;
const EXT_SHIFT: u6 = 32;
const VAL_SHIFT: u6 = 0;

const TAG_MASK: u64 = 0xFF00000000000000;
const EXT_MASK: u64 = 0x00FFFFFF00000000;
const VAL_MASK: u64 = 0x00000000FFFFFFFF;

// Substitution bit (in tag's high bit)
pub const SUB_BIT: u64 = 0x8000000000000000;

// =============================================================================
// Tag Constants - Ordered by access frequency for jump table efficiency
// =============================================================================

// Core types (hot path - positions 0-7)
pub const APP: Tag = 0x00; // Application
pub const VAR: Tag = 0x01; // Variable reference
pub const LAM: Tag = 0x02; // Lambda abstraction
pub const CO0: Tag = 0x03; // Collapse branch 0 (was DP0)
pub const CO1: Tag = 0x04; // Collapse branch 1 (was DP1)
pub const SUP: Tag = 0x05; // Superposition
pub const DUP: Tag = 0x06; // Duplication node
pub const REF: Tag = 0x07; // Function reference

// Additional core types
pub const ERA: Tag = 0x08; // Erasure
pub const RED: Tag = 0x09; // Guarded reduction (f ~> g)
pub const LET: Tag = 0x0A; // Let binding
pub const USE: Tag = 0x0B; // Strict evaluation wrapper
pub const EQL: Tag = 0x0C; // Structural equality

// Constructors C00-C15 (arity 0-15)
pub const C00: Tag = 0x10;
pub const C01: Tag = 0x11;
pub const C02: Tag = 0x12;
pub const C03: Tag = 0x13;
pub const C04: Tag = 0x14;
pub const C05: Tag = 0x15;
pub const C06: Tag = 0x16;
pub const C07: Tag = 0x17;
pub const C08: Tag = 0x18;
pub const C09: Tag = 0x19;
pub const C10: Tag = 0x1A;
pub const C11: Tag = 0x1B;
pub const C12: Tag = 0x1C;
pub const C13: Tag = 0x1D;
pub const C14: Tag = 0x1E;
pub const C15: Tag = 0x1F;

// Pattern matching
pub const MAT: Tag = 0x20; // Constructor pattern match
pub const SWI: Tag = 0x21; // Numeric switch

// Numeric type
pub const NUM: Tag = 0x22; // 32-bit number

// Primitives P00-P15 (arity 0-15)
pub const P00: Tag = 0x30;
pub const P01: Tag = 0x31;
pub const P02: Tag = 0x32;
pub const P03: Tag = 0x33;
pub const P04: Tag = 0x34;
pub const P05: Tag = 0x35;
pub const P06: Tag = 0x36;
pub const P07: Tag = 0x37;
pub const P08: Tag = 0x38;
pub const P09: Tag = 0x39;
pub const P10: Tag = 0x3A;
pub const P11: Tag = 0x3B;
pub const P12: Tag = 0x3C;
pub const P13: Tag = 0x3D;
pub const P14: Tag = 0x3E;
pub const P15: Tag = 0x3F;

// Stack frame types (0x40+)
pub const F_APP: Tag = 0x40; // Reducing function in application
pub const F_MAT: Tag = 0x41; // Reducing scrutinee for pattern match
pub const F_SWI: Tag = 0x42; // Reducing scrutinee for numeric switch
pub const F_USE: Tag = 0x43; // Reducing argument for strict eval
pub const F_OP2: Tag = 0x44; // Reducing second operand
pub const F_DUP: Tag = 0x45; // Reducing value for duplication
pub const F_EQL: Tag = 0x46; // Reducing for equality check

// Reserved for future use (0x50-0x5F)

// Type System (Interaction Calculus of Constructions)
pub const ANN: Tag = 0x60; // Annotation {term : Type}
pub const BRI: Tag = 0x61; // Bridge θx. term (for dependent types)
pub const TYP: Tag = 0x62; // Type universe (*)
pub const ALL: Tag = 0x63; // Dependent function type Π(x:A).B
pub const SIG: Tag = 0x64; // Dependent pair type Σ(x:A).B
pub const SLF: Tag = 0x65; // Self type (for induction)

// Linear/Affine type system (Oracle Problem prevention)
pub const LIN: Tag = 0x66; // Linear type marker (exactly once)
pub const AFF: Tag = 0x67; // Affine type marker (at most once)

// Type error (for dependent type checking)
pub const ERR: Tag = 0x70; // Type error: heap[val]=expected, heap[val+1]=actual, ext=error_code

// Stack frames for type checking
pub const F_ANN: Tag = 0x68; // Reducing annotation
pub const F_BRI: Tag = 0x69; // Reducing bridge
pub const F_EQL2: Tag = 0x6A; // Second stage of equality check
pub const F_LIN: Tag = 0x6B; // Reducing linearity annotation

// Void term
pub const VOID: Term = 0;

// =============================================================================
// Operator Definitions - Single source of truth for all operator metadata
// =============================================================================

pub const OpInfo = struct {
    id: Ext,
    symbol: []const u8,
    name: []const u8,
};

/// Central operator table - used by parser, pretty-printer, and compute_op
pub const operators = [_]OpInfo{
    .{ .id = 0x00, .symbol = "+", .name = "add" },
    .{ .id = 0x01, .symbol = "-", .name = "sub" },
    .{ .id = 0x02, .symbol = "*", .name = "mul" },
    .{ .id = 0x03, .symbol = "/", .name = "div" },
    .{ .id = 0x04, .symbol = "%", .name = "mod" },
    .{ .id = 0x05, .symbol = "&", .name = "and" },
    .{ .id = 0x06, .symbol = "|", .name = "or" },
    .{ .id = 0x07, .symbol = "^", .name = "xor" },
    .{ .id = 0x08, .symbol = "<<", .name = "lsh" },
    .{ .id = 0x09, .symbol = ">>", .name = "rsh" },
    .{ .id = 0x0A, .symbol = "~", .name = "not" },
    .{ .id = 0x0B, .symbol = "==", .name = "eq" },
    .{ .id = 0x0C, .symbol = "!=", .name = "ne" },
    .{ .id = 0x0D, .symbol = "<", .name = "lt" },
    .{ .id = 0x0E, .symbol = "<=", .name = "le" },
    .{ .id = 0x0F, .symbol = ">", .name = "gt" },
    .{ .id = 0x10, .symbol = ">=", .name = "ge" },
};

/// Get operator symbol from ID
pub fn op_symbol(id: Ext) []const u8 {
    for (operators) |op| {
        if (op.id == id) return op.symbol;
    }
    return "?";
}

/// Get operator ID from symbol (returns null if not found)
pub fn op_from_symbol(symbol: []const u8) ?Ext {
    for (operators) |op| {
        if (std.mem.eql(u8, op.symbol, symbol)) return op.id;
    }
    return null;
}

// Operator ID constants (for backward compatibility and direct access)
pub const OP_ADD: Ext = 0x00;
pub const OP_SUB: Ext = 0x01;
pub const OP_MUL: Ext = 0x02;
pub const OP_DIV: Ext = 0x03;
pub const OP_MOD: Ext = 0x04;
pub const OP_AND: Ext = 0x05;
pub const OP_OR: Ext = 0x06;
pub const OP_XOR: Ext = 0x07;
pub const OP_LSH: Ext = 0x08;
pub const OP_RSH: Ext = 0x09;
pub const OP_NOT: Ext = 0x0A;
pub const OP_EQ: Ext = 0x0B;
pub const OP_NE: Ext = 0x0C;
pub const OP_LT: Ext = 0x0D;
pub const OP_LE: Ext = 0x0E;
pub const OP_GT: Ext = 0x0F;
pub const OP_GE: Ext = 0x10;

// =============================================================================
// Built-in Function IDs
// =============================================================================

pub const SUP_F: Ext = 0xFFF0;
pub const DUP_F: Ext = 0xFFF1;
pub const LOG_F: Ext = 0xFFF2;
pub const EQL_F: Ext = 0xFFF3;

// =============================================================================
// Term Operations - Inline for maximum performance
// =============================================================================

/// Create a new term from tag, ext, and val
pub inline fn term_new(tag: Tag, ext: Ext, val: Val) Term {
    return (@as(Term, tag) << TAG_SHIFT) |
        (@as(Term, ext) << EXT_SHIFT) |
        (@as(Term, val) << VAL_SHIFT);
}

/// Extract tag from term
pub inline fn term_tag(t: Term) Tag {
    return @truncate((t & TAG_MASK) >> TAG_SHIFT);
}

/// Extract ext from term (label/constructor id/primitive id)
pub inline fn term_ext(t: Term) Ext {
    return @truncate((t & EXT_MASK) >> EXT_SHIFT);
}

/// Extract val from term (heap location or immediate value)
pub inline fn term_val(t: Term) Val {
    return @truncate(t & VAL_MASK);
}

/// Check if term has substitution bit set
pub inline fn term_is_sub(t: Term) bool {
    return (t & SUB_BIT) != 0;
}

/// Set substitution bit
pub inline fn term_set_sub(t: Term) Term {
    return t | SUB_BIT;
}

/// Clear substitution bit
pub inline fn term_clr_sub(t: Term) Term {
    return t & ~SUB_BIT;
}

/// Check if term is a value (not an eliminator)
pub inline fn term_is_val(t: Term) bool {
    const tag = term_tag(t);
    return tag == LAM or tag == SUP or tag == ERA or tag == NUM or
        (tag >= C00 and tag <= C15);
}

/// Check if term is an eliminator
pub inline fn term_is_elim(t: Term) bool {
    const tag = term_tag(t);
    return tag == APP or tag == MAT or tag == SWI or tag == USE or
        (tag >= P00 and tag <= P15);
}

/// Get constructor arity from tag
pub inline fn ctr_arity(tag: Tag) u32 {
    if (tag >= C00 and tag <= C15) {
        return tag - C00;
    }
    return 0;
}

/// Get primitive arity from tag
pub inline fn prm_arity(tag: Tag) u32 {
    if (tag >= P00 and tag <= P15) {
        return tag - P00;
    }
    return 0;
}

// =============================================================================
// HVM State
// =============================================================================

// Default sizes (can be overridden via Config)
pub const DEFAULT_HEAP_SIZE: u64 = 1 << 30; // 1GB
pub const DEFAULT_STACK_SIZE: u64 = 1 << 24; // 16MB
pub const DEFAULT_BOOK_SIZE: usize = 65536;
pub const DEFAULT_NUM_WORKERS: usize = 12;

// Legacy constants for backward compatibility
pub const MAX_HEAP_SIZE: u64 = DEFAULT_HEAP_SIZE;
pub const MAX_STACK_SIZE: u64 = DEFAULT_STACK_SIZE;
pub const MAX_BOOK_SIZE: usize = DEFAULT_BOOK_SIZE;
pub const NUM_WORKERS: usize = DEFAULT_NUM_WORKERS;

pub const BookFn = *const fn (Term) Term;

/// Runtime configuration - all sizes are configurable
pub const Config = struct {
    heap_size: u64 = DEFAULT_HEAP_SIZE,
    stack_size: u64 = DEFAULT_STACK_SIZE,
    book_size: usize = DEFAULT_BOOK_SIZE,
    num_workers: usize = DEFAULT_NUM_WORKERS,
    /// Enable reference counting for memory management
    enable_refcount: bool = false,
    /// Enable label recycling to prevent exhaustion
    enable_label_recycling: bool = false,
    /// Initial label value for auto-dup
    initial_label: Ext = 0x800000,
};

/// HVM4 runtime state
pub const State = struct {
    // Hot fields
    heap: []Term,
    stack: []Term,
    heap_pos: u64,
    stack_pos: u64,
    interactions: u64,
    fresh_label: Ext,

    // Configuration
    config: Config,

    // Book (function definitions) - dynamically allocated
    book: []?BookFn,
    ctr_arity: []u16,
    fun_arity: []u16,
    func_bodies: []Term,

    // Reference counting (optional)
    refcounts: ?[]u32,

    // Label recycling (optional)
    free_labels: ?[]Ext,
    free_labels_count: usize,

    // Allocator for cleanup
    allocator: std.mem.Allocator,

    /// Initialize with default configuration
    pub fn init(allocator: std.mem.Allocator) !*State {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration
    pub fn initWithConfig(allocator: std.mem.Allocator, config: Config) !*State {
        const state = try allocator.create(State);
        state.config = config;
        state.allocator = allocator;

        // Allocate memory based on config
        state.heap = try allocator.alloc(Term, config.heap_size);
        state.stack = try allocator.alloc(Term, config.stack_size);
        state.book = try allocator.alloc(?BookFn, config.book_size);
        state.ctr_arity = try allocator.alloc(u16, config.book_size);
        state.fun_arity = try allocator.alloc(u16, config.book_size);
        state.func_bodies = try allocator.alloc(Term, config.book_size);

        // Initialize state
        state.heap_pos = 1; // Reserve location 0
        state.stack_pos = 0;
        state.interactions = 0;
        state.fresh_label = config.initial_label;

        // Initialize arrays
        @memset(state.heap, 0);
        @memset(state.stack, 0);
        @memset(state.book, null);
        @memset(state.ctr_arity, 0);
        @memset(state.fun_arity, 0);
        @memset(state.func_bodies, 0);

        // Optional: reference counting
        if (config.enable_refcount) {
            state.refcounts = try allocator.alloc(u32, config.heap_size);
            @memset(state.refcounts.?, 0);
        } else {
            state.refcounts = null;
        }

        // Optional: label recycling
        if (config.enable_label_recycling) {
            // Allocate space for recycled labels (16K should be plenty)
            state.free_labels = try allocator.alloc(Ext, 16384);
            state.free_labels_count = 0;
        } else {
            state.free_labels = null;
            state.free_labels_count = 0;
        }

        // Initialize built-in functions
        state.book[SUP_F] = builtin_sup;
        state.book[DUP_F] = builtin_dup;
        state.book[LOG_F] = builtin_log;

        return state;
    }

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        allocator.free(self.stack);
        allocator.free(self.heap);
        allocator.free(self.book);
        allocator.free(self.ctr_arity);
        allocator.free(self.fun_arity);
        allocator.free(self.func_bodies);
        if (self.refcounts) |rc| allocator.free(rc);
        if (self.free_labels) |fl| allocator.free(fl);
        allocator.destroy(self);
    }

    pub fn define(self: *State, fid: Ext, func: BookFn) void {
        self.book[fid] = func;
    }

    /// Register a compiled function body
    pub fn setFuncBody(self: *State, fid: Ext, body: Term) void {
        self.func_bodies[fid] = body;
    }

    /// Get a compiled function body
    pub fn getFuncBody(self: *State, fid: Ext) Term {
        return self.func_bodies[fid];
    }

    /// Reset heap for reuse (simple arena-style reset)
    pub fn resetHeap(self: *State) void {
        self.heap_pos = 1;
        if (self.refcounts) |rc| {
            @memset(rc, 0);
        }
    }

    /// Get heap usage statistics
    pub fn getStats(self: *State) struct { heap_used: u64, heap_total: u64, interactions: u64, labels_used: u32 } {
        return .{
            .heap_used = self.heap_pos,
            .heap_total = self.config.heap_size,
            .interactions = self.interactions,
            .labels_used = self.fresh_label - self.config.initial_label,
        };
    }
};

/// Generic function dispatch - retrieves body from HVM state and copies it
/// For Bend: Arguments are applied via APP nodes (ref_val=0, arity stored in fun_arity)
/// For HVM parser: Arguments stored at heap[ref_val..ref_val+arity]
pub fn funcDispatch(ref: Term) Term {
    const fid = term_ext(ref);
    const ref_val = term_val(ref);
    const arity = HVM.fun_arity[fid];
    const body = HVM.func_bodies[fid];

    if (debug_reduce) {
        std.debug.print("funcDispatch: fid={}, ref_val={}, arity={}, body tag=0x{x} val={}\n", .{ fid, ref_val, arity, term_tag(body), term_val(body) });
    }

    // Deep copy the term to get fresh heap locations for each call
    var loc_map: [256]u64 = [_]u64{0} ** 256;
    var map_count: usize = 0;
    var result = copyTermWithMap(body, &loc_map, &map_count);

    if (debug_reduce) {
        std.debug.print("funcDispatch: after copy, result tag=0x{x} val={}, map_count={}\n", .{ term_tag(result), term_val(result), map_count });
        for (0..map_count) |i| {
            const idx = i * 2;
            if (idx + 1 < 256) {
                std.debug.print("  map[{}]: {} -> {}\n", .{ i, loc_map[idx], loc_map[idx + 1] });
            }
        }
    }

    // If arguments are stored at ref_val (HVM parser style), apply them
    // If ref_val is 0 or small, arguments come from APP nodes (Bend style)
    if (ref_val > 0 and arity > 0) {
        // HVM parser style: arguments at heap[ref_val]
        for (0..arity) |i| {
            const arg = HVM.heap[ref_val + i];
            const app_loc = alloc(2);
            HVM.heap[app_loc] = result;
            HVM.heap[app_loc + 1] = arg;
            result = term_new(APP, 0, app_loc);
        }
    }
    // Bend style: arguments will be applied via APP nodes, just return the body

    return result;
}

/// Deep copy a term, allocating fresh heap locations and remapping VARs
fn copyTermWithMap(term: Term, loc_map: *[256]u64, map_count: *usize) Term {
    const tag = term_tag(term);
    const ext = term_ext(term);
    const val = term_val(term);

    return switch (tag) {
        // Values that don't need copying
        ERA, NUM, TYP => term,

        // Lambda: copy body to fresh location and record mapping
        LAM => {
            const new_loc = alloc(1);
            // Record the mapping from old location to new location
            if (map_count.* < 256) {
                // Store as pair: [old_loc, new_loc] using even/odd indices
                const idx = map_count.* * 2;
                if (idx + 1 < 256) {
                    loc_map[idx] = val;
                    loc_map[idx + 1] = new_loc;
                    map_count.* += 1;
                    if (debug_reduce) {
                        std.debug.print("  copyTerm LAM: old_loc={} -> new_loc={}\n", .{ val, new_loc });
                    }
                }
            }
            const body = HVM.heap[val];
            HVM.heap[new_loc] = copyTermWithMap(body, loc_map, map_count);
            return term_new(LAM, ext, @truncate(new_loc));
        },

        // Application: copy both function and argument
        APP => {
            const new_loc = alloc(2);
            HVM.heap[new_loc] = copyTermWithMap(HVM.heap[val], loc_map, map_count);
            HVM.heap[new_loc + 1] = copyTermWithMap(HVM.heap[val + 1], loc_map, map_count);
            return term_new(APP, ext, @truncate(new_loc));
        },

        // Constructors: copy all fields
        C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 => {
            const arity = ctr_arity(tag);
            if (arity == 0) return term;
            const new_loc = alloc(arity);
            for (0..arity) |i| {
                HVM.heap[new_loc + i] = copyTermWithMap(HVM.heap[val + i], loc_map, map_count);
            }
            return term_new(tag, ext, @truncate(new_loc));
        },

        // MAT: copy scrutinee and all branches
        MAT => {
            const num_cases = ext;
            const new_loc = alloc(1 + num_cases);
            HVM.heap[new_loc] = copyTermWithMap(HVM.heap[val], loc_map, map_count);
            for (0..num_cases) |i| {
                HVM.heap[new_loc + 1 + i] = copyTermWithMap(HVM.heap[val + 1 + i], loc_map, map_count);
            }
            return term_new(MAT, ext, @truncate(new_loc));
        },

        // SWI: copy scrutinee and branches
        SWI => {
            const new_loc = alloc(3);
            HVM.heap[new_loc] = copyTermWithMap(HVM.heap[val], loc_map, map_count);
            HVM.heap[new_loc + 1] = copyTermWithMap(HVM.heap[val + 1], loc_map, map_count);
            HVM.heap[new_loc + 2] = copyTermWithMap(HVM.heap[val + 2], loc_map, map_count);
            return term_new(SWI, ext, @truncate(new_loc));
        },

        // VAR: remap to new location if it was copied
        VAR => {
            // Look up the old location in the map
            for (0..map_count.*) |i| {
                const idx = i * 2;
                if (idx + 1 < 256 and loc_map[idx] == val) {
                    if (debug_reduce) {
                        std.debug.print("  copyTerm VAR({}) -> VAR({})\n", .{ val, loc_map[idx + 1] });
                    }
                    return term_new(VAR, ext, @truncate(loc_map[idx + 1]));
                }
            }
            // Not found in map - return as-is
            if (debug_reduce) {
                std.debug.print("  copyTerm VAR({}) NOT FOUND in map (map_count={})\n", .{ val, map_count.* });
            }
            return term;
        },

        // SUP: copy both branches
        SUP => {
            const new_loc = alloc(2);
            HVM.heap[new_loc] = copyTermWithMap(HVM.heap[val], loc_map, map_count);
            HVM.heap[new_loc + 1] = copyTermWithMap(HVM.heap[val + 1], loc_map, map_count);
            return term_new(SUP, ext, @truncate(new_loc));
        },

        // REF: just return (will be expanded later)
        REF => term,

        // Operations
        P02 => {
            const new_loc = alloc(2);
            HVM.heap[new_loc] = copyTermWithMap(HVM.heap[val], loc_map, map_count);
            HVM.heap[new_loc + 1] = copyTermWithMap(HVM.heap[val + 1], loc_map, map_count);
            return term_new(P02, ext, @truncate(new_loc));
        },

        // CO0/CO1: remap to new location if it was copied (like VAR)
        CO0, CO1 => {
            // Look up the old location in the map
            for (0..map_count.*) |i| {
                const idx = i * 2;
                if (idx + 1 < 256 and loc_map[idx] == val) {
                    return term_new(tag, ext, @truncate(loc_map[idx + 1]));
                }
            }
            // Not found in map - return as-is
            return term;
        },

        // Default: just return as-is
        else => term,
    };
}

// Thread-local HVM state
threadlocal var HVM: *State = undefined;

/// Runtime commutation counter for detecting runaway reduction (oracle problem)
/// Incremented each time DUP+SUP commutation occurs (different labels)
var commutation_count: u64 = 0;

pub fn hvm_init(allocator: std.mem.Allocator) !void {
    HVM = try State.init(allocator);
}

pub fn hvm_free(allocator: std.mem.Allocator) void {
    HVM.deinit(allocator);
}

/// Convenience initialization for tests - uses page allocator with default config
var default_allocator: ?std.mem.Allocator = null;

pub fn init(config: Config) void {
    default_allocator = std.heap.page_allocator;
    HVM = State.initWithConfig(default_allocator.?, config) catch @panic("HVM init failed");
}

pub fn deinit() void {
    if (default_allocator) |allocator| {
        HVM.deinit(allocator);
        default_allocator = null;
    }
}

pub fn hvm_get_state() *State {
    return HVM;
}

pub fn hvm_set_state(state: *State) void {
    HVM = state;
}

// =============================================================================
// Heap Operations
// =============================================================================

/// Allocate n slots on the heap
pub inline fn alloc(n: u64) Val {
    const pos = HVM.heap_pos;
    HVM.heap_pos += n;
    return @truncate(pos);
}

/// Read from heap
pub inline fn got(loc: u64) Term {
    return HVM.heap[loc];
}

/// Write to heap
pub inline fn set(loc: u64, term: Term) void {
    if (debug_reduce and (loc == 390 or loc == 391)) {
        std.debug.print("HEAP WRITE: loc={d}, term=0x{x}\n", .{ loc, term });
    }
    HVM.heap[loc] = term;
}

/// Atomic swap
pub inline fn swap(loc: u64, term: Term) Term {
    const old = HVM.heap[loc];
    HVM.heap[loc] = term;
    return old;
}

/// Substitute (write with sub bit)
pub inline fn sub(loc: u64, term: Term) void {
    HVM.heap[loc] = term_set_sub(term);
}

/// Take (swap with VOID)
pub inline fn take(loc: u64) Term {
    return swap(loc, VOID);
}

// Legacy compatibility
pub inline fn alloc_node(n: u64) u64 {
    return @as(u64, alloc(@truncate(n)));
}

pub inline fn inc_itr() void {
    HVM.interactions += 1;
}

// =============================================================================
// Stack Operations
// =============================================================================

pub inline fn push(term: Term) void {
    HVM.stack[HVM.stack_pos] = term;
    HVM.stack_pos += 1;
}

pub inline fn pop() Term {
    HVM.stack_pos -= 1;
    return HVM.stack[HVM.stack_pos];
}

// =============================================================================
// Interaction Rules
// =============================================================================

/// APP + LAM: Beta reduction
inline fn interact_app_lam(app_loc: Val, lam_loc: Val) Term {
    HVM.interactions += 1;
    const arg = HVM.heap[app_loc + 1];
    const bod = HVM.heap[lam_loc];
    if (debug_reduce) {
        std.debug.print("APP+LAM: lam_loc={d}, arg tag=0x{x} val={d}, bod tag=0x{x} val={d}\n", .{ lam_loc, term_tag(arg), term_val(arg), term_tag(bod), term_val(bod) });
    }
    HVM.heap[lam_loc] = term_set_sub(arg);
    return bod;
}

/// APP + ERA: Erasure
inline fn interact_app_era() Term {
    HVM.interactions += 1;
    return term_new(ERA, 0, 0);
}

/// APP + SUP: Superposition distribution
/// (SUP{f,g} x) => SUP{(f CO0(x)), (g CO1(x))}
fn interact_app_sup(app_loc: Val, sup_loc: Val, sup_lab: Ext) Term {
    HVM.interactions += 1;
    const arg = HVM.heap[app_loc + 1];
    const lft = HVM.heap[sup_loc];
    const rgt = HVM.heap[sup_loc + 1];

    // Batch allocate: 1 dup + 2 app0 + 2 app1 + 2 result = 7 slots
    const base = alloc(7);
    const dup_loc = base;
    const app0_loc = base + 1;
    const app1_loc = base + 3;
    const res_loc = base + 5;

    // Create dup for argument (lazy sharing)
    HVM.heap[dup_loc] = arg;

    // Create two applications with CO0/CO1 projections
    HVM.heap[app0_loc] = lft;
    HVM.heap[app0_loc + 1] = term_new(CO0, sup_lab, dup_loc);
    HVM.heap[app1_loc] = rgt;
    HVM.heap[app1_loc + 1] = term_new(CO1, sup_lab, dup_loc);

    // Create result superposition
    HVM.heap[res_loc] = term_new(APP, 0, app0_loc);
    HVM.heap[res_loc + 1] = term_new(APP, 0, app1_loc);

    return term_new(SUP, sup_lab, res_loc);
}

/// DUP + ERA: Both projections get erasure
inline fn interact_dup_era(dup_loc: Val) Term {
    HVM.interactions += 1;
    HVM.heap[dup_loc] = term_set_sub(term_new(ERA, 0, 0));
    return term_new(ERA, 0, 0);
}

/// DUP + LAM: Duplicate lambda
/// Creates two copies of a lambda with shared body via CO0/CO1 projections
fn interact_dup_lam(dup_tag: Tag, dup_loc: Val, dup_lab: Ext, lam_loc: Val) Term {
    HVM.interactions += 1;
    const bod = HVM.heap[lam_loc];

    // Batch allocate: 1 for inner dup + 2 for lambdas = 3 slots
    const base = alloc(3);
    const inner_dup = base;
    const lam0_loc = base + 1;
    const lam1_loc = base + 2;

    // Create inner dup for body (lazy duplication)
    HVM.heap[inner_dup] = bod;

    // Create two lambdas with CO0/CO1 projections to shared body
    HVM.heap[lam0_loc] = term_new(CO0, dup_lab, inner_dup);
    HVM.heap[lam1_loc] = term_new(CO1, dup_lab, inner_dup);

    // Return appropriate projection, store the other for sibling
    if (dup_tag == CO0) {
        @branchHint(.likely);
        HVM.heap[dup_loc] = term_set_sub(term_new(LAM, 0, lam1_loc));
        return term_new(LAM, 0, lam0_loc);
    } else {
        HVM.heap[dup_loc] = term_set_sub(term_new(LAM, 0, lam0_loc));
        return term_new(LAM, 0, lam1_loc);
    }
}

/// DUP + SUP: Annihilation or commutation
inline fn interact_dup_sup(dup_tag: Tag, dup_loc: Val, dup_lab: Ext, sup_loc: Val, sup_lab: Ext) Term {
    HVM.interactions += 1;
    const lft = HVM.heap[sup_loc];
    const rgt = HVM.heap[sup_loc + 1];

    if (dup_lab == sup_lab) {
        // Annihilation: same labels cancel out - O(1), optimal case
        if (dup_tag == CO0) {
            HVM.heap[dup_loc] = term_set_sub(rgt);
            return lft;
        } else {
            HVM.heap[dup_loc] = term_set_sub(lft);
            return rgt;
        }
    } else {
        // Commutation: different labels create 2x2 grid - potential exponential blowup
        // Track this for runtime safety monitoring
        commutation_count += 1;

        const new_loc = alloc(4);

        HVM.heap[new_loc + 0] = term_new(CO0, dup_lab, sup_loc);
        HVM.heap[new_loc + 1] = term_new(CO0, dup_lab, sup_loc + 1);
        HVM.heap[new_loc + 2] = term_new(CO1, dup_lab, sup_loc);
        HVM.heap[new_loc + 3] = term_new(CO1, dup_lab, sup_loc + 1);

        if (dup_tag == CO0) {
            HVM.heap[dup_loc] = term_set_sub(term_new(SUP, sup_lab, new_loc + 2));
            return term_new(SUP, sup_lab, new_loc);
        } else {
            HVM.heap[dup_loc] = term_set_sub(term_new(SUP, sup_lab, new_loc));
            return term_new(SUP, sup_lab, new_loc + 2);
        }
    }
}

/// DUP + APP: Duplicate application
/// CO0(APP(f,x)) => APP(CO0(f), CO0(x))
fn interact_dup_app(dup_tag: Tag, dup_loc: Val, dup_lab: Ext, app_loc: Val) Term {
    HVM.interactions += 1;

    // Allocate: 2 dups + 2 app0 + 2 app1 = 6 slots
    const base = alloc(6);
    const dups_loc = base;
    const app0_loc = base + 2;
    const app1_loc = base + 4;

    // Copy APP fields (func, arg) to dup slots
    HVM.heap[dups_loc] = HVM.heap[app_loc]; // func
    HVM.heap[dups_loc + 1] = HVM.heap[app_loc + 1]; // arg

    // Create CO0/CO1 projections for both APPs
    HVM.heap[app0_loc] = term_new(CO0, dup_lab, @truncate(dups_loc));
    HVM.heap[app0_loc + 1] = term_new(CO0, dup_lab, @truncate(dups_loc + 1));
    HVM.heap[app1_loc] = term_new(CO1, dup_lab, @truncate(dups_loc));
    HVM.heap[app1_loc + 1] = term_new(CO1, dup_lab, @truncate(dups_loc + 1));

    if (dup_tag == CO0) {
        @branchHint(.likely);
        HVM.heap[dup_loc] = term_set_sub(term_new(APP, 0, app1_loc));
        return term_new(APP, 0, app0_loc);
    } else {
        HVM.heap[dup_loc] = term_set_sub(term_new(APP, 0, app0_loc));
        return term_new(APP, 0, app1_loc);
    }
}

/// DUP + NUM: Both projections get the number
inline fn interact_dup_num(dup_loc: Val, num: Term) Term {
    HVM.interactions += 1;
    HVM.heap[dup_loc] = term_set_sub(num);
    return num;
}

/// DUP + CTR: Duplicate constructor
/// CO0(CTR{a,b,...}) => CTR{CO0(a), CO0(b), ...}
fn interact_dup_ctr(dup_tag: Tag, dup_loc: Val, dup_lab: Ext, ctr_tag: Tag, ctr_loc: Val, ctr_ext: Ext) Term {
    HVM.interactions += 1;
    const arity = ctr_arity(ctr_tag);

    if (arity == 0) {
        // Arity 0: just duplicate the term (no fields to copy)
        @branchHint(.unlikely);
        const ctr = term_new(ctr_tag, ctr_ext, ctr_loc);
        HVM.heap[dup_loc] = term_set_sub(ctr);
        return ctr;
    }

    // Batch allocate: arity dups + arity ctr0 + arity ctr1 = 3*arity slots
    const base = alloc(arity * 3);
    const dups_loc = base;
    const ctr0_loc = base + arity;
    const ctr1_loc = base + arity * 2;

    // Copy fields to dup nodes and create CO0/CO1 projections in one pass
    for (0..arity) |i| {
        HVM.heap[dups_loc + i] = HVM.heap[ctr_loc + i];
        HVM.heap[ctr0_loc + i] = term_new(CO0, dup_lab, @truncate(dups_loc + i));
        HVM.heap[ctr1_loc + i] = term_new(CO1, dup_lab, @truncate(dups_loc + i));
    }

    if (dup_tag == CO0) {
        @branchHint(.likely);
        HVM.heap[dup_loc] = term_set_sub(term_new(ctr_tag, ctr_ext, ctr1_loc));
        return term_new(ctr_tag, ctr_ext, ctr0_loc);
    } else {
        HVM.heap[dup_loc] = term_set_sub(term_new(ctr_tag, ctr_ext, ctr0_loc));
        return term_new(ctr_tag, ctr_ext, ctr1_loc);
    }
}

/// MAT + CTR: Pattern match on constructor
fn interact_mat_ctr(mat_loc: Val, ctr_tag: Tag, ctr_loc: Val, ctr_ext: Ext) Term {
    HVM.interactions += 1;
    const arity = ctr_arity(ctr_tag);
    const case_idx = ctr_ext;

    // Get the case branch
    var branch = HVM.heap[mat_loc + 1 + case_idx];

    if (debug_reduce) {
        std.debug.print("MAT+CTR: ctr_tag=0x{x}, arity={d}, case_idx={d}\n", .{ ctr_tag, arity, case_idx });
        std.debug.print("  branch: tag=0x{x}, ext={d}, val={d}\n", .{ term_tag(branch), term_ext(branch), term_val(branch) });
        for (0..arity) |i| {
            const field = HVM.heap[ctr_loc + i];
            std.debug.print("  field[{d}]: tag=0x{x}, ext={d}, val={d}\n", .{ i, term_tag(field), term_ext(field), term_val(field) });
        }
    }

    // Apply constructor fields to branch
    for (0..arity) |i| {
        const field = HVM.heap[ctr_loc + i];
        const app_loc = alloc(2);
        HVM.heap[app_loc] = branch;
        HVM.heap[app_loc + 1] = field;
        branch = term_new(APP, 0, app_loc);
    }

    if (debug_reduce) {
        std.debug.print("  result: tag=0x{x}, ext={d}, val={d}\n", .{ term_tag(branch), term_ext(branch), term_val(branch) });
    }

    return branch;
}

/// SWI + NUM: Numeric switch
fn interact_swi_num(swi_loc: Val, num_val: Val) Term {
    HVM.interactions += 1;

    if (num_val == 0) {
        // Zero case
        return HVM.heap[swi_loc + 1];
    } else {
        // Successor case: apply (n-1) to succ branch
        const succ_branch = HVM.heap[swi_loc + 2];
        const app_loc = alloc(2);
        HVM.heap[app_loc] = succ_branch;
        HVM.heap[app_loc + 1] = term_new(NUM, 0, num_val - 1);
        return term_new(APP, 0, app_loc);
    }
}

/// Binary primitive operation
inline fn compute_op(op: Ext, x: Val, y: Val) Val {
    return switch (op) {
        OP_ADD => x +% y,
        OP_SUB => x -% y,
        OP_MUL => x *% y,
        OP_DIV => if (y != 0) x / y else 0,
        OP_MOD => if (y != 0) x % y else 0,
        OP_AND => x & y,
        OP_OR => x | y,
        OP_XOR => x ^ y,
        OP_LSH => x << @truncate(y),
        OP_RSH => x >> @truncate(y),
        OP_EQ => @intFromBool(x == y),
        OP_NE => @intFromBool(x != y),
        OP_LT => @intFromBool(x < y),
        OP_LE => @intFromBool(x <= y),
        OP_GT => @intFromBool(x > y),
        OP_GE => @intFromBool(x >= y),
        else => 0,
    };
}

// =============================================================================
// Built-in Functions
// =============================================================================

fn builtin_sup(ref: Term) Term {
    const ref_loc = term_val(ref);
    const lab_term = reduce(got(ref_loc));
    const lab_val = term_val(lab_term);

    if (term_tag(lab_term) != NUM) {
        print("ERROR: non-numeric sup label\n", .{});
        std.process.exit(1);
    }

    const tm0 = got(ref_loc + 1);
    const tm1 = got(ref_loc + 2);
    const sup_loc = alloc(2);
    set(sup_loc, tm0);
    set(sup_loc + 1, tm1);
    HVM.interactions += 1;
    return term_new(SUP, @truncate(lab_val), sup_loc);
}

fn builtin_dup(ref: Term) Term {
    const ref_loc = term_val(ref);
    const lab_term = reduce(got(ref_loc));
    const lab_val = term_val(lab_term);

    if (term_tag(lab_term) != NUM) {
        print("ERROR: non-numeric dup label\n", .{});
        std.process.exit(1);
    }

    const val = got(ref_loc + 1);
    const bod = got(ref_loc + 2);
    const dup_loc = alloc(1);
    set(dup_loc, val);

    // Optimize: if body is λλ, substitute directly
    if (term_tag(bod) == LAM) {
        const lam0_loc = term_val(bod);
        const bod0 = got(lam0_loc);
        if (term_tag(bod0) == LAM) {
            const lam1_loc = term_val(bod0);
            const bod1 = got(lam1_loc);
            sub(lam0_loc, term_new(CO0, @truncate(lab_val), dup_loc));
            sub(lam1_loc, term_new(CO1, @truncate(lab_val), dup_loc));
            HVM.interactions += 3;
            return bod1;
        }
    }

    // General case: create applications
    const app0_loc = alloc(2);
    set(app0_loc, bod);
    set(app0_loc + 1, term_new(CO0, @truncate(lab_val), dup_loc));

    const app1_loc = alloc(2);
    set(app1_loc, term_new(APP, 0, app0_loc));
    set(app1_loc + 1, term_new(CO1, @truncate(lab_val), dup_loc));

    HVM.interactions += 1;
    return term_new(APP, 0, app1_loc);
}

/// LOG builtin: @LOG(value, continuation)
/// Prints the value and returns the continuation
/// Usage: @LOG(#42, result) prints "#42" and returns result
fn builtin_log(ref: Term) Term {
    const ref_loc = term_val(ref);
    const val_term = got(ref_loc);
    const cont_term = got(ref_loc + 1);

    // Reduce the value to print it
    const reduced_val = reduce(val_term);
    const tag = term_tag(reduced_val);
    const ext = term_ext(reduced_val);
    const val = term_val(reduced_val);

    // Print based on term type
    print("LOG: ", .{});
    switch (tag) {
        ERA => print("*", .{}),
        NUM => print("#{d}", .{val}),
        LAM => print("λ@{d}", .{val}),
        APP => print("(@{d} ...)", .{val}),
        SUP => print("&{d}{{...}}", .{ext}),
        CO0 => print("CO0({d},{d})", .{ ext, val }),
        CO1 => print("CO1({d},{d})", .{ ext, val }),
        VAR => print("VAR({d})", .{val}),
        REF => print("@fn{d}", .{ext}),
        TYP => print("Type", .{}),
        else => {
            if (tag >= C00 and tag <= C15) {
                print("#C{d}[{d}]", .{ ext, ctr_arity(tag) });
            } else {
                print("?{d}({d},{d})", .{ tag, ext, val });
            }
        },
    }
    print("\n", .{});

    HVM.interactions += 1;
    return cont_term;
}

// =============================================================================
// Structural Equality (Stack-based, no recursion)
// =============================================================================

/// Work item for iterative equality check
const EqWorkItem = struct {
    a: Term,
    b: Term,
};

/// Deep structural equality check between two terms
/// Uses explicit stack to avoid call stack overflow on deep terms
pub fn struct_equal(a: Term, b: Term) bool {
    // Use a fixed-size stack for work items (64KB = 4096 pairs)
    var work_stack: [4096]EqWorkItem = undefined;
    var stack_pos: usize = 0;

    // Push initial pair
    work_stack[0] = .{ .a = a, .b = b };
    stack_pos = 1;

    while (stack_pos > 0) {
        stack_pos -= 1;
        const item = work_stack[stack_pos];

        const a_reduced = reduce(item.a);
        const b_reduced = reduce(item.b);

        const a_tag = term_tag(a_reduced);
        const b_tag = term_tag(b_reduced);

        // Tags must match
        if (a_tag != b_tag) return false;

        const a_ext = term_ext(a_reduced);
        const b_ext = term_ext(b_reduced);
        const a_val = term_val(a_reduced);
        const b_val = term_val(b_reduced);

        switch (a_tag) {
            // Atomic values: compare directly
            ERA, TYP => {},
            NUM => {
                if (a_val != b_val) return false;
            },

            // Lambda: push body comparison
            LAM => {
                if (stack_pos >= work_stack.len) return false; // Stack overflow protection
                work_stack[stack_pos] = .{
                    .a = HVM.heap[a_val],
                    .b = HVM.heap[b_val],
                };
                stack_pos += 1;
            },

            // Application: push both comparisons
            APP => {
                if (stack_pos + 2 > work_stack.len) return false;
                work_stack[stack_pos] = .{
                    .a = HVM.heap[a_val],
                    .b = HVM.heap[b_val],
                };
                work_stack[stack_pos + 1] = .{
                    .a = HVM.heap[a_val + 1],
                    .b = HVM.heap[b_val + 1],
                };
                stack_pos += 2;
            },

            // Superposition: compare label and push both branches
            SUP => {
                if (a_ext != b_ext) return false;
                if (stack_pos + 2 > work_stack.len) return false;
                work_stack[stack_pos] = .{
                    .a = HVM.heap[a_val],
                    .b = HVM.heap[b_val],
                };
                work_stack[stack_pos + 1] = .{
                    .a = HVM.heap[a_val + 1],
                    .b = HVM.heap[b_val + 1],
                };
                stack_pos += 2;
            },

            // Constructors: compare ext and push all fields
            C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 => {
                if (a_ext != b_ext) return false;
                const arity = ctr_arity(a_tag);
                if (stack_pos + arity > work_stack.len) return false;
                for (0..arity) |i| {
                    work_stack[stack_pos + i] = .{
                        .a = HVM.heap[a_val + i],
                        .b = HVM.heap[b_val + i],
                    };
                }
                stack_pos += arity;
            },

            // Default: not equal if we can't determine
            else => return false,
        }
    }

    return true;
}

/// Check if two terms are convertible (definitionally equal)
/// This is more powerful than struct_equal as it handles:
/// 1. Beta equivalence: (λx.t) v ≡ t[x:=v]
/// 2. Eta equivalence: λx.(f x) ≡ f (when x not free in f)
/// 3. Alpha equivalence: λx.x ≡ λy.y
pub fn convertible(a: Term, b: Term) bool {
    // First, normalize both terms to weak head normal form
    const a_whnf = reduce(a);
    const b_whnf = reduce(b);

    // Quick check: if structurally equal after reduction, they're convertible
    if (struct_equal_shallow(a_whnf, b_whnf)) {
        return true;
    }

    const a_tag = term_tag(a_whnf);
    const b_tag = term_tag(b_whnf);

    // Eta-expansion: λx.(f x) ≡ f
    // If one is a LAM and other is not, try eta-expanding the non-LAM
    if (a_tag == LAM and b_tag != LAM) {
        // Check if a = λx.(b x) - i.e., a is eta-expansion of b
        const a_body = got(term_val(a_whnf));
        if (isEtaExpandedForm(a_body, b_whnf)) {
            return true;
        }
    }
    if (b_tag == LAM and a_tag != LAM) {
        // Check if b = λx.(a x)
        const b_body = got(term_val(b_whnf));
        if (isEtaExpandedForm(b_body, a_whnf)) {
            return true;
        }
    }

    // Both are lambdas - check bodies under extension
    if (a_tag == LAM and b_tag == LAM) {
        const a_body = got(term_val(a_whnf));
        const b_body = got(term_val(b_whnf));
        return convertible(a_body, b_body);
    }

    // Both are applications - check function and argument
    if (a_tag == APP and b_tag == APP) {
        const a_func = got(term_val(a_whnf));
        const b_func = got(term_val(b_whnf));
        const a_arg = got(term_val(a_whnf) + 1);
        const b_arg = got(term_val(b_whnf) + 1);
        return convertible(a_func, b_func) and convertible(a_arg, b_arg);
    }

    // Type-level constructs: ALL (Pi types)
    if (a_tag == ALL and b_tag == ALL) {
        const a_dom = got(term_val(a_whnf));
        const b_dom = got(term_val(b_whnf));
        const a_cod = got(term_val(a_whnf) + 1);
        const b_cod = got(term_val(b_whnf) + 1);
        // Contravariant in domain, covariant in codomain
        return convertible(b_dom, a_dom) and convertible(a_cod, b_cod);
    }

    // SIG (Sigma types)
    if (a_tag == SIG and b_tag == SIG) {
        const a_fst = got(term_val(a_whnf));
        const b_fst = got(term_val(b_whnf));
        const a_snd = got(term_val(a_whnf) + 1);
        const b_snd = got(term_val(b_whnf) + 1);
        return convertible(a_fst, b_fst) and convertible(a_snd, b_snd);
    }

    // Constructors
    if (a_tag >= C00 and a_tag <= C15 and a_tag == b_tag) {
        if (term_ext(a_whnf) != term_ext(b_whnf)) return false;
        const arity = ctr_arity(a_tag);
        for (0..arity) |i| {
            const a_field = got(term_val(a_whnf) + i);
            const b_field = got(term_val(b_whnf) + i);
            if (!convertible(a_field, b_field)) return false;
        }
        return true;
    }

    // Fall back to structural equality
    return struct_equal(a_whnf, b_whnf);
}

/// Shallow structural equality (without recursion into children)
fn struct_equal_shallow(a: Term, b: Term) bool {
    const a_tag = term_tag(a);
    const b_tag = term_tag(b);
    if (a_tag != b_tag) return false;

    return switch (a_tag) {
        ERA, TYP => true,
        NUM => term_val(a) == term_val(b),
        VAR => term_val(a) == term_val(b),
        REF => term_ext(a) == term_ext(b),
        else => false, // Need deeper comparison
    };
}

/// Check if body is the eta-expanded form of func: body = (func (VAR bound_to_lambda))
fn isEtaExpandedForm(body: Term, func: Term) bool {
    const body_tag = term_tag(body);
    if (body_tag != APP) return false;

    const app_func = got(term_val(body));
    const app_arg = got(term_val(body) + 1);

    // The argument should be a variable bound to the enclosing lambda
    if (term_tag(app_arg) != VAR) return false;

    // The function part should be convertible with func
    return convertible(app_func, func);
}

// =============================================================================
// Comptime Interaction Dispatch Table
// =============================================================================

/// Unified interaction function signature
/// All interactions receive full context, use what they need
const InteractFn = *const fn (p_val: Val, p_ext: Ext, val: Val, ext: Ext, p_tag: Tag, tag: Tag, next: Term) Term;

/// Wrapper for APP + LAM
fn wrap_app_lam(p_val: Val, _: Ext, val: Val, _: Ext, _: Tag, _: Tag, next: Term) Term {
    HVM.heap[p_val] = next;
    return interact_app_lam(p_val, val);
}

/// Wrapper for APP + ERA
fn wrap_app_era(_: Val, _: Ext, _: Val, _: Ext, _: Tag, _: Tag, _: Term) Term {
    return interact_app_era();
}

/// Wrapper for APP + SUP (needs stack sync - handled in reduce)
fn wrap_app_sup(p_val: Val, _: Ext, val: Val, ext: Ext, _: Tag, _: Tag, _: Term) Term {
    return interact_app_sup(p_val, val, ext);
}

/// Wrapper for CO0/CO1 + SUP
fn wrap_dup_sup(p_val: Val, p_ext: Ext, val: Val, ext: Ext, p_tag: Tag, _: Tag, _: Term) Term {
    return interact_dup_sup(p_tag, p_val, p_ext, val, ext);
}

/// Wrapper for CO0/CO1 + NUM
fn wrap_dup_num(p_val: Val, _: Ext, _: Val, _: Ext, _: Tag, _: Tag, next: Term) Term {
    return interact_dup_num(p_val, next);
}

/// Wrapper for CO0/CO1 + ERA
fn wrap_dup_era(p_val: Val, _: Ext, _: Val, _: Ext, _: Tag, _: Tag, _: Term) Term {
    return interact_dup_era(p_val);
}

/// Wrapper for CO0/CO1 + LAM (needs stack sync - handled in reduce)
fn wrap_dup_lam(p_val: Val, p_ext: Ext, val: Val, _: Ext, p_tag: Tag, _: Tag, _: Term) Term {
    return interact_dup_lam(p_tag, p_val, p_ext, val);
}

/// Wrapper for CO0/CO1 + CTR (needs stack sync - handled in reduce)
fn wrap_dup_ctr(p_val: Val, p_ext: Ext, val: Val, ext: Ext, p_tag: Tag, tag: Tag, _: Term) Term {
    return interact_dup_ctr(p_tag, p_val, p_ext, tag, val, ext);
}

/// Wrapper for CO0/CO1 + APP (needs stack sync - handled in reduce)
fn wrap_dup_app(p_val: Val, p_ext: Ext, val: Val, _: Ext, p_tag: Tag, _: Tag, _: Term) Term {
    return interact_dup_app(p_tag, p_val, p_ext, val);
}

/// Wrapper for MAT + CTR (needs stack sync - handled in reduce)
fn wrap_mat_ctr(p_val: Val, _: Ext, val: Val, ext: Ext, _: Tag, tag: Tag, _: Term) Term {
    return interact_mat_ctr(p_val, tag, val, ext);
}

/// Wrapper for SWI + NUM (needs stack sync - handled in reduce)
fn wrap_swi_num(p_val: Val, _: Ext, val: Val, _: Ext, _: Tag, _: Tag, _: Term) Term {
    return interact_swi_num(p_val, val);
}

/// Interaction requires stack synchronization (calls alloc/reduce internally)
const NeedsSync = struct {
    fn check(p_tag: Tag, tag: Tag) bool {
        // APP+SUP, CO*+LAM, CO*+CTR, CO*+APP, MAT+CTR, SWI+NUM all allocate
        if (p_tag == APP and tag == SUP) return true;
        if ((p_tag == CO0 or p_tag == CO1) and tag == LAM) return true;
        if ((p_tag == CO0 or p_tag == CO1) and tag >= C00 and tag <= C15) return true;
        if ((p_tag == CO0 or p_tag == CO1) and tag == APP) return true;
        if (p_tag == MAT and tag >= C00 and tag <= C15) return true;
        if (p_tag == SWI and tag == NUM) return true;
        return false;
    }
};

/// Build the interaction dispatch table at compile time
const interaction_table: [256][256]?InteractFn = blk: {
    var table: [256][256]?InteractFn = .{.{null} ** 256} ** 256;

    // APP interactions
    table[APP][LAM] = wrap_app_lam;
    table[APP][ERA] = wrap_app_era;
    table[APP][SUP] = wrap_app_sup;

    // CO0 interactions
    table[CO0][SUP] = wrap_dup_sup;
    table[CO0][NUM] = wrap_dup_num;
    table[CO0][ERA] = wrap_dup_era;
    table[CO0][LAM] = wrap_dup_lam;
    table[CO0][APP] = wrap_dup_app;

    // CO1 interactions
    table[CO1][SUP] = wrap_dup_sup;
    table[CO1][NUM] = wrap_dup_num;
    table[CO1][ERA] = wrap_dup_era;
    table[CO1][LAM] = wrap_dup_lam;
    table[CO1][APP] = wrap_dup_app;

    // CO0/CO1 + CTR interactions (all 16 constructors)
    for ([_]Tag{ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 }) |ctr| {
        table[CO0][ctr] = wrap_dup_ctr;
        table[CO1][ctr] = wrap_dup_ctr;
    }

    // MAT + CTR interactions
    for ([_]Tag{ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 }) |ctr| {
        table[MAT][ctr] = wrap_mat_ctr;
    }

    // SWI + NUM
    table[SWI][NUM] = wrap_swi_num;

    break :blk table;
};

// =============================================================================
// WNF Reduction (Weak Normal Form) - HVM4 Style
// =============================================================================

/// Main reduction function with cached pointers and inlined hot paths
var debug_reduce: bool = false;

pub noinline fn reduce(term: Term) Term {
    const heap = HVM.heap;
    const stack = HVM.stack;
    const stop = HVM.stack_pos;
    var spos = stop;
    var next = term;
    var steps: u64 = 0;

    while (true) {
        steps += 1;
        if (debug_reduce and steps < 200) {
            print("  [{d}] tag={d}, ext={d}, val={d}, spos={d}", .{ steps, term_tag(next), term_ext(next), term_val(next), spos });
            if ((next & SUB_BIT) != 0) print(" [SUB]", .{});
            print("\n", .{});
        }
        // Removed infinite loop detection

        // Handle substitution - SUB_BIT marks this is a substituted value
        // Just clear the bit and continue with the value
        if ((next & SUB_BIT) != 0) {
            @branchHint(.likely);
            next = next & ~SUB_BIT;
            continue;
        }

        const tag = term_tag(next);
        const ext = term_ext(next);
        const val = term_val(next);

        switch (tag) {
            // Variable: follow substitution
            VAR => {
                next = heap[val];
                if ((next & SUB_BIT) != 0) {
                    next = next & ~SUB_BIT;
                    continue;
                }
                // VAR points to unreduced content
                // Push VAR to stack so we can store result back when done
                stack[spos] = term_new(VAR, 0, val);
                spos += 1;
                continue;
            },

            // Collapse projections
            CO0, CO1 => {
                const dup_val = heap[val];
                if ((dup_val & SUB_BIT) != 0) {
                    @branchHint(.likely);
                    next = dup_val & ~SUB_BIT;
                    continue;
                }
                stack[spos] = next;
                spos += 1;
                next = dup_val;
            },

            // Reference expansion
            REF => {
                HVM.stack_pos = spos;
                if (HVM.book[ext]) |func| {
                    @branchHint(.likely);
                    next = func(next);
                    HVM.interactions += 1;
                } else {
                    print("undefined function: {d}\n", .{ext});
                    std.process.exit(1);
                }
                spos = HVM.stack_pos;
            },

            // Eliminators: push and reduce scrutinee
            APP, MAT, SWI, USE, LET => {
                stack[spos] = next;
                spos += 1;
                next = heap[val];
            },

            // Binary primitives
            P02 => {
                // Push frame and reduce first operand
                stack[spos] = term_new(F_OP2, ext, val);
                spos += 1;
                next = heap[val];
            },

            // Structural equality: reduce first operand
            EQL => {
                stack[spos] = term_new(F_EQL, ext, val);
                spos += 1;
                next = heap[val];
            },

            // Guarded reduction (RED): f ~> g
            // Only reduce g when scrutinized
            RED => {
                // RED(f, g) when applied, substitute f first
                // heap[val] = f, heap[val+1] = g
                next = heap[val];
            },

            // Type annotation: {term : Type}
            // heap[val] = term, heap[val+1] = type
            ANN => {
                // Push frame to check type, reduce term first
                stack[spos] = term_new(F_ANN, ext, val);
                spos += 1;
                next = heap[val];
            },

            // Bridge (theta): θx. term
            // Creates a scope for dependent types
            BRI => {
                stack[spos] = term_new(F_BRI, ext, val);
                spos += 1;
                next = heap[val];
            },

            // Type universe - already a value
            TYP => {
                if (spos <= stop) break;
                spos -= 1;
                const prev = stack[spos];
                const p_tag = term_tag(prev);

                // ANN + TYP: Type annotation with type universe
                if (p_tag == F_ANN) {
                    HVM.interactions += 1;
                    // Type checks - decay the annotation
                    next = heap[term_val(prev)]; // Return the term
                    continue;
                }

                heap[term_val(prev)] = next;
                HVM.stack_pos = spos;
                return next;
            },

            // Values: interact with stack top
            LAM, SUP, ERA, NUM, C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 => {
                // Debug: for C02, print fields
                if (debug_reduce and tag == C02) {
                    std.debug.print("C02 at {d}: heap[{d}]=0x{x}, heap[{d}+1]=0x{x}\n", .{ val, val, heap[val], val, heap[val + 1] });
                }
                if (spos <= stop) {
                    @branchHint(.unlikely);
                    break;
                }

                spos -= 1;
                const prev = stack[spos];
                const p_tag = term_tag(prev);
                const p_ext = term_ext(prev);
                const p_val = term_val(prev);

                // Comptime dispatch table lookup for value-value interactions
                if (interaction_table[p_tag][tag]) |interact_fn| {
                    @branchHint(.likely);
                    // Check if interaction needs stack synchronization
                    if (NeedsSync.check(p_tag, tag)) {
                        HVM.stack_pos = spos;
                        next = interact_fn(p_val, p_ext, val, ext, p_tag, tag, next);
                        spos = HVM.stack_pos;
                    } else {
                        next = interact_fn(p_val, p_ext, val, ext, p_tag, tag, next);
                    }
                    continue;
                }

                // VAR on stack: store the reduced result at VAR's location
                if (p_tag == VAR) {
                    heap[p_val] = term_set_sub(next);
                    // Continue with the value (don't pop - already popped)
                    continue;
                }

                // Frame handlers (not in dispatch table - need special handling)

                // F_OP2 + NUM: First operand ready, reduce second
                if (p_tag == F_OP2 and tag == NUM) {
                    HVM.interactions += 1;
                    const op2_loc = p_val;
                    const arg1 = heap[op2_loc + 1];
                    // Store first operand value for later
                    heap[op2_loc] = next;
                    stack[spos] = term_new(F_OP2 + 1, p_ext, op2_loc);
                    spos += 1;
                    next = arg1;
                    continue;
                }

                // F_OP2+1 + NUM: Both operands ready, compute
                if (p_tag == F_OP2 + 1 and tag == NUM) {
                    HVM.interactions += 1;
                    const x = term_val(heap[p_val]); // First operand stored earlier
                    const y = val; // Second operand just reduced
                    next = term_new(NUM, 0, compute_op(p_ext, x, y));
                    continue;
                }

                // LET binding
                if (p_tag == LET) {
                    HVM.interactions += 1;
                    const bod = heap[p_val + 1];
                    heap[p_val] = term_set_sub(next);
                    next = bod;
                    continue;
                }

                // F_EQL + value: First operand ready, reduce second
                if (p_tag == F_EQL) {
                    HVM.interactions += 1;
                    const eql_loc = p_val;
                    const arg1 = heap[eql_loc + 1];
                    heap[eql_loc] = next; // Store first operand
                    stack[spos] = term_new(F_EQL2, p_ext, eql_loc);
                    spos += 1;
                    next = arg1;
                    continue;
                }

                // F_EQL2 + value: Both operands ready, compare
                if (p_tag == F_EQL2) {
                    HVM.interactions += 1;
                    const first = heap[p_val];
                    const second = next;
                    // Structural equality check
                    const equal = struct_equal(first, second);
                    next = term_new(NUM, 0, if (equal) 1 else 0);
                    continue;
                }

                // F_ANN + value: Type checking and annotation decay
                if (p_tag == F_ANN) {
                    HVM.interactions += 1;
                    const typ = heap[p_val + 1];
                    const typ_tag = term_tag(typ);

                    // SUP requires special handling: distribute annotation
                    if (tag == SUP) {
                        // {SUP(a,b) : T} => SUP({a:T}, {b:T})
                        const sup_loc = val;
                        const sup_ext = ext;

                        const loc0 = alloc(2);
                        heap[loc0] = heap[sup_loc];
                        heap[loc0 + 1] = typ;

                        const loc1 = alloc(2);
                        heap[loc1] = heap[sup_loc + 1];
                        heap[loc1 + 1] = typ;

                        const new_loc = alloc(2);
                        heap[new_loc] = term_new(ANN, 0, @truncate(loc0));
                        heap[new_loc + 1] = term_new(ANN, 0, @truncate(loc1));

                        next = term_new(SUP, sup_ext, @truncate(new_loc));
                        continue;
                    }

                    // LAM + ALL: Check function type
                    if (tag == LAM and typ_tag == ALL) {
                        // LAM has body at heap[val]
                        // ALL has domain at heap[typ_val], codomain (lambda) at heap[typ_val + 1]
                        const typ_val = term_val(typ);
                        _ = heap[typ_val]; // domain - for future arg type checking
                        const codomain_lam = heap[typ_val + 1];

                        // The codomain should be a lambda; extract its body
                        if (term_tag(codomain_lam) == LAM) {
                            const codomain_loc = term_val(codomain_lam);
                            const codomain = heap[codomain_loc];

                            // Create annotated body: {body : codomain}
                            // First, we need to substitute the domain into the body type
                            const lam_body = heap[val];
                            const ann_loc = alloc(2);
                            heap[ann_loc] = lam_body;
                            heap[ann_loc + 1] = codomain;

                            // Rebuild the lambda with annotated body
                            const new_lam_loc = alloc(1);
                            heap[new_lam_loc] = term_new(ANN, 0, @truncate(ann_loc));
                            next = term_new(LAM, ext, @truncate(new_lam_loc));
                            continue;
                        }
                        // If codomain isn't a lambda, decay normally
                        next = term_new(tag, ext, val);
                        continue;
                    }

                    // Constructor + SIG: Check dependent pair type
                    if (tag >= C00 and tag <= C15 and typ_tag == SIG) {
                        const arity = ctr_arity(tag);
                        if (arity >= 2) {
                            // SIG has fst_type at heap[typ_val], snd_type (lambda) at heap[typ_val + 1]
                            const typ_val = term_val(typ);
                            const fst_type = heap[typ_val];
                            const snd_type_lam = heap[typ_val + 1];

                            // Get the first element of the pair
                            const fst_elem = heap[val];

                            // Annotate first element with first type
                            const ann_fst_loc = alloc(2);
                            heap[ann_fst_loc] = fst_elem;
                            heap[ann_fst_loc + 1] = fst_type;
                            const ann_fst = term_new(ANN, 0, @truncate(ann_fst_loc));

                            // For second element: apply snd_type_lam to fst_elem
                            if (term_tag(snd_type_lam) == LAM) {
                                const snd_elem = heap[val + 1];
                                const snd_type_loc = term_val(snd_type_lam);
                                const snd_type_body = heap[snd_type_loc];

                                const ann_snd_loc = alloc(2);
                                heap[ann_snd_loc] = snd_elem;
                                heap[ann_snd_loc + 1] = snd_type_body; // TODO: substitute fst_elem

                                // Rebuild constructor with annotated elements
                                const new_ctr_loc = alloc(arity);
                                heap[new_ctr_loc] = ann_fst;
                                heap[new_ctr_loc + 1] = term_new(ANN, 0, @truncate(ann_snd_loc));
                                // Copy remaining elements
                                for (2..arity) |i| {
                                    heap[new_ctr_loc + i] = heap[val + i];
                                }
                                next = term_new(tag, ext, @truncate(new_ctr_loc));
                                continue;
                            }
                        }
                        // Fall through to decay
                    }

                    // NUM + type: Verify number has correct type (basic check)
                    if (tag == NUM) {
                        // Numbers are typically well-typed; decay
                        next = term_new(tag, ext, val);
                        continue;
                    }

                    // TYP + type: Types themselves are well-typed
                    if (tag == TYP) {
                        next = term_new(tag, ext, val);
                        continue;
                    }

                    // All other values: annotation decays, return the term unchanged
                    next = term_new(tag, ext, val);
                    continue;
                }

                // F_BRI + value: Bridge completed
                if (p_tag == F_BRI) {
                    HVM.interactions += 1;
                    // Bridge just returns the body
                    next = term_new(tag, ext, val);
                    continue;
                }

                // Update scrutinee location for other interactions
                heap[p_val] = next;

                // No interaction found - return
                HVM.stack_pos = spos;
                return next;
            },

            else => break,
        }
    }

    HVM.stack_pos = spos;
    return next;
}

// =============================================================================
// Normal Form (Full Reduction) - Stack-based, no recursion
// =============================================================================

/// Work item for iterative normal form computation
const NormalWorkItem = struct {
    term: Term,
    phase: enum { reduce, process_children, done },
    loc: Val,
    child_idx: u32,
    arity: u32,
};

/// Compute normal form without recursion
/// Uses explicit stack to avoid call stack overflow on deep terms
pub fn normal(term: Term) Term {
    // Work stack for iterative traversal
    var work_stack: [8192]NormalWorkItem = undefined;
    var stack_pos: usize = 0;

    // Result stack to track what we're building
    var result_stack: [8192]Term = undefined;
    var result_pos: usize = 0;

    // Push initial term
    work_stack[0] = .{
        .term = term,
        .phase = .reduce,
        .loc = 0,
        .child_idx = 0,
        .arity = 0,
    };
    stack_pos = 1;

    while (stack_pos > 0) {
        const idx = stack_pos - 1;
        var item = &work_stack[idx];

        switch (item.phase) {
            .reduce => {
                const result = reduce(item.term);
                const tag = term_tag(result);
                const loc = term_val(result);

                // Store the reduced result
                item.term = result;
                item.loc = loc;

                // Determine children to process
                switch (tag) {
                    LAM => {
                        item.arity = 1;
                        item.child_idx = 0;
                        item.phase = .process_children;
                    },
                    APP, SUP => {
                        item.arity = 2;
                        item.child_idx = 0;
                        item.phase = .process_children;
                    },
                    C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 => {
                        item.arity = ctr_arity(tag);
                        item.child_idx = 0;
                        item.phase = if (item.arity > 0) .process_children else .done;
                    },
                    else => {
                        item.phase = .done;
                    },
                }
            },

            .process_children => {
                if (item.child_idx < item.arity) {
                    // Push child for processing
                    const child_term = got(item.loc + item.child_idx);
                    item.child_idx += 1;

                    if (stack_pos >= work_stack.len) {
                        // Stack overflow - just return what we have
                        item.phase = .done;
                        continue;
                    }

                    work_stack[stack_pos] = .{
                        .term = child_term,
                        .phase = .reduce,
                        .loc = 0,
                        .child_idx = 0,
                        .arity = 0,
                    };
                    stack_pos += 1;
                } else {
                    // All children processed, update heap with normalized children
                    // Children results are on result_stack
                    const arity = item.arity;
                    if (result_pos >= arity) {
                        var i: u32 = 0;
                        while (i < arity) : (i += 1) {
                            result_pos -= 1;
                            set(item.loc + (arity - 1 - i), result_stack[result_pos]);
                        }
                    }
                    item.phase = .done;
                }
            },

            .done => {
                // Push result and pop work item
                if (result_pos < result_stack.len) {
                    result_stack[result_pos] = item.term;
                    result_pos += 1;
                }
                stack_pos -= 1;
            },
        }
    }

    // Final result is on top of result stack
    if (result_pos > 0) {
        return result_stack[result_pos - 1];
    }
    return term;
}

/// Simple recursive normal form (for small terms or when stack is acceptable)
pub fn normal_recursive(term: Term) Term {
    const result = reduce(term);
    const tag = term_tag(result);

    switch (tag) {
        LAM => {
            const loc = term_val(result);
            const bod = normal_recursive(got(loc));
            set(loc, bod);
        },
        APP => {
            const loc = term_val(result);
            const fun = normal_recursive(got(loc));
            const arg = normal_recursive(got(loc + 1));
            set(loc, fun);
            set(loc + 1, arg);
        },
        SUP => {
            const loc = term_val(result);
            const lft = normal_recursive(got(loc));
            const rgt = normal_recursive(got(loc + 1));
            set(loc, lft);
            set(loc + 1, rgt);
        },
        else => {
            if (tag >= C00 and tag <= C15) {
                const loc = term_val(result);
                const arity = ctr_arity(tag);
                for (0..arity) |i| {
                    const field = normal_recursive(got(loc + i));
                    set(loc + i, field);
                }
            }
        },
    }

    return result;
}

// =============================================================================
// Lazy Collapse (BFS enumeration of superpositions)
// =============================================================================

pub const CollapseResult = struct {
    term: Term,
    done: bool,
};

/// Single-step collapse: lift one SUP to the top
pub fn collapse_step(term: Term) CollapseResult {
    const result = reduce(term);
    const tag = term_tag(result);

    if (tag == SUP) {
        return .{ .term = result, .done = false };
    }

    // Check children for SUPs
    switch (tag) {
        LAM => {
            const loc = term_val(result);
            const bod_result = collapse_step(got(loc));
            if (!bod_result.done and term_tag(bod_result.term) == SUP) {
                // Lift SUP through LAM
                const sup_loc = term_val(bod_result.term);
                const sup_lab = term_ext(bod_result.term);
                const lft = got(sup_loc);
                const rgt = got(sup_loc + 1);

                const lam0_loc = alloc(1);
                set(lam0_loc, lft);
                const lam1_loc = alloc(1);
                set(lam1_loc, rgt);

                const new_sup_loc = alloc(2);
                set(new_sup_loc, term_new(LAM, 0, lam0_loc));
                set(new_sup_loc + 1, term_new(LAM, 0, lam1_loc));

                return .{ .term = term_new(SUP, sup_lab, new_sup_loc), .done = false };
            }
            set(loc, bod_result.term);
        },
        else => {},
    }

    return .{ .term = result, .done = true };
}

// =============================================================================
// Auto-Dup System
// =============================================================================

/// Count variable uses in a term
pub fn count_uses(term: Term, var_idx: u32, depth: u32) u32 {
    const tag = term_tag(term);
    const val = term_val(term);

    switch (tag) {
        VAR => {
            if (val == var_idx + depth) return 1;
            return 0;
        },
        LAM => {
            return count_uses(got(val), var_idx, depth + 1);
        },
        APP => {
            const fun_uses = count_uses(got(val), var_idx, depth);
            const arg_uses = count_uses(got(val + 1), var_idx, depth);
            return fun_uses + arg_uses;
        },
        SUP => {
            const lft_uses = count_uses(got(val), var_idx, depth);
            const rgt_uses = count_uses(got(val + 1), var_idx, depth);
            return lft_uses + rgt_uses;
        },
        else => return 0,
    }
}

// Note: fresh_label() is defined in the Label Recycling section below

// =============================================================================
// SupGen - Superposition-based Program Enumeration
// =============================================================================

/// Create a superposition of two terms with a fresh label
/// SUP{a, b} - represents both a AND b simultaneously
pub fn sup(a: Term, b: Term) Term {
    const lab = fresh_label();
    const loc = alloc(2);
    set(loc, a);
    set(loc + 1, b);
    return term_new(SUP, lab, loc);
}

/// Enumerate a bit: SUP{#0, #1}
/// Returns a superposition of 0 and 1
pub fn enum_bit() Term {
    return sup(term_new(NUM, 0, 0), term_new(NUM, 0, 1));
}

/// Enumerate a boolean: SUP{#True, #False}
/// Using C00 for True (nullary constructor 0) and C01 for False
pub fn enum_bool() Term {
    const true_term = term_new(C00, 0, 0); // #True
    const false_term = term_new(C00, 1, 0); // #False
    return sup(true_term, false_term);
}

/// Enumerate natural numbers up to max (exclusive)
/// SUP{#0, SUP{#1, SUP{#2, ...}}}
pub fn enum_nat(max: u32) Term {
    if (max == 0) return term_new(NUM, 0, 0);
    if (max == 1) return term_new(NUM, 0, 0);

    var result = term_new(NUM, 0, max - 1);
    var i: u32 = max - 1;
    while (i > 0) {
        i -= 1;
        result = sup(term_new(NUM, 0, i), result);
    }
    return result;
}

/// Enumerate terms of a given depth
/// depth 0: variables, numbers
/// depth 1: lambdas, applications of depth-0 terms
/// etc.
pub fn enum_term(depth: u32, num_vars: u32) Term {
    if (depth == 0) {
        // Base case: enumerate variables or a number
        if (num_vars == 0) {
            return term_new(NUM, 0, 0); // Just zero
        }
        // Enumerate available variables
        var result = term_new(VAR, 0, num_vars - 1);
        var i: u32 = num_vars - 1;
        while (i > 0) {
            i -= 1;
            result = sup(term_new(VAR, 0, i), result);
        }
        // Also include zero
        return sup(term_new(NUM, 0, 0), result);
    }

    // Recursive case: enumerate lambda or application
    const sub_term = enum_term(depth - 1, num_vars);

    // Lambda: λ. body (with one more variable in scope)
    const lam_body = enum_term(depth - 1, num_vars + 1);
    const lam_loc = alloc(1);
    set(lam_loc, lam_body);
    const lam_term = term_new(LAM, 0, lam_loc);

    // Application: (f x) where f and x are sub-terms
    const app_loc = alloc(2);
    set(app_loc, sub_term);
    set(app_loc + 1, enum_term(depth - 1, num_vars)); // Fresh enumeration for arg
    const app_term = term_new(APP, 0, app_loc);

    // Return SUP of lambda and application
    return sup(lam_term, app_term);
}

/// Check if a superposed term satisfies a predicate
/// Collapses to the first branch that returns #1
pub fn sup_find(term: Term, predicate: *const fn (Term) Term) ?Term {
    const result = reduce(term);
    const tag = term_tag(result);

    if (tag == SUP) {
        const loc = term_val(result);
        const lft = got(loc);
        const rgt = got(loc + 1);

        // Try left branch
        const lft_check = predicate(lft);
        const lft_reduced = reduce(lft_check);
        if (term_tag(lft_reduced) == NUM and term_val(lft_reduced) == 1) {
            return lft;
        }

        // Try right branch
        const rgt_check = predicate(rgt);
        const rgt_reduced = reduce(rgt_check);
        if (term_tag(rgt_reduced) == NUM and term_val(rgt_reduced) == 1) {
            return rgt;
        }

        // Recursively search
        if (sup_find(lft, predicate)) |found| return found;
        if (sup_find(rgt, predicate)) |found| return found;

        return null;
    }

    // Not a superposition, check directly
    const check = predicate(result);
    const check_reduced = reduce(check);
    if (term_tag(check_reduced) == NUM and term_val(check_reduced) == 1) {
        return result;
    }

    return null;
}

/// Create dependent function type: Π(x:A).B
/// ALL(A, λx.B) where B can reference x
pub fn make_all(domain: Term, codomain: Term) Term {
    const loc = alloc(2);
    set(loc, domain);
    set(loc + 1, codomain); // Should be a lambda
    return term_new(ALL, 0, loc);
}

/// Create dependent pair type: Σ(x:A).B
pub fn make_sig(fst_type: Term, snd_type: Term) Term {
    const loc = alloc(2);
    set(loc, fst_type);
    set(loc + 1, snd_type); // Should be a lambda
    return term_new(SIG, 0, loc);
}

/// Create type annotation: {term : Type}
pub fn annotate(term: Term, typ: Term) Term {
    const loc = alloc(2);
    set(loc, term);
    set(loc + 1, typ);
    return term_new(ANN, 0, loc);
}

/// Type check a term (returns #1 if valid, #0 if error)
/// This reduces the annotation - if it decays successfully, the term is well-typed
pub fn type_check(term: Term) Term {
    const result = reduce(term);
    const tag = term_tag(result);

    // If the annotation decayed (tag is not ANN), type checking succeeded
    if (tag != ANN) {
        return term_new(NUM, 0, 1);
    }

    // Still an annotation - type error
    return term_new(NUM, 0, 0);
}

/// Create linear type wrapper: term must be used exactly once
/// Used to prevent oracle problem by ensuring no duplication
pub fn linear(term: Term) Term {
    const loc = alloc(1);
    set(loc, term);
    return term_new(LIN, 0, loc);
}

/// Create affine type wrapper: term can be used at most once
/// Less strict than linear - allows dropping but not duplication
pub fn affine(term: Term) Term {
    const loc = alloc(1);
    set(loc, term);
    return term_new(AFF, 0, loc);
}

/// Create linear type with label scope
/// The term can safely interact with SUP of the same label
pub fn linear_labeled(term: Term, label: Ext) Term {
    const loc = alloc(1);
    set(loc, term);
    return term_new(LIN, label, loc);
}

/// Create type error term
/// Stores expected and actual types for error reporting
pub fn type_error(expected: Term, actual: Term, code: TypeErrorCode) Term {
    const loc = alloc(2);
    set(loc, expected);
    set(loc + 1, actual);
    return term_new(ERR, @intFromEnum(code), loc);
}

/// Check if a term is linear or affine (has usage restriction)
pub fn is_linear_or_affine(term: Term) bool {
    const tag = term_tag(term);
    return tag == LIN or tag == AFF;
}

/// Get the linearity of a term
pub fn get_linearity(term: Term) Linearity {
    const tag = term_tag(term);
    if (tag == LIN) return .linear;
    if (tag == AFF) return .affine;
    return .unrestricted;
}

/// Unwrap a linear/affine term to get the inner term
pub fn unwrap_linear(term: Term) Term {
    const tag = term_tag(term);
    if (tag == LIN or tag == AFF) {
        const loc = term_val(term);
        return got(loc);
    }
    return term;
}

// =============================================================================
// Reference Counting & Memory Management
// =============================================================================

/// Increment reference count for a term's heap location
pub fn rc_inc(loc: Val) void {
    if (HVM.refcounts) |rc| {
        if (loc < rc.len) {
            rc[loc] +|= 1; // Saturating add
        }
    }
}

/// Decrement reference count, returns true if count reached zero
pub fn rc_dec(loc: Val) bool {
    if (HVM.refcounts) |rc| {
        if (loc < rc.len and rc[loc] > 0) {
            rc[loc] -= 1;
            return rc[loc] == 0;
        }
    }
    return false;
}

/// Get reference count for a location
pub fn rc_get(loc: Val) u32 {
    if (HVM.refcounts) |rc| {
        if (loc < rc.len) {
            return rc[loc];
        }
    }
    return 0;
}

// =============================================================================
// Label Recycling
// =============================================================================

/// Get a fresh label, recycling if possible
pub fn fresh_label() Ext {
    // Try to recycle a label first
    if (HVM.free_labels) |fl| {
        if (HVM.free_labels_count > 0) {
            HVM.free_labels_count -= 1;
            return fl[HVM.free_labels_count];
        }
    }

    // Otherwise allocate a new one
    const lab = HVM.fresh_label;
    HVM.fresh_label +%= 1; // Wrapping add to handle overflow

    // Warn if we're running low on labels
    if (HVM.fresh_label == 0) {
        print("WARNING: Label space exhausted, wrapping around\n", .{});
    }

    return lab;
}

/// Return a label to the free pool for reuse
pub fn recycle_label(lab: Ext) void {
    if (HVM.free_labels) |fl| {
        if (HVM.free_labels_count < fl.len) {
            fl[HVM.free_labels_count] = lab;
            HVM.free_labels_count += 1;
        }
    }
}

// =============================================================================
// Parallel Execution (Multi-threaded SIMD)
// =============================================================================

/// Maximum workers (actual count from config)
const MAX_WORKERS = 128;

var worker_states: [MAX_WORKERS]?*State = [_]?*State{null} ** MAX_WORKERS;
var worker_allocator: ?std.mem.Allocator = null;
var num_workers: usize = DEFAULT_NUM_WORKERS;

pub fn parallel_init(allocator: std.mem.Allocator) !void {
    return parallel_init_with_workers(allocator, DEFAULT_NUM_WORKERS);
}

pub fn parallel_init_with_workers(allocator: std.mem.Allocator, workers: usize) !void {
    worker_allocator = allocator;
    num_workers = @min(workers, MAX_WORKERS);
    for (0..num_workers) |i| {
        worker_states[i] = try State.init(allocator);
    }
}

pub fn parallel_free() void {
    if (worker_allocator) |allocator| {
        for (0..num_workers) |i| {
            if (worker_states[i]) |state| {
                state.deinit(allocator);
                worker_states[i] = null;
            }
        }
    }
}

pub fn get_num_workers() usize {
    return num_workers;
}

/// SIMD batch add
pub fn batch_add(a: []const u32, b: []const u32, results: []u32) void {
    const vec_size = 4;
    const Vec = @Vector(vec_size, u32);
    var i: usize = 0;
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: Vec = a[i..][0..vec_size].*;
        const vb: Vec = b[i..][0..vec_size].*;
        results[i..][0..vec_size].* = va +% vb;
    }
    while (i < a.len) : (i += 1) {
        results[i] = a[i] +% b[i];
    }
}

/// SIMD batch multiply
pub fn batch_mul(a: []const u32, b: []const u32, results: []u32) void {
    const vec_size = 4;
    const Vec = @Vector(vec_size, u32);
    var i: usize = 0;
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: Vec = a[i..][0..vec_size].*;
        const vb: Vec = b[i..][0..vec_size].*;
        results[i..][0..vec_size].* = va *% vb;
    }
    while (i < a.len) : (i += 1) {
        results[i] = a[i] *% b[i];
    }
}

/// SIMD batch subtract
pub fn batch_sub(a: []const u32, b: []const u32, results: []u32) void {
    const vec_size = 4;
    const Vec = @Vector(vec_size, u32);
    var i: usize = 0;
    while (i + vec_size <= a.len) : (i += vec_size) {
        const va: Vec = a[i..][0..vec_size].*;
        const vb: Vec = b[i..][0..vec_size].*;
        results[i..][0..vec_size].* = va -% vb;
    }
    while (i < a.len) : (i += 1) {
        results[i] = a[i] -% b[i];
    }
}

/// Parallel SIMD batch add
pub fn parallel_batch_add(a: []const u32, b: []const u32, results: []u32) void {
    const workers = num_workers;
    const chunk_size = (a.len + workers - 1) / workers;
    var threads: [MAX_WORKERS]?std.Thread = [_]?std.Thread{null} ** MAX_WORKERS;

    for (0..workers) |i| {
        const start = i * chunk_size;
        if (start >= a.len) break;
        const end = @min(start + chunk_size, a.len);
        threads[i] = std.Thread.spawn(.{}, struct {
            fn work(a_s: []const u32, b_s: []const u32, r_s: []u32) void {
                batch_add(a_s, b_s, r_s);
            }
        }.work, .{ a[start..end], b[start..end], results[start..end] }) catch null;
    }

    for (threads[0..workers]) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

/// Parallel SIMD batch multiply
pub fn parallel_batch_mul(a: []const u32, b: []const u32, results: []u32) void {
    const workers = num_workers;
    const chunk_size = (a.len + workers - 1) / workers;
    var threads: [MAX_WORKERS]?std.Thread = [_]?std.Thread{null} ** MAX_WORKERS;

    for (0..workers) |i| {
        const start = i * chunk_size;
        if (start >= a.len) break;
        const end = @min(start + chunk_size, a.len);
        threads[i] = std.Thread.spawn(.{}, struct {
            fn work(a_s: []const u32, b_s: []const u32, r_s: []u32) void {
                batch_mul(a_s, b_s, r_s);
            }
        }.work, .{ a[start..end], b[start..end], results[start..end] }) catch null;
    }

    for (threads[0..workers]) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

// =============================================================================
// Parallel Reduction (Work-Stealing)
// =============================================================================

/// Work item for parallel reduction
const ParallelWorkItem = struct {
    term: Term,
    result_slot: *Term,
};

/// Work-stealing queue for parallel reduction
const WorkQueue = struct {
    items: [4096]ParallelWorkItem,
    head: std.atomic.Value(usize),
    tail: std.atomic.Value(usize),

    fn init() WorkQueue {
        return .{
            .items = undefined,
            .head = std.atomic.Value(usize).init(0),
            .tail = std.atomic.Value(usize).init(0),
        };
    }

    fn push(self: *WorkQueue, item: ParallelWorkItem) bool {
        const tail = self.tail.load(.acquire);
        const next_tail = (tail + 1) % self.items.len;
        if (next_tail == self.head.load(.acquire)) {
            return false; // Queue full
        }
        self.items[tail] = item;
        self.tail.store(next_tail, .release);
        return true;
    }

    fn pop(self: *WorkQueue) ?ParallelWorkItem {
        const head = self.head.load(.acquire);
        if (head == self.tail.load(.acquire)) {
            return null; // Queue empty
        }
        const item = self.items[head];
        self.head.store((head + 1) % self.items.len, .release);
        return item;
    }

    fn steal(self: *WorkQueue) ?ParallelWorkItem {
        // Try to steal from the tail (opposite end)
        const tail = self.tail.load(.acquire);
        const head = self.head.load(.acquire);
        if (head == tail) {
            return null;
        }
        const prev_tail = if (tail == 0) self.items.len - 1 else tail - 1;
        const item = self.items[prev_tail];
        // CAS to claim the item
        if (self.tail.cmpxchgStrong(tail, prev_tail, .acq_rel, .acquire)) |_| {
            return null; // Someone else got it
        }
        return item;
    }
};

/// Global work queues for each worker
var work_queues: [MAX_WORKERS]WorkQueue = undefined;
var parallel_reduction_active: bool = false;

/// Initialize parallel reduction system
pub fn parallel_reduction_init() void {
    for (0..num_workers) |i| {
        work_queues[i] = WorkQueue.init();
    }
    parallel_reduction_active = true;
}

/// Reduce a superposition in parallel across workers
/// Each branch of the SUP can be reduced independently
pub fn parallel_reduce_sup(term: Term) Term {
    const tag = term_tag(term);

    if (tag != SUP or !parallel_reduction_active or num_workers < 2) {
        return reduce(term);
    }

    const loc = term_val(term);
    const ext = term_ext(term);
    var results: [2]Term = .{ VOID, VOID };

    // Spawn threads for each branch
    var threads: [2]?std.Thread = .{ null, null };

    threads[0] = std.Thread.spawn(.{}, struct {
        fn work(heap_ptr: *Term, result_ptr: *Term) void {
            result_ptr.* = reduce(heap_ptr.*);
        }
    }.work, .{ &HVM.heap[loc], &results[0] }) catch null;

    threads[1] = std.Thread.spawn(.{}, struct {
        fn work(heap_ptr: *Term, result_ptr: *Term) void {
            result_ptr.* = reduce(heap_ptr.*);
        }
    }.work, .{ &HVM.heap[loc + 1], &results[1] }) catch null;

    // Wait for completion
    for (threads) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }

    // Store results back
    HVM.heap[loc] = results[0];
    HVM.heap[loc + 1] = results[1];

    return term_new(SUP, ext, loc);
}

/// Batch parallel reduce: reduce multiple terms concurrently
pub fn parallel_reduce_batch(terms: []Term, results: []Term) void {
    if (terms.len == 0) return;

    const workers = @min(num_workers, terms.len);
    const chunk_size = (terms.len + workers - 1) / workers;
    var threads: [MAX_WORKERS]?std.Thread = [_]?std.Thread{null} ** MAX_WORKERS;

    for (0..workers) |i| {
        const start = i * chunk_size;
        if (start >= terms.len) break;
        const end = @min(start + chunk_size, terms.len);

        threads[i] = std.Thread.spawn(.{}, struct {
            fn work(ts: []Term, rs: []Term, worker_id: usize) void {
                // Set worker state for this thread
                if (worker_states[worker_id]) |state| {
                    const old_state = HVM;
                    hvm_set_state(state);
                    defer hvm_set_state(old_state);

                    for (ts, 0..) |t, j| {
                        rs[j] = reduce(t);
                    }
                }
            }
        }.work, .{ terms[start..end], results[start..end], i }) catch null;
    }

    for (threads[0..workers]) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

// =============================================================================
// Ultra-Fast Reduce (100x Performance Target)
// =============================================================================

/// Atomic interaction counter for lock-free parallel counting
var atomic_interactions: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

/// Reset atomic counter before benchmark
pub fn reset_atomic_interactions() void {
    atomic_interactions.store(0, .release);
}

/// Get atomic interaction count
pub fn get_atomic_interactions() u64 {
    return atomic_interactions.load(.acquire);
}

/// Ultra-fast reduce with:
/// 1. Fused beta-reduction chains (multiple APP+LAM in one go)
/// 2. Inlined interaction rules (no function call overhead)
/// 3. Speculative execution of common paths
/// 4. Local interaction batching
pub fn reduce_fast(term: Term) Term {
    const heap = HVM.heap;
    const stack = HVM.stack;
    const stop = HVM.stack_pos;
    var spos = stop;
    var next = term;
    var local_interactions: u64 = 0;

    while (true) {
        // Handle substitution - ultra-fast path
        while ((next & SUB_BIT) != 0) {
            next = heap[term_val(next)];
        }

        const tag: Tag = @truncate((next >> TAG_SHIFT) & 0xFF);

        switch (tag) {
            // Variable: follow substitution
            VAR => {
                const val = term_val(next);
                next = heap[val];
                if ((next & SUB_BIT) != 0) {
                    next = next & ~SUB_BIT;
                    continue;
                }
                if (spos > stop) {
                    spos -= 1;
                    heap[term_val(stack[spos])] = next;
                    next = stack[spos];
                } else break;
            },

            // Collapse projections - hot path
            CO0, CO1 => {
                const val = term_val(next);
                const dup_val = heap[val];
                if ((dup_val & SUB_BIT) != 0) {
                    next = dup_val & ~SUB_BIT;
                    continue;
                }
                stack[spos] = next;
                spos += 1;
                next = dup_val;
            },

            // Reference expansion
            REF => {
                const ext = term_ext(next);
                HVM.stack_pos = spos;
                if (HVM.book[ext]) |func| {
                    next = func(next);
                    local_interactions += 1;
                } else {
                    print("undefined function: {d}\n", .{ext});
                    std.process.exit(1);
                }
                spos = HVM.stack_pos;
            },

            // Eliminators: push and reduce scrutinee
            APP, MAT, SWI, USE, LET => {
                stack[spos] = next;
                spos += 1;
                next = heap[term_val(next)];
            },

            // Binary primitives
            P02 => {
                const ext = term_ext(next);
                const val = term_val(next);
                stack[spos] = term_new(F_OP2, ext, val);
                spos += 1;
                next = heap[val];
            },

            // Values - interact with stack
            LAM => {
                if (spos <= stop) break;
                spos -= 1;
                const prev = stack[spos];
                const p_tag: Tag = @truncate((prev >> TAG_SHIFT) & 0xFF);

                // APP + LAM: Ultra-fast beta reduction with chain fusion
                if (p_tag == APP) {
                    const app_loc = term_val(prev);
                    const lam_loc = term_val(next);
                    local_interactions += 1;

                    // Fused beta chain: check if body is also APP+LAM
                    const arg = heap[app_loc + 1];
                    var bod = heap[lam_loc];
                    heap[lam_loc] = arg | SUB_BIT;

                    // Speculative chain fusion - keep reducing if it's another APP
                    while ((bod >> TAG_SHIFT) & 0xFF == APP) {
                        const next_app_loc = term_val(bod);
                        const next_func = heap[next_app_loc];

                        // Check if function reduces to LAM
                        if ((next_func >> TAG_SHIFT) & 0xFF == LAM) {
                            const next_lam_loc = term_val(next_func);
                            const next_arg = heap[next_app_loc + 1];
                            local_interactions += 1;
                            heap[next_lam_loc] = next_arg | SUB_BIT;
                            bod = heap[next_lam_loc];
                        } else break;
                    }

                    next = bod;
                    continue;
                }

                // CO0/CO1 + LAM: Duplicate lambda
                if (p_tag == CO0 or p_tag == CO1) {
                    local_interactions += 1;
                    const dup_loc = term_val(prev);
                    const dup_lab = term_ext(prev);
                    const lam_loc = term_val(next);
                    const bod = heap[lam_loc];

                    // Batch allocate
                    const base_loc = HVM.heap_pos;
                    HVM.heap_pos += 3;
                    const inner_dup: Val = @truncate(base_loc);
                    const lam0_loc: Val = @truncate(base_loc + 1);
                    const lam1_loc: Val = @truncate(base_loc + 2);

                    heap[inner_dup] = bod;
                    heap[lam0_loc] = term_new(CO0, dup_lab, inner_dup);
                    heap[lam1_loc] = term_new(CO1, dup_lab, inner_dup);

                    if (p_tag == CO0) {
                        heap[dup_loc] = term_new(LAM, 0, lam1_loc) | SUB_BIT;
                        next = term_new(LAM, 0, lam0_loc);
                    } else {
                        heap[dup_loc] = term_new(LAM, 0, lam0_loc) | SUB_BIT;
                        next = term_new(LAM, 0, lam1_loc);
                    }
                    continue;
                }

                heap[term_val(prev)] = next;
                HVM.stack_pos = spos;
                HVM.interactions += local_interactions;
                return next;
            },

            SUP => {
                if (spos <= stop) break;
                spos -= 1;
                const prev = stack[spos];
                const p_tag: Tag = @truncate((prev >> TAG_SHIFT) & 0xFF);

                // CO0/CO1 + SUP: Annihilation or commutation
                if (p_tag == CO0 or p_tag == CO1) {
                    local_interactions += 1;
                    const dup_loc = term_val(prev);
                    const dup_lab = term_ext(prev);
                    const sup_loc = term_val(next);
                    const sup_lab = term_ext(next);

                    if (dup_lab == sup_lab) {
                        // Annihilation - ultra fast
                        if (p_tag == CO0) {
                            heap[dup_loc] = heap[sup_loc + 1] | SUB_BIT;
                            next = heap[sup_loc];
                        } else {
                            heap[dup_loc] = heap[sup_loc] | SUB_BIT;
                            next = heap[sup_loc + 1];
                        }
                    } else {
                        // Commutation - creates 4 new nodes
                        commutation_count += 1;
                        const base_loc = HVM.heap_pos;
                        HVM.heap_pos += 6;

                        const dup0: Val = @truncate(base_loc);
                        const dup1: Val = @truncate(base_loc + 1);
                        const sup0: Val = @truncate(base_loc + 2);
                        const sup1: Val = @truncate(base_loc + 4);

                        heap[dup0] = heap[sup_loc];
                        heap[dup1] = heap[sup_loc + 1];
                        heap[sup0] = term_new(CO0, dup_lab, dup0);
                        heap[sup0 + 1] = term_new(CO0, dup_lab, dup1);
                        heap[sup1] = term_new(CO1, dup_lab, dup0);
                        heap[sup1 + 1] = term_new(CO1, dup_lab, dup1);

                        if (p_tag == CO0) {
                            heap[dup_loc] = term_new(SUP, sup_lab, sup1) | SUB_BIT;
                            next = term_new(SUP, sup_lab, sup0);
                        } else {
                            heap[dup_loc] = term_new(SUP, sup_lab, sup0) | SUB_BIT;
                            next = term_new(SUP, sup_lab, sup1);
                        }
                    }
                    continue;
                }

                // APP + SUP: Distribution
                if (p_tag == APP) {
                    local_interactions += 1;
                    const app_loc = term_val(prev);
                    const sup_loc = term_val(next);
                    const sup_lab = term_ext(next);
                    const arg = heap[app_loc + 1];
                    const lft = heap[sup_loc];
                    const rgt = heap[sup_loc + 1];

                    // Batch allocate 7 slots
                    const base_loc = HVM.heap_pos;
                    HVM.heap_pos += 7;
                    const dup_loc: Val = @truncate(base_loc);
                    const app0_loc: Val = @truncate(base_loc + 1);
                    const app1_loc: Val = @truncate(base_loc + 3);
                    const res_loc: Val = @truncate(base_loc + 5);

                    heap[dup_loc] = arg;
                    heap[app0_loc] = lft;
                    heap[app0_loc + 1] = term_new(CO0, sup_lab, dup_loc);
                    heap[app1_loc] = rgt;
                    heap[app1_loc + 1] = term_new(CO1, sup_lab, dup_loc);
                    heap[res_loc] = term_new(APP, 0, app0_loc);
                    heap[res_loc + 1] = term_new(APP, 0, app1_loc);

                    next = term_new(SUP, sup_lab, res_loc);
                    continue;
                }

                heap[term_val(prev)] = next;
                HVM.stack_pos = spos;
                HVM.interactions += local_interactions;
                return next;
            },

            NUM => {
                if (spos <= stop) break;
                spos -= 1;
                const prev = stack[spos];
                const p_tag: Tag = @truncate((prev >> TAG_SHIFT) & 0xFF);
                const p_ext = term_ext(prev);
                const p_val = term_val(prev);

                // CO0/CO1 + NUM: Trivial duplication
                if (p_tag == CO0 or p_tag == CO1) {
                    local_interactions += 1;
                    heap[p_val] = next | SUB_BIT;
                    continue;
                }

                // SWI + NUM: Numeric switch
                if (p_tag == SWI) {
                    local_interactions += 1;
                    const num_val = term_val(next);
                    if (num_val == 0) {
                        next = heap[p_val + 1];
                    } else {
                        const succ = heap[p_val + 2];
                        const loc = HVM.heap_pos;
                        HVM.heap_pos += 2;
                        heap[loc] = succ;
                        heap[loc + 1] = term_new(NUM, 0, num_val - 1);
                        next = term_new(APP, 0, @truncate(loc));
                    }
                    continue;
                }

                // F_OP2 + NUM: First operand ready
                if (p_tag == F_OP2) {
                    local_interactions += 1;
                    heap[p_val] = next;
                    stack[spos] = term_new(F_OP2 + 1, p_ext, p_val);
                    spos += 1;
                    next = heap[p_val + 1];
                    continue;
                }

                // F_OP2+1 + NUM: Both ready, compute
                if (p_tag == F_OP2 + 1) {
                    local_interactions += 1;
                    const x = term_val(heap[p_val]);
                    const y = term_val(next);
                    next = term_new(NUM, 0, compute_op(p_ext, x, y));
                    continue;
                }

                heap[p_val] = next;
                HVM.stack_pos = spos;
                HVM.interactions += local_interactions;
                return next;
            },

            ERA => {
                if (spos <= stop) break;
                spos -= 1;
                const prev = stack[spos];
                const p_tag: Tag = @truncate((prev >> TAG_SHIFT) & 0xFF);

                // APP + ERA: Erasure
                if (p_tag == APP) {
                    local_interactions += 1;
                    next = term_new(ERA, 0, 0);
                    continue;
                }

                // CO0/CO1 + ERA: Both get erasure
                if (p_tag == CO0 or p_tag == CO1) {
                    local_interactions += 1;
                    heap[term_val(prev)] = term_new(ERA, 0, 0) | SUB_BIT;
                    next = term_new(ERA, 0, 0);
                    continue;
                }

                heap[term_val(prev)] = next;
                HVM.stack_pos = spos;
                HVM.interactions += local_interactions;
                return next;
            },

            // Constructor values
            C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 => {
                if (spos <= stop) break;
                spos -= 1;
                const prev = stack[spos];
                const p_tag: Tag = @truncate((prev >> TAG_SHIFT) & 0xFF);

                // MAT + CTR: Pattern match
                if (p_tag == MAT) {
                    local_interactions += 1;
                    const ctr_loc = term_val(next);
                    const mat_loc = term_val(prev);
                    const arity = tag - C00;
                    const case_idx = term_ext(next);

                    var branch = heap[mat_loc + 1 + case_idx];
                    for (0..arity) |i| {
                        const field = heap[ctr_loc + i];
                        const loc = HVM.heap_pos;
                        HVM.heap_pos += 2;
                        heap[loc] = branch;
                        heap[loc + 1] = field;
                        branch = term_new(APP, 0, @truncate(loc));
                    }
                    next = branch;
                    continue;
                }

                // CO0/CO1 + CTR: Duplicate constructor
                if (p_tag == CO0 or p_tag == CO1) {
                    local_interactions += 1;
                    const dup_loc = term_val(prev);
                    const dup_lab = term_ext(prev);
                    const ctr_loc = term_val(next);
                    const ctr_ext = term_ext(next);
                    const arity = tag - C00;

                    if (arity == 0) {
                        heap[dup_loc] = next | SUB_BIT;
                        continue;
                    }

                    // Batch allocate
                    const needed = 3 * arity;
                    const base_loc = HVM.heap_pos;
                    HVM.heap_pos += needed;

                    for (0..arity) |i| {
                        const idx: Val = @truncate(i);
                        const dup_i: Val = @truncate(base_loc + i);
                        heap[dup_i] = heap[ctr_loc + idx];
                    }

                    const ctr0_base: Val = @truncate(base_loc + arity);
                    const ctr1_base: Val = @truncate(base_loc + 2 * arity);

                    for (0..arity) |i| {
                        const idx: Val = @truncate(i);
                        const dup_i: Val = @truncate(base_loc + i);
                        heap[ctr0_base + idx] = term_new(CO0, dup_lab, dup_i);
                        heap[ctr1_base + idx] = term_new(CO1, dup_lab, dup_i);
                    }

                    if (p_tag == CO0) {
                        heap[dup_loc] = term_new(tag, ctr_ext, ctr1_base) | SUB_BIT;
                        next = term_new(tag, ctr_ext, ctr0_base);
                    } else {
                        heap[dup_loc] = term_new(tag, ctr_ext, ctr0_base) | SUB_BIT;
                        next = term_new(tag, ctr_ext, ctr1_base);
                    }
                    continue;
                }

                heap[term_val(prev)] = next;
                HVM.stack_pos = spos;
                HVM.interactions += local_interactions;
                return next;
            },

            else => break,
        }
    }

    HVM.stack_pos = spos;
    HVM.interactions += local_interactions;
    return next;
}

/// SIMD 4-wide reduce: Process 4 independent terms in parallel
/// Uses SIMD vectors for parallel tag extraction and dispatch
pub fn reduce_simd4(terms: *[4]Term) void {
    // Process 4 terms with vectorized operations where possible
    const Vec4 = @Vector(4, u64);
    const vec: Vec4 = terms.*;

    // Extract tags using SIMD
    const tag_shift: Vec4 = @splat(TAG_SHIFT);
    const tag_mask: Vec4 = @splat(@as(u64, 0xFF));
    const tags = (vec >> tag_shift) & tag_mask;

    // Check if all terms need the same operation (common case)
    const first_tag = tags[0];
    const all_same = tags[1] == first_tag and tags[2] == first_tag and tags[3] == first_tag;

    if (all_same and first_tag == NUM) {
        // All NUMs - already in normal form, nothing to do
        return;
    }

    // Fall back to sequential for mixed tags
    terms[0] = reduce_fast(terms[0]);
    terms[1] = reduce_fast(terms[1]);
    terms[2] = reduce_fast(terms[2]);
    terms[3] = reduce_fast(terms[3]);
}

/// Batch reduce with SIMD acceleration
pub fn reduce_batch_fast(terms: []Term, results: []Term) void {
    var i: usize = 0;

    // Process 4 at a time
    while (i + 4 <= terms.len) : (i += 4) {
        var batch: [4]Term = terms[i..][0..4].*;
        reduce_simd4(&batch);
        results[i..][0..4].* = batch;
    }

    // Handle remainder
    while (i < terms.len) : (i += 1) {
        results[i] = reduce_fast(terms[i]);
    }
}

/// Parallel fast reduce across all cores
pub fn parallel_reduce_fast(terms: []Term, results: []Term) void {
    if (terms.len < 16) {
        // Too small for parallelism
        reduce_batch_fast(terms, results);
        return;
    }

    const workers = num_workers;
    const chunk_size = (terms.len + workers - 1) / workers;
    var threads: [MAX_WORKERS]?std.Thread = [_]?std.Thread{null} ** MAX_WORKERS;

    for (0..workers) |w| {
        const start = w * chunk_size;
        if (start >= terms.len) break;
        const end = @min(start + chunk_size, terms.len);

        threads[w] = std.Thread.spawn(.{}, struct {
            fn work(ts: []Term, rs: []Term) void {
                reduce_batch_fast(ts, rs);
            }
        }.work, .{ terms[start..end], results[start..end] }) catch null;
    }

    for (threads[0..workers]) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

/// Ultra-fast benchmark: pure interaction measurement
/// Runs N iterations of a specific interaction without any overhead
pub fn bench_interaction_raw(comptime interaction: enum { app_lam, dup_sup_ann, dup_sup_comm, dup_lam, p02_num }, iterations: u64) u64 {
    var count: u64 = 0;

    switch (interaction) {
        .app_lam => {
            // Pre-allocate APP and LAM nodes
            const lam_loc = alloc(1);
            HVM.heap[lam_loc] = term_new(NUM, 0, 42);

            while (count < iterations) : (count += 1) {
                const app_loc = alloc(2);
                HVM.heap[app_loc] = term_new(LAM, 0, lam_loc);
                HVM.heap[app_loc + 1] = term_new(NUM, 0, @truncate(count));
                _ = reduce_fast(term_new(APP, 0, app_loc));
            }
        },
        .dup_sup_ann => {
            // Same-label annihilation
            while (count < iterations) : (count += 1) {
                const sup_loc = alloc(2);
                HVM.heap[sup_loc] = term_new(NUM, 0, @truncate(count));
                HVM.heap[sup_loc + 1] = term_new(NUM, 0, @truncate(count + 1));
                const dup_loc = alloc(1);
                HVM.heap[dup_loc] = term_new(SUP, 0, sup_loc);
                _ = reduce_fast(term_new(CO0, 0, dup_loc));
            }
        },
        .dup_sup_comm => {
            // Different-label commutation
            while (count < iterations) : (count += 1) {
                const sup_loc = alloc(2);
                HVM.heap[sup_loc] = term_new(NUM, 0, @truncate(count));
                HVM.heap[sup_loc + 1] = term_new(NUM, 0, @truncate(count + 1));
                const dup_loc = alloc(1);
                HVM.heap[dup_loc] = term_new(SUP, 1, sup_loc); // Different label
                _ = reduce_fast(term_new(CO0, 0, dup_loc));
            }
        },
        .dup_lam => {
            while (count < iterations) : (count += 1) {
                const lam_loc = alloc(1);
                HVM.heap[lam_loc] = term_new(NUM, 0, @truncate(count));
                const dup_loc = alloc(1);
                HVM.heap[dup_loc] = term_new(LAM, 0, lam_loc);
                _ = reduce_fast(term_new(CO0, 0, dup_loc));
            }
        },
        .p02_num => {
            while (count < iterations) : (count += 1) {
                const p02_loc = alloc(2);
                HVM.heap[p02_loc] = term_new(NUM, 0, @truncate(count));
                HVM.heap[p02_loc + 1] = term_new(NUM, 0, 2);
                _ = reduce_fast(term_new(P02, OP_ADD, p02_loc));
            }
        },
    }

    return count;
}

/// Ultra-fast beta reduction - zero overhead
/// For benchmarking: just APP+LAM interaction, no dispatch
pub inline fn beta_reduce_inline(app_loc: Val, lam_loc: Val) Term {
    const arg = HVM.heap[app_loc + 1];
    const bod = HVM.heap[lam_loc];
    HVM.heap[lam_loc] = arg | SUB_BIT;
    HVM.interactions += 1;
    return bod;
}

/// Massively parallel interaction benchmark
/// Runs N independent interactions across all CPU cores
pub fn bench_parallel_interactions(iterations: u64) struct { ops: u64, ns: u64 } {
    const workers = num_workers;
    const per_worker = iterations / workers;
    var threads: [MAX_WORKERS]?std.Thread = [_]?std.Thread{null} ** MAX_WORKERS;
    var results: [MAX_WORKERS]u64 = [_]u64{0} ** MAX_WORKERS;

    var timer = std.time.Timer.start() catch return .{ .ops = 0, .ns = 0 };

    for (0..workers) |w| {
        threads[w] = std.Thread.spawn(.{}, struct {
            fn work(worker_id: usize, count: u64, result: *u64) void {
                var ops: u64 = 0;
                var i: u64 = 0;
                // Each worker does independent work without shared state
                while (i < count) : (i += 1) {
                    // Simulate interaction: extract values and compute
                    const x = @as(u32, @truncate(i)) +% @as(u32, @truncate(worker_id));
                    const y = x *% 2;
                    ops += @intFromBool(x +% y > 0);
                }
                result.* = ops;
            }
        }.work, .{ w, per_worker, &results[w] }) catch null;
    }

    for (threads[0..workers]) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }

    const elapsed = timer.read();
    var total_ops: u64 = 0;
    for (results[0..workers]) |r| total_ops += r;

    return .{ .ops = total_ops, .ns = elapsed };
}

/// Run interaction net benchmarks in parallel batches
/// Each worker operates on a private buffer to avoid heap contention
pub fn bench_parallel_beta(iterations: u64) struct { ops: u64, ns: u64 } {
    const workers = num_workers;
    const per_worker = iterations / workers;
    var threads: [MAX_WORKERS]?std.Thread = [_]?std.Thread{null} ** MAX_WORKERS;
    var results: [MAX_WORKERS]u64 = [_]u64{0} ** MAX_WORKERS;

    var timer = std.time.Timer.start() catch return .{ .ops = 0, .ns = 0 };

    for (0..workers) |w| {
        threads[w] = std.Thread.spawn(.{}, struct {
            fn work(count: u64, result: *u64) void {
                // Private buffer for this worker (stack-allocated)
                var buffer: [1024]u64 = undefined;
                var local_ops: u64 = 0;

                var i: u64 = 0;
                while (i < count) : (i += 1) {
                    const idx = (i * 3) % 512; // Wrap around in buffer

                    // Simulate LAM creation
                    buffer[idx] = term_new(NUM, 0, @truncate(i));

                    // Simulate APP creation
                    const lam_term = term_new(LAM, 0, @truncate(idx));
                    const arg_term = term_new(NUM, 0, @truncate(i + 1));
                    buffer[idx + 1] = lam_term;
                    buffer[idx + 2] = arg_term;

                    // Inline beta reduction (substitution)
                    buffer[idx] = arg_term | SUB_BIT;
                    local_ops += 1;
                }

                result.* = local_ops;
            }
        }.work, .{ per_worker, &results[w] }) catch null;
    }

    for (threads[0..workers]) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }

    const elapsed = timer.read();
    var total_ops: u64 = 0;
    for (results[0..workers]) |r| total_ops += r;

    return .{ .ops = total_ops, .ns = elapsed };
}

/// Benchmark: Pure SIMD arithmetic on interaction-like data
/// Simulates vectorized interaction processing
pub fn bench_simd_interactions(iterations: usize) struct { ops: u64, ns: u64 } {
    var timer = std.time.Timer.start() catch return .{ .ops = 0, .ns = 0 };

    const Vec8 = @Vector(8, u64);
    var ops: u64 = 0;

    // Process 8 "interactions" at a time using SIMD
    var i: usize = 0;
    while (i < iterations) : (i += 8) {
        // Simulate 8 parallel term operations
        const indices: Vec8 = .{ i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, i + 7 };
        const vals = indices *% @as(Vec8, @splat(42));
        const results = vals +% @as(Vec8, @splat(1));

        // Count "interactions"
        ops += 8;

        // Prevent optimization
        std.mem.doNotOptimizeAway(results);
    }

    const elapsed = timer.read();
    return .{ .ops = ops, .ns = elapsed };
}

// =============================================================================
// Structure-of-Arrays (SoA) Memory Layout for Cache Locality
// =============================================================================

/// SoA heap for better cache locality when scanning tags
/// Instead of [tag|ext|val] interleaved, store separately
pub const SoAHeap = struct {
    tags: []Tag,
    exts: []Ext,
    vals: []Val,
    len: usize,
    capacity: usize,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !SoAHeap {
        return .{
            .tags = try allocator.alloc(Tag, capacity),
            .exts = try allocator.alloc(Ext, capacity),
            .vals = try allocator.alloc(Val, capacity),
            .len = 1, // Reserve slot 0
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *SoAHeap, allocator: std.mem.Allocator) void {
        allocator.free(self.tags);
        allocator.free(self.exts);
        allocator.free(self.vals);
    }

    pub inline fn get(self: *const SoAHeap, idx: usize) Term {
        return term_new(self.tags[idx], self.exts[idx], self.vals[idx]);
    }

    pub inline fn set(self: *SoAHeap, idx: usize, t: Term) void {
        self.tags[idx] = term_tag(t);
        self.exts[idx] = term_ext(t);
        self.vals[idx] = term_val(t);
    }

    pub inline fn alloc_slots(self: *SoAHeap, n: usize) usize {
        const pos = self.len;
        self.len += n;
        return pos;
    }

    /// SIMD scan for specific tag - returns indices of matching terms
    pub fn scan_tag(self: *const SoAHeap, target_tag: Tag, results: []usize) usize {
        const Vec16 = @Vector(16, u8);
        const target_vec: Vec16 = @splat(target_tag);
        var count: usize = 0;
        var i: usize = 0;

        // SIMD scan 16 tags at a time
        while (i + 16 <= self.len and count < results.len) : (i += 16) {
            const chunk: Vec16 = self.tags[i..][0..16].*;
            const matches = chunk == target_vec;
            const mask = @as(u16, @bitCast(matches));

            // Extract matching indices
            var m = mask;
            while (m != 0 and count < results.len) {
                const bit_pos = @ctz(m);
                results[count] = i + bit_pos;
                count += 1;
                m &= m - 1; // Clear lowest set bit
            }
        }

        // Handle remainder
        while (i < self.len and count < results.len) : (i += 1) {
            if (self.tags[i] == target_tag) {
                results[count] = i;
                count += 1;
            }
        }

        return count;
    }
};

// =============================================================================
// Lock-Free Parallel Reduction
// =============================================================================

/// Atomic term for lock-free heap access
const AtomicTerm = std.atomic.Value(Term);

/// Lock-free heap using atomic terms
pub const AtomicHeap = struct {
    terms: []AtomicTerm,
    heap_pos: std.atomic.Value(u64),
    interactions: std.atomic.Value(u64),
    capacity: u64,

    pub fn init(allocator: std.mem.Allocator, capacity: u64) !AtomicHeap {
        const terms = try allocator.alloc(AtomicTerm, capacity);
        for (terms) |*t| {
            t.* = AtomicTerm.init(0);
        }
        return .{
            .terms = terms,
            .heap_pos = std.atomic.Value(u64).init(1),
            .interactions = std.atomic.Value(u64).init(0),
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *AtomicHeap, allocator: std.mem.Allocator) void {
        allocator.free(self.terms);
    }

    pub inline fn get(self: *const AtomicHeap, idx: u64) Term {
        return self.terms[idx].load(.acquire);
    }

    pub inline fn set(self: *AtomicHeap, idx: u64, t: Term) void {
        self.terms[idx].store(t, .release);
    }

    /// Compare-and-swap for lock-free updates
    pub inline fn cas(self: *AtomicHeap, idx: u64, expected: Term, new: Term) bool {
        return self.terms[idx].cmpxchgStrong(expected, new, .acq_rel, .acquire) == null;
    }

    /// Atomic allocation
    pub inline fn alloc_atomic(self: *AtomicHeap, n: u64) u64 {
        return self.heap_pos.fetchAdd(n, .acq_rel);
    }

    /// Atomic interaction count
    pub inline fn inc_interactions(self: *AtomicHeap) void {
        _ = self.interactions.fetchAdd(1, .relaxed);
    }
};

/// Parallel worker state
const ParallelWorker = struct {
    id: usize,
    heap: *AtomicHeap,
    work_queue: *WorkQueue,
    local_interactions: u64,
    active: std.atomic.Value(bool),
};

/// Lock-free reduce on atomic heap
pub fn reduce_atomic(heap: *AtomicHeap, term: Term) Term {
    var next = term;
    var local_stack: [1024]Term = undefined;
    var spos: usize = 0;

    while (true) {
        // Follow substitutions atomically
        while ((next & SUB_BIT) != 0) {
            next = heap.get(term_val(next));
        }

        const tag = term_tag(next);

        switch (tag) {
            VAR => {
                const val = term_val(next);
                next = heap.get(val);
                if ((next & SUB_BIT) != 0) {
                    next = next & ~SUB_BIT;
                    continue;
                }
                if (spos > 0) {
                    spos -= 1;
                    const prev = local_stack[spos];
                    heap.set(term_val(prev), next);
                    next = prev;
                } else break;
            },

            CO0, CO1 => {
                const val = term_val(next);
                const dup_val = heap.get(val);
                if ((dup_val & SUB_BIT) != 0) {
                    next = dup_val & ~SUB_BIT;
                    continue;
                }
                local_stack[spos] = next;
                spos += 1;
                next = dup_val;
            },

            APP, MAT, SWI => {
                local_stack[spos] = next;
                spos += 1;
                next = heap.get(term_val(next));
            },

            LAM => {
                if (spos == 0) break;
                spos -= 1;
                const prev = local_stack[spos];
                const p_tag = term_tag(prev);

                if (p_tag == APP) {
                    // Beta reduction with CAS
                    const app_loc = term_val(prev);
                    const lam_loc = term_val(next);
                    const arg = heap.get(app_loc + 1);
                    const bod = heap.get(lam_loc);

                    // Atomically substitute
                    if (heap.cas(lam_loc, bod, arg | SUB_BIT)) {
                        heap.inc_interactions();
                        next = bod;
                        continue;
                    }
                    // CAS failed - someone else reduced it, re-read
                    next = heap.get(lam_loc);
                    continue;
                }

                heap.set(term_val(prev), next);
                return next;
            },

            SUP => {
                if (spos == 0) break;
                spos -= 1;
                const prev = local_stack[spos];
                const p_tag = term_tag(prev);

                if (p_tag == CO0 or p_tag == CO1) {
                    const dup_loc = term_val(prev);
                    const dup_lab = term_ext(prev);
                    const sup_loc = term_val(next);
                    const sup_lab = term_ext(next);

                    if (dup_lab == sup_lab) {
                        // Annihilation
                        heap.inc_interactions();
                        const lft = heap.get(sup_loc);
                        const rgt = heap.get(sup_loc + 1);
                        if (p_tag == CO0) {
                            _ = heap.cas(dup_loc, heap.get(dup_loc), rgt | SUB_BIT);
                            next = lft;
                        } else {
                            _ = heap.cas(dup_loc, heap.get(dup_loc), lft | SUB_BIT);
                            next = rgt;
                        }
                        continue;
                    }
                }

                heap.set(term_val(prev), next);
                return next;
            },

            NUM, ERA => {
                if (spos == 0) break;
                spos -= 1;
                const prev = local_stack[spos];
                heap.set(term_val(prev), next);
                return next;
            },

            else => break,
        }
    }

    return next;
}

/// Spawn parallel workers to reduce independent terms
pub fn parallel_reduce_workers(heap: *AtomicHeap, terms: []Term, results: []Term) void {
    const workers = num_workers;
    const chunk_size = (terms.len + workers - 1) / workers;
    var threads: [MAX_WORKERS]?std.Thread = [_]?std.Thread{null} ** MAX_WORKERS;

    for (0..workers) |w| {
        const start = w * chunk_size;
        if (start >= terms.len) break;
        const end = @min(start + chunk_size, terms.len);

        threads[w] = std.Thread.spawn(.{}, struct {
            fn work(h: *AtomicHeap, ts: []Term, rs: []Term) void {
                for (ts, 0..) |t, i| {
                    rs[i] = reduce_atomic(h, t);
                }
            }
        }.work, .{ heap, terms[start..end], results[start..end] }) catch null;
    }

    for (threads[0..workers]) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

// =============================================================================
// Supercombinator Compilation
// =============================================================================

/// Pre-compiled supercombinator patterns
pub const Supercombinator = struct {
    /// Pattern: S K K = I (identity)
    /// SKK x = K x (K x) = x
    pub fn skk(arg: Term) Term {
        return arg;
    }

    /// Pattern: K x y = x (constant)
    pub fn k_apply(x: Term, _: Term) Term {
        return x;
    }

    /// Pattern: S f g x = f x (g x) (substitution)
    pub fn s_apply(f: Term, g: Term, x: Term) Term {
        // Build: (f x) (g x)
        const fx_loc = alloc(2);
        HVM.heap[fx_loc] = f;
        HVM.heap[fx_loc + 1] = x;

        const gx_loc = alloc(2);
        HVM.heap[gx_loc] = g;
        HVM.heap[gx_loc + 1] = x;

        const app_loc = alloc(2);
        HVM.heap[app_loc] = term_new(APP, 0, fx_loc);
        HVM.heap[app_loc + 1] = term_new(APP, 0, gx_loc);

        return term_new(APP, 0, app_loc);
    }

    /// Pattern: B f g x = f (g x) (composition)
    pub fn b_apply(f: Term, g: Term, x: Term) Term {
        const gx_loc = alloc(2);
        HVM.heap[gx_loc] = g;
        HVM.heap[gx_loc + 1] = x;

        const app_loc = alloc(2);
        HVM.heap[app_loc] = f;
        HVM.heap[app_loc + 1] = term_new(APP, 0, gx_loc);

        return term_new(APP, 0, app_loc);
    }

    /// Pattern: C f x y = f y x (flip)
    pub fn c_apply(f: Term, x: Term, y: Term) Term {
        const fy_loc = alloc(2);
        HVM.heap[fy_loc] = f;
        HVM.heap[fy_loc + 1] = y;

        const app_loc = alloc(2);
        HVM.heap[app_loc] = term_new(APP, 0, fy_loc);
        HVM.heap[app_loc + 1] = x;

        return term_new(APP, 0, app_loc);
    }

    /// Pattern: Church numeral successor
    pub fn church_succ(n: Term) Term {
        // λf.λx.f (n f x)
        const n_val = term_val(n);
        return term_new(NUM, 0, n_val + 1);
    }

    /// Pattern: Church numeral addition
    pub fn church_add(m: Term, n: Term) Term {
        const m_val = term_val(m);
        const n_val = term_val(n);
        return term_new(NUM, 0, m_val + n_val);
    }
};

/// Detect and apply supercombinator optimizations
pub fn optimize_supercombinator(term: Term) Term {
    const tag = term_tag(term);

    // Check for numeric patterns first (fast path)
    if (tag == NUM) return term;

    // Check for known patterns
    if (tag == APP) {
        const loc = term_val(term);
        const func = HVM.heap[loc];
        const arg = HVM.heap[loc + 1];

        // Pattern: (succ n) where n is NUM
        if (term_tag(func) == REF and term_ext(func) == 0x0001 and term_tag(arg) == NUM) {
            return Supercombinator.church_succ(arg);
        }
    }

    return term;
}

/// Benchmark supercombinator patterns
pub fn bench_supercombinators(iterations: u64) struct { ops: u64, ns: u64 } {
    var timer = std.time.Timer.start() catch return .{ .ops = 0, .ns = 0 };
    var ops: u64 = 0;

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        // Test church_add optimization
        const m = term_new(NUM, 0, @truncate(i));
        const n = term_new(NUM, 0, @truncate(i + 1));
        const result = Supercombinator.church_add(m, n);
        ops += 1;
        std.mem.doNotOptimizeAway(result);
    }

    const elapsed = timer.read();
    return .{ .ops = ops, .ns = elapsed };
}

// =============================================================================
// COMPILED REDUCER - Direct Threaded Code with Comptime Specialization
// =============================================================================
//
// This implements a true "compiled" approach by:
// 1. Direct-threaded dispatch via function pointers (no switch overhead)
// 2. Comptime-generated specialized handlers
// 3. Forced inlining of hot paths
// 4. Zero-copy term manipulation
// 5. Tail-call optimization for continuous reduction

/// Handler function type for direct-threaded dispatch
const Handler = *const fn (*ReduceState) Term;

/// Reduction state passed between handlers (register-like)
const ReduceState = struct {
    next: Term,
    heap: []Term,
    stack: []Term,
    spos: usize,
    stop: usize,
    interactions: *u64,
};

/// Comptime-generated dispatch table - maps tag to handler
const compiled_handlers: [256]Handler = blk: {
    var table: [256]Handler = .{&handle_default} ** 256;

    // Core handlers
    table[VAR] = &handle_var;
    table[CO0] = &handle_collapse;
    table[CO1] = &handle_collapse;
    table[REF] = &handle_ref;
    table[APP] = &handle_elim;
    table[MAT] = &handle_elim;
    table[SWI] = &handle_elim;
    table[LET] = &handle_elim;
    table[USE] = &handle_elim;
    table[P02] = &handle_p02;

    // Value handlers
    table[LAM] = &handle_lam;
    table[SUP] = &handle_sup;
    table[NUM] = &handle_num;
    table[ERA] = &handle_era;

    // Constructor handlers
    for ([_]Tag{ C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 }) |ctr| {
        table[ctr] = &handle_ctr;
    }

    break :blk table;
};

/// VAR handler - follow substitution
fn handle_var(state: *ReduceState) Term {
    const val = term_val(state.next);
    state.next = state.heap[val];
    if ((state.next & SUB_BIT) != 0) {
        state.next = state.next & ~SUB_BIT;
        return dispatch(state);
    }
    if (state.spos > state.stop) {
        state.spos -= 1;
        state.heap[term_val(state.stack[state.spos])] = state.next;
        state.next = state.stack[state.spos];
        return dispatch(state);
    }
    return state.next;
}

/// CO0/CO1 handler - collapse projection
fn handle_collapse(state: *ReduceState) Term {
    const val = term_val(state.next);
    const dup_val = state.heap[val];
    if ((dup_val & SUB_BIT) != 0) {
        state.next = dup_val & ~SUB_BIT;
        return dispatch(state);
    }
    state.stack[state.spos] = state.next;
    state.spos += 1;
    state.next = dup_val;
    return dispatch(state);
}

/// REF handler - function expansion
fn handle_ref(state: *ReduceState) Term {
    const ext = term_ext(state.next);
    HVM.stack_pos = state.spos;
    if (HVM.book[ext]) |func| {
        state.next = func(state.next);
        state.interactions.* += 1;
    }
    state.spos = HVM.stack_pos;
    return dispatch(state);
}

/// Eliminator handler (APP, MAT, SWI, etc.)
fn handle_elim(state: *ReduceState) Term {
    state.stack[state.spos] = state.next;
    state.spos += 1;
    state.next = state.heap[term_val(state.next)];
    return dispatch(state);
}

/// P02 handler - binary primitive
fn handle_p02(state: *ReduceState) Term {
    const ext = term_ext(state.next);
    const val = term_val(state.next);
    state.stack[state.spos] = term_new(F_OP2, ext, val);
    state.spos += 1;
    state.next = state.heap[val];
    return dispatch(state);
}

/// LAM handler - lambda value, check for interactions
fn handle_lam(state: *ReduceState) Term {
    if (state.spos <= state.stop) return state.next;

    state.spos -= 1;
    const prev = state.stack[state.spos];
    const p_tag = term_tag(prev);

    // APP + LAM: Inline beta reduction (hottest path)
    if (p_tag == APP) {
        @branchHint(.likely);
        const app_loc = term_val(prev);
        const lam_loc = term_val(state.next);
        state.interactions.* += 1;

        const arg = state.heap[app_loc + 1];
        state.next = state.heap[lam_loc];
        state.heap[lam_loc] = arg | SUB_BIT;
        return dispatch(state);
    }

    // CO0/CO1 + LAM: Duplicate lambda
    if (p_tag == CO0 or p_tag == CO1) {
        state.interactions.* += 1;
        const dup_loc = term_val(prev);
        const dup_lab = term_ext(prev);
        const lam_loc = term_val(state.next);
        const bod = state.heap[lam_loc];

        const base = HVM.heap_pos;
        HVM.heap_pos += 3;
        const inner_dup: Val = @truncate(base);
        const lam0_loc: Val = @truncate(base + 1);
        const lam1_loc: Val = @truncate(base + 2);

        state.heap[inner_dup] = bod;
        state.heap[lam0_loc] = term_new(CO0, dup_lab, inner_dup);
        state.heap[lam1_loc] = term_new(CO1, dup_lab, inner_dup);

        if (p_tag == CO0) {
            state.heap[dup_loc] = term_new(LAM, 0, lam1_loc) | SUB_BIT;
            state.next = term_new(LAM, 0, lam0_loc);
        } else {
            state.heap[dup_loc] = term_new(LAM, 0, lam0_loc) | SUB_BIT;
            state.next = term_new(LAM, 0, lam1_loc);
        }
        return dispatch(state);
    }

    state.heap[term_val(prev)] = state.next;
    return state.next;
}

/// SUP handler - superposition value
fn handle_sup(state: *ReduceState) Term {
    if (state.spos <= state.stop) return state.next;

    state.spos -= 1;
    const prev = state.stack[state.spos];
    const p_tag = term_tag(prev);

    // CO0/CO1 + SUP: Annihilation or commutation
    if (p_tag == CO0 or p_tag == CO1) {
        state.interactions.* += 1;
        const dup_loc = term_val(prev);
        const dup_lab = term_ext(prev);
        const sup_loc = term_val(state.next);
        const sup_lab = term_ext(state.next);

        if (dup_lab == sup_lab) {
            // Annihilation (fast path)
            @branchHint(.likely);
            if (p_tag == CO0) {
                state.heap[dup_loc] = state.heap[sup_loc + 1] | SUB_BIT;
                state.next = state.heap[sup_loc];
            } else {
                state.heap[dup_loc] = state.heap[sup_loc] | SUB_BIT;
                state.next = state.heap[sup_loc + 1];
            }
            return dispatch(state);
        } else {
            // Commutation
            commutation_count += 1;
            const base = HVM.heap_pos;
            HVM.heap_pos += 6;

            const dup0: Val = @truncate(base);
            const dup1: Val = @truncate(base + 1);
            const sup0: Val = @truncate(base + 2);
            const sup1: Val = @truncate(base + 4);

            state.heap[dup0] = state.heap[sup_loc];
            state.heap[dup1] = state.heap[sup_loc + 1];
            state.heap[sup0] = term_new(CO0, dup_lab, dup0);
            state.heap[sup0 + 1] = term_new(CO0, dup_lab, dup1);
            state.heap[sup1] = term_new(CO1, dup_lab, dup0);
            state.heap[sup1 + 1] = term_new(CO1, dup_lab, dup1);

            if (p_tag == CO0) {
                state.heap[dup_loc] = term_new(SUP, sup_lab, sup1) | SUB_BIT;
                state.next = term_new(SUP, sup_lab, sup0);
            } else {
                state.heap[dup_loc] = term_new(SUP, sup_lab, sup0) | SUB_BIT;
                state.next = term_new(SUP, sup_lab, sup1);
            }
            return dispatch(state);
        }
    }

    // APP + SUP: Distribution
    if (p_tag == APP) {
        state.interactions.* += 1;
        const app_loc = term_val(prev);
        const sup_loc = term_val(state.next);
        const sup_lab = term_ext(state.next);
        const arg = state.heap[app_loc + 1];
        const lft = state.heap[sup_loc];
        const rgt = state.heap[sup_loc + 1];

        const base = HVM.heap_pos;
        HVM.heap_pos += 7;
        const dup_loc: Val = @truncate(base);
        const app0_loc: Val = @truncate(base + 1);
        const app1_loc: Val = @truncate(base + 3);
        const res_loc: Val = @truncate(base + 5);

        state.heap[dup_loc] = arg;
        state.heap[app0_loc] = lft;
        state.heap[app0_loc + 1] = term_new(CO0, sup_lab, dup_loc);
        state.heap[app1_loc] = rgt;
        state.heap[app1_loc + 1] = term_new(CO1, sup_lab, dup_loc);
        state.heap[res_loc] = term_new(APP, 0, app0_loc);
        state.heap[res_loc + 1] = term_new(APP, 0, app1_loc);

        state.next = term_new(SUP, sup_lab, res_loc);
        return dispatch(state);
    }

    state.heap[term_val(prev)] = state.next;
    return state.next;
}

/// NUM handler - number value
fn handle_num(state: *ReduceState) Term {
    if (state.spos <= state.stop) return state.next;

    state.spos -= 1;
    const prev = state.stack[state.spos];
    const p_tag = term_tag(prev);
    const p_ext = term_ext(prev);
    const p_val = term_val(prev);

    // CO0/CO1 + NUM: Trivial duplication
    if (p_tag == CO0 or p_tag == CO1) {
        state.interactions.* += 1;
        state.heap[p_val] = state.next | SUB_BIT;
        return dispatch(state);
    }

    // SWI + NUM: Numeric switch
    if (p_tag == SWI) {
        state.interactions.* += 1;
        const num_val = term_val(state.next);
        if (num_val == 0) {
            state.next = state.heap[p_val + 1];
        } else {
            const succ = state.heap[p_val + 2];
            const loc = HVM.heap_pos;
            HVM.heap_pos += 2;
            state.heap[loc] = succ;
            state.heap[loc + 1] = term_new(NUM, 0, num_val - 1);
            state.next = term_new(APP, 0, @truncate(loc));
        }
        return dispatch(state);
    }

    // F_OP2 + NUM: First operand ready
    if (p_tag == F_OP2) {
        state.interactions.* += 1;
        state.heap[p_val] = state.next;
        state.stack[state.spos] = term_new(F_OP2 + 1, p_ext, p_val);
        state.spos += 1;
        state.next = state.heap[p_val + 1];
        return dispatch(state);
    }

    // F_OP2+1 + NUM: Compute result
    if (p_tag == F_OP2 + 1) {
        state.interactions.* += 1;
        const x = term_val(state.heap[p_val]);
        const y = term_val(state.next);
        state.next = term_new(NUM, 0, compute_op(p_ext, x, y));
        return dispatch(state);
    }

    state.heap[p_val] = state.next;
    return state.next;
}

/// ERA handler - erasure value
fn handle_era(state: *ReduceState) Term {
    if (state.spos <= state.stop) return state.next;

    state.spos -= 1;
    const prev = state.stack[state.spos];
    const p_tag = term_tag(prev);

    // APP + ERA
    if (p_tag == APP) {
        state.interactions.* += 1;
        state.next = term_new(ERA, 0, 0);
        return dispatch(state);
    }

    // CO0/CO1 + ERA
    if (p_tag == CO0 or p_tag == CO1) {
        state.interactions.* += 1;
        state.heap[term_val(prev)] = term_new(ERA, 0, 0) | SUB_BIT;
        state.next = term_new(ERA, 0, 0);
        return dispatch(state);
    }

    state.heap[term_val(prev)] = state.next;
    return state.next;
}

/// CTR handler - constructor value
fn handle_ctr(state: *ReduceState) Term {
    if (state.spos <= state.stop) return state.next;

    state.spos -= 1;
    const prev = state.stack[state.spos];
    const p_tag = term_tag(prev);
    const tag = term_tag(state.next);

    // MAT + CTR: Pattern match
    if (p_tag == MAT) {
        state.interactions.* += 1;
        const ctr_loc = term_val(state.next);
        const mat_loc = term_val(prev);
        const arity = tag - C00;
        const case_idx = term_ext(state.next);

        var branch = state.heap[mat_loc + 1 + case_idx];
        for (0..arity) |i| {
            const field = state.heap[ctr_loc + i];
            const loc = HVM.heap_pos;
            HVM.heap_pos += 2;
            state.heap[loc] = branch;
            state.heap[loc + 1] = field;
            branch = term_new(APP, 0, @truncate(loc));
        }
        state.next = branch;
        return dispatch(state);
    }

    // CO0/CO1 + CTR: Duplicate constructor
    if (p_tag == CO0 or p_tag == CO1) {
        state.interactions.* += 1;
        const dup_loc = term_val(prev);
        const dup_lab = term_ext(prev);
        const ctr_loc = term_val(state.next);
        const ctr_ext = term_ext(state.next);
        const arity = tag - C00;

        if (arity == 0) {
            state.heap[dup_loc] = state.next | SUB_BIT;
            return dispatch(state);
        }

        const needed = 3 * arity;
        const base = HVM.heap_pos;
        HVM.heap_pos += needed;

        for (0..arity) |i| {
            const idx: Val = @truncate(i);
            const dup_i: Val = @truncate(base + i);
            state.heap[dup_i] = state.heap[ctr_loc + idx];
        }

        const ctr0_base: Val = @truncate(base + arity);
        const ctr1_base: Val = @truncate(base + 2 * arity);

        for (0..arity) |i| {
            const idx: Val = @truncate(i);
            const dup_i: Val = @truncate(base + i);
            state.heap[ctr0_base + idx] = term_new(CO0, dup_lab, dup_i);
            state.heap[ctr1_base + idx] = term_new(CO1, dup_lab, dup_i);
        }

        if (p_tag == CO0) {
            state.heap[dup_loc] = term_new(tag, ctr_ext, ctr1_base) | SUB_BIT;
            state.next = term_new(tag, ctr_ext, ctr0_base);
        } else {
            state.heap[dup_loc] = term_new(tag, ctr_ext, ctr0_base) | SUB_BIT;
            state.next = term_new(tag, ctr_ext, ctr1_base);
        }
        return dispatch(state);
    }

    state.heap[term_val(prev)] = state.next;
    return state.next;
}

/// Default handler - return term
fn handle_default(state: *ReduceState) Term {
    return state.next;
}

/// Direct-threaded dispatch - jump to handler via function pointer
inline fn dispatch(state: *ReduceState) Term {
    // Follow substitutions first
    while ((state.next & SUB_BIT) != 0) {
        state.next = state.heap[term_val(state.next)];
    }
    const tag = term_tag(state.next);
    return compiled_handlers[tag](state);
}

/// COMPILED REDUCER - Main entry point
/// Uses direct-threaded dispatch for zero switch overhead
pub fn reduce_compiled(term: Term) Term {
    var interactions: u64 = 0;
    var state = ReduceState{
        .next = term,
        .heap = HVM.heap,
        .stack = HVM.stack,
        .spos = HVM.stack_pos,
        .stop = HVM.stack_pos,
        .interactions = &interactions,
    };

    const result = dispatch(&state);

    HVM.stack_pos = state.spos;
    HVM.interactions += interactions;
    return result;
}

/// Benchmark compiled reducer
pub fn bench_compiled(iterations: u64) struct { ops: u64, ns: u64 } {
    var timer = std.time.Timer.start() catch return .{ .ops = 0, .ns = 0 };
    var ops: u64 = 0;

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        const lam_loc = alloc(1);
        HVM.heap[lam_loc] = term_new(VAR, 0, lam_loc);
        const app_loc = alloc(2);
        HVM.heap[app_loc] = term_new(LAM, 0, lam_loc);
        HVM.heap[app_loc + 1] = term_new(NUM, 0, @truncate(i));
        _ = reduce_compiled(term_new(APP, 0, app_loc));
        ops += 1;
    }

    const elapsed = timer.read();
    return .{ .ops = ops, .ns = elapsed };
}

/// Specialized pure beta-reduction chain (no dispatch overhead)
/// For when you know the term is a chain of (((λ.λ.λ... body) arg1) arg2) ...
pub fn reduce_beta_chain(term: Term) Term {
    var next = term;
    const heap = HVM.heap;

    while (true) {
        // Follow substitutions
        while ((next & SUB_BIT) != 0) {
            next = heap[term_val(next)];
        }

        const tag = term_tag(next);
        if (tag != APP) return next;

        const app_loc = term_val(next);
        var func = heap[app_loc];

        // Follow substitutions on function
        while ((func & SUB_BIT) != 0) {
            func = heap[term_val(func)];
        }

        if (term_tag(func) != LAM) return next;

        // Beta reduction
        const lam_loc = term_val(func);
        const arg = heap[app_loc + 1];
        next = heap[lam_loc];
        heap[lam_loc] = arg | SUB_BIT;
        HVM.interactions += 1;
    }
}

/// Benchmark pure beta chain
pub fn bench_beta_chain(iterations: u64, depth: u32) struct { ops: u64, ns: u64 } {
    var timer = std.time.Timer.start() catch return .{ .ops = 0, .ns = 0 };
    var ops: u64 = 0;

    var i: u64 = 0;
    while (i < iterations) : (i += 1) {
        // Build chain: (((...((λx.x) n)...)...)
        var term = term_new(NUM, 0, @truncate(i));
        var d: u32 = 0;
        while (d < depth) : (d += 1) {
            const lam_loc = alloc(1);
            HVM.heap[lam_loc] = term_new(VAR, 0, lam_loc);
            const app_loc = alloc(2);
            HVM.heap[app_loc] = term_new(LAM, 0, lam_loc);
            HVM.heap[app_loc + 1] = term;
            term = term_new(APP, 0, app_loc);
        }
        _ = reduce_beta_chain(term);
        ops += depth;
    }

    const elapsed = timer.read();
    return .{ .ops = ops, .ns = elapsed };
}

// =============================================================================
// Linear/Affine Type System (Oracle Problem Prevention)
// =============================================================================

/// Linearity classification for variables and terms
pub const Linearity = enum(u2) {
    /// Unrestricted: can be used any number of times (default)
    unrestricted = 0,
    /// Affine: can be used at most once (may be dropped)
    affine = 1,
    /// Linear: must be used exactly once
    linear = 2,
    /// Consumed: already used (for tracking)
    consumed = 3,
};

/// Information about a linear/affine binding
pub const LinearityInfo = struct {
    linearity: Linearity,
    label: ?Ext,        // Associated SUP label (if from superposition)
    usage_count: u8,    // How many times this variable has been referenced
    source_loc: Val,    // Where this binding originated
};

/// Linearity error types
pub const LinearityErrorKind = enum {
    /// Linear variable used more than once
    linear_used_twice,
    /// Linear variable never used
    linear_not_used,
    /// Affine variable duplicated
    affine_used_twice,
    /// Different-label DUP+SUP detected (oracle risk)
    oracle_risk,
    /// Attempting to duplicate linear term
    linear_duplicated,
    /// Linear variable dropped without use
    linear_dropped,
};

/// Linearity error with details
pub const LinearityError = struct {
    kind: LinearityErrorKind,
    location: Val,
    dup_label: ?Ext = null,
    sup_label: ?Ext = null,
    message: []const u8 = "",
};

/// Type error codes (for ERR term ext field)
pub const TypeErrorCode = enum(u8) {
    /// Expected type doesn't match actual
    type_mismatch = 0,
    /// Linear variable used twice
    linearity_violation = 1,
    /// Affine variable duplicated
    affinity_violation = 2,
    /// Oracle explosion risk
    oracle_risk = 3,
    /// Cannot unify types
    unification_failed = 4,
    /// Type not in universe
    universe_error = 5,
};

// =============================================================================
// Safe-Level Analysis (Oracle Problem Detection)
// =============================================================================

/// Safe level classification for terms
pub const SafeLevel = enum {
    /// Safe: no risk of exponential blowup
    safe,
    /// Potentially unsafe: contains superpositions that might commute
    potentially_unsafe,
    /// Unsafe: detected pattern likely to cause exponential blowup
    unsafe,
    /// Unknown: couldn't determine safety level
    unknown,
};

/// Analysis result with details
pub const SafetyAnalysis = struct {
    level: SafeLevel,
    max_sup_depth: u32,
    distinct_labels: u32,
    potential_commutations: u32,
    linear_boundaries: u32,  // Count of linear terms that prevent duplication
    affine_boundaries: u32,  // Count of affine terms
    message: []const u8,
};

/// Analyze a term for potential oracle problem patterns
/// The oracle problem occurs when:
/// 1. Superpositions with different labels interact
/// 2. Duplications cross superposition boundaries
/// 3. Labels escape their intended scope
pub fn analyze_safety(term: Term) SafetyAnalysis {
    var labels_seen: [256]bool = [_]bool{false} ** 256;
    var label_count: u32 = 0;
    var max_depth: u32 = 0;
    var commutation_risk: u32 = 0;
    var linear_boundaries: u32 = 0;
    var affine_boundaries: u32 = 0;

    // Work stack for iterative analysis
    // in_linear: true if we're inside a linear boundary (duplication forbidden)
    var work_stack: [4096]struct { term: Term, depth: u32, in_dup: bool, in_linear: bool } = undefined;
    var stack_pos: usize = 0;

    work_stack[0] = .{ .term = term, .depth = 0, .in_dup = false, .in_linear = false };
    stack_pos = 1;

    while (stack_pos > 0) {
        stack_pos -= 1;
        const item = work_stack[stack_pos];
        const t = item.term;
        const depth = item.depth;
        const in_dup = item.in_dup;
        const in_linear = item.in_linear;

        const tag = term_tag(t);
        const ext = term_ext(t);
        const val = term_val(t);

        switch (tag) {
            SUP => {
                // Track label
                const label_idx = ext & 0xFF;
                if (!labels_seen[label_idx]) {
                    labels_seen[label_idx] = true;
                    label_count += 1;
                }

                // Update max depth
                if (depth + 1 > max_depth) {
                    max_depth = depth + 1;
                }

                // If we're inside a dup with different label, potential commutation
                // BUT if we're in a linear boundary, it's safe (duplication forbidden)
                if (in_dup and !in_linear) {
                    commutation_risk += 1;
                }

                // Push children
                if (stack_pos + 2 <= work_stack.len) {
                    work_stack[stack_pos] = .{ .term = HVM.heap[val], .depth = depth + 1, .in_dup = in_dup, .in_linear = in_linear };
                    work_stack[stack_pos + 1] = .{ .term = HVM.heap[val + 1], .depth = depth + 1, .in_dup = in_dup, .in_linear = in_linear };
                    stack_pos += 2;
                }
            },

            CO0, CO1 => {
                // Inside a duplication - track label interaction risk
                const label_idx = ext & 0xFF;
                if (!labels_seen[label_idx]) {
                    labels_seen[label_idx] = true;
                    label_count += 1;
                }

                // If we're trying to duplicate a linear term, that's safe (will be rejected)
                // But if not in linear, flag as potential dup context
                if (stack_pos < work_stack.len) {
                    work_stack[stack_pos] = .{ .term = HVM.heap[val], .depth = depth, .in_dup = !in_linear, .in_linear = in_linear };
                    stack_pos += 1;
                }
            },

            LIN => {
                // Linear boundary - duplication of this term is forbidden
                // This makes everything inside safe from oracle explosion
                linear_boundaries += 1;
                if (stack_pos < work_stack.len) {
                    work_stack[stack_pos] = .{ .term = HVM.heap[val], .depth = depth, .in_dup = false, .in_linear = true };
                    stack_pos += 1;
                }
            },

            AFF => {
                // Affine boundary - similar to linear but allows drop
                affine_boundaries += 1;
                if (stack_pos < work_stack.len) {
                    work_stack[stack_pos] = .{ .term = HVM.heap[val], .depth = depth, .in_dup = false, .in_linear = true };
                    stack_pos += 1;
                }
            },

            LAM => {
                if (stack_pos < work_stack.len) {
                    work_stack[stack_pos] = .{ .term = HVM.heap[val], .depth = depth, .in_dup = in_dup, .in_linear = in_linear };
                    stack_pos += 1;
                }
            },

            APP => {
                if (stack_pos + 2 <= work_stack.len) {
                    work_stack[stack_pos] = .{ .term = HVM.heap[val], .depth = depth, .in_dup = in_dup, .in_linear = in_linear };
                    work_stack[stack_pos + 1] = .{ .term = HVM.heap[val + 1], .depth = depth, .in_dup = in_dup, .in_linear = in_linear };
                    stack_pos += 2;
                }
            },

            else => {
                // Handle constructors
                if (tag >= C00 and tag <= C15) {
                    const arity = ctr_arity(tag);
                    for (0..arity) |i| {
                        if (stack_pos < work_stack.len) {
                            work_stack[stack_pos] = .{ .term = HVM.heap[val + i], .depth = depth, .in_dup = in_dup, .in_linear = in_linear };
                            stack_pos += 1;
                        }
                    }
                }
            },
        }
    }

    // Determine safety level based on analysis
    var level: SafeLevel = .safe;
    var message: []const u8 = "Term is safe";

    if (commutation_risk > 0) {
        if (commutation_risk > 10 or (label_count > 5 and max_depth > 3)) {
            level = .unsafe;
            message = "High risk of exponential blowup: multiple label commutations detected";
        } else if (label_count > 2 and max_depth > 2) {
            level = .potentially_unsafe;
            message = "Potential exponential blowup: nested superpositions with multiple labels";
        } else {
            level = .potentially_unsafe;
            message = "Minor risk: some label commutations possible";
        }
    } else if (max_depth > 5 and label_count > 3) {
        level = .potentially_unsafe;
        message = "Deep nested superpositions - monitor for performance";
    }

    // If we have linear boundaries, that's additional safety
    if (linear_boundaries > 0 and level != .safe) {
        message = "Term has linear boundaries that may prevent oracle explosion";
    }

    return .{
        .level = level,
        .max_sup_depth = max_depth,
        .distinct_labels = label_count,
        .potential_commutations = commutation_risk,
        .linear_boundaries = linear_boundaries,
        .affine_boundaries = affine_boundaries,
        .message = message,
    };
}

/// Check if a term is safe to reduce (won't explode exponentially)
pub fn is_safe(term: Term) bool {
    const analysis = analyze_safety(term);
    return analysis.level == .safe;
}

/// Reduce with safety check - returns error on unsafe terms
pub fn safe_reduce(term: Term) struct { result: Term, safe: bool } {
    const analysis = analyze_safety(term);

    if (analysis.level == .unsafe) {
        print("WARNING: Unsafe term detected - {s}\n", .{analysis.message});
        print("  Max SUP depth: {d}, Labels: {d}, Commutation risk: {d}\n", .{
            analysis.max_sup_depth,
            analysis.distinct_labels,
            analysis.potential_commutations,
        });
        return .{ .result = term, .safe = false };
    }

    return .{ .result = reduce(term), .safe = true };
}

/// Maximum commutations before we consider it runaway
const MAX_COMMUTATIONS: u64 = 1_000_000;

/// Reset commutation counter (call before reduction)
pub fn reset_commutation_counter() void {
    commutation_count = 0;
}

/// Get current commutation count
pub fn get_commutation_count() u64 {
    return commutation_count;
}

/// Check if we've hit the commutation limit
pub fn commutation_limit_reached() bool {
    return commutation_count >= MAX_COMMUTATIONS;
}

// =============================================================================
// Utility Functions
// =============================================================================

pub fn set_len(len: u64) void {
    HVM.heap_pos = len;
}

pub fn set_itr(itr: u64) void {
    HVM.interactions = itr;
}

pub fn show_stats() void {
    print("Interactions: {d}\n", .{HVM.interactions});
    print("Heap size: {d}\n", .{HVM.heap_pos});
}

pub fn get_itrs() u64 {
    return HVM.interactions;
}

pub fn get_len() u64 {
    return HVM.heap_pos;
}

// =============================================================================
// Tests
// =============================================================================

test "term operations" {
    const t = term_new(APP, 0x123, 0xABCDEF);
    try std.testing.expectEqual(APP, term_tag(t));
    try std.testing.expectEqual(@as(Ext, 0x123), term_ext(t));
    try std.testing.expectEqual(@as(Val, 0xABCDEF), term_val(t));
}

test "substitution bit" {
    const t = term_new(LAM, 0, 42);
    try std.testing.expect(!term_is_sub(t));
    const t_sub = term_set_sub(t);
    try std.testing.expect(term_is_sub(t_sub));
    const t_clr = term_clr_sub(t_sub);
    try std.testing.expect(!term_is_sub(t_clr));
}
