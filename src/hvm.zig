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
const SUB_BIT: u64 = 0x8000000000000000;

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

// Dynamic sup/dup
pub const DSU: Tag = 0x50; // Dynamic superposition
pub const DDU: Tag = 0x51; // Dynamic duplication

// Type System (Interaction Calculus of Constructions)
pub const ANN: Tag = 0x60; // Annotation {term : Type}
pub const BRI: Tag = 0x61; // Bridge θx. term (for dependent types)
pub const TYP: Tag = 0x62; // Type universe (*)
pub const ALL: Tag = 0x63; // Dependent function type Π(x:A).B
pub const SIG: Tag = 0x64; // Dependent pair type Σ(x:A).B
pub const SLF: Tag = 0x65; // Self type (for induction)

// Stack frames for type checking
pub const F_ANN: Tag = 0x68; // Reducing annotation
pub const F_BRI: Tag = 0x69; // Reducing bridge
pub const F_EQL2: Tag = 0x6A; // Second stage of equality check

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

pub const MAX_HEAP_SIZE: u64 = 1 << 30; // 1GB
pub const MAX_STACK_SIZE: u64 = 1 << 24; // 16MB
pub const MAX_BOOK_SIZE: usize = 65536;

pub const BookFn = *const fn (Term) Term;

/// HVM4 runtime state
pub const State = struct {
    // Hot fields
    heap: []Term,
    stack: []Term,
    heap_pos: u64,
    stack_pos: u64,
    interactions: u64,
    fresh_label: Ext,

    // Book (function definitions)
    book: [MAX_BOOK_SIZE]?BookFn,
    ctr_arity: [MAX_BOOK_SIZE]u16,
    fun_arity: [MAX_BOOK_SIZE]u16,

    // Compiled function bodies (moved from parser global state)
    func_bodies: [MAX_BOOK_SIZE]Term,

    pub fn init(allocator: std.mem.Allocator) !*State {
        const state = try allocator.create(State);
        state.heap = try allocator.alloc(Term, MAX_HEAP_SIZE);
        state.stack = try allocator.alloc(Term, MAX_STACK_SIZE);
        state.heap_pos = 1; // Reserve location 0
        state.stack_pos = 0;
        state.interactions = 0;
        state.fresh_label = 0x800000; // Auto-dup labels start here
        state.book = [_]?BookFn{null} ** MAX_BOOK_SIZE;
        state.ctr_arity = [_]u16{0} ** MAX_BOOK_SIZE;
        state.fun_arity = [_]u16{0} ** MAX_BOOK_SIZE;
        state.func_bodies = [_]Term{0} ** MAX_BOOK_SIZE;

        // Initialize built-in functions
        state.book[SUP_F] = builtin_sup;
        state.book[DUP_F] = builtin_dup;
        state.book[LOG_F] = builtin_log;

        @memset(state.heap, 0);
        @memset(state.stack, 0);

        return state;
    }

    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        allocator.free(self.stack);
        allocator.free(self.heap);
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
};

/// Generic function dispatch - retrieves body from HVM state
pub fn funcDispatch(ref: Term) Term {
    const fid = term_ext(ref);
    return HVM.func_bodies[fid];
}

// Thread-local HVM state
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
    HVM.heap[lam_loc] = term_set_sub(arg);
    return bod;
}

/// APP + ERA: Erasure
inline fn interact_app_era() Term {
    HVM.interactions += 1;
    return term_new(ERA, 0, 0);
}

/// APP + SUP: Superposition distribution
fn interact_app_sup(app_loc: Val, sup_loc: Val, sup_lab: Ext) Term {
    HVM.interactions += 1;
    const arg = HVM.heap[app_loc + 1];
    const lft = HVM.heap[sup_loc];
    const rgt = HVM.heap[sup_loc + 1];

    // Create dup for argument
    const dup_loc = alloc(1);
    HVM.heap[dup_loc] = arg;

    // Create two applications
    const app0_loc = alloc(2);
    HVM.heap[app0_loc] = lft;
    HVM.heap[app0_loc + 1] = term_new(CO0, sup_lab, dup_loc);

    const app1_loc = alloc(2);
    HVM.heap[app1_loc] = rgt;
    HVM.heap[app1_loc + 1] = term_new(CO1, sup_lab, dup_loc);

    // Create result superposition
    const res_loc = alloc(2);
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
fn interact_dup_lam(dup_tag: Tag, dup_loc: Val, dup_lab: Ext, lam_loc: Val) Term {
    HVM.interactions += 1;
    const bod = HVM.heap[lam_loc];

    // Create inner dup for body
    const inner_dup = alloc(1);
    HVM.heap[inner_dup] = bod;

    // Create two lambdas
    const lam0_loc = alloc(1);
    HVM.heap[lam0_loc] = term_new(CO0, dup_lab, inner_dup);

    const lam1_loc = alloc(1);
    HVM.heap[lam1_loc] = term_new(CO1, dup_lab, inner_dup);

    // Create superposition of lambdas
    const sup_loc = alloc(2);
    HVM.heap[sup_loc] = term_new(LAM, 0, lam0_loc);
    HVM.heap[sup_loc + 1] = term_new(LAM, 0, lam1_loc);

    // Return appropriate projection
    if (dup_tag == CO0) {
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
        // Annihilation: same labels cancel out
        if (dup_tag == CO0) {
            HVM.heap[dup_loc] = term_set_sub(rgt);
            return lft;
        } else {
            HVM.heap[dup_loc] = term_set_sub(lft);
            return rgt;
        }
    } else {
        // Commutation: different labels create new nodes
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

/// DUP + NUM: Both projections get the number
inline fn interact_dup_num(dup_loc: Val, num: Term) Term {
    HVM.interactions += 1;
    HVM.heap[dup_loc] = term_set_sub(num);
    return num;
}

/// DUP + CTR: Duplicate constructor
fn interact_dup_ctr(dup_tag: Tag, dup_loc: Val, dup_lab: Ext, ctr_tag: Tag, ctr_loc: Val, ctr_ext: Ext) Term {
    HVM.interactions += 1;
    const arity = ctr_arity(ctr_tag);

    if (arity == 0) {
        // Arity 0: just duplicate the term
        const ctr = term_new(ctr_tag, ctr_ext, ctr_loc);
        HVM.heap[dup_loc] = term_set_sub(ctr);
        return ctr;
    }

    // Create dups for each field
    const dups_loc = alloc(arity);
    for (0..arity) |i| {
        HVM.heap[dups_loc + i] = HVM.heap[ctr_loc + i];
    }

    // Create two constructors with CO0/CO1 refs
    const ctr0_loc = alloc(arity);
    const ctr1_loc = alloc(arity);
    for (0..arity) |i| {
        HVM.heap[ctr0_loc + i] = term_new(CO0, dup_lab, @truncate(dups_loc + i));
        HVM.heap[ctr1_loc + i] = term_new(CO1, dup_lab, @truncate(dups_loc + i));
    }

    if (dup_tag == CO0) {
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

    // Apply constructor fields to branch
    for (0..arity) |i| {
        const field = HVM.heap[ctr_loc + i];
        const app_loc = alloc(2);
        HVM.heap[app_loc] = branch;
        HVM.heap[app_loc + 1] = field;
        branch = term_new(APP, 0, app_loc);
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

fn builtin_log(ref: Term) Term {
    _ = ref;
    print("LOG: not implemented\n", .{});
    return term_new(ERA, 0, 0);
}

// =============================================================================
// Structural Equality
// =============================================================================

/// Deep structural equality check between two terms
pub fn struct_equal(a: Term, b: Term) bool {
    const a_reduced = reduce(a);
    const b_reduced = reduce(b);

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
        ERA => return true,
        NUM => return a_val == b_val,

        // Lambda: compare bodies (alpha equivalence)
        LAM => {
            const a_bod = HVM.heap[a_val];
            const b_bod = HVM.heap[b_val];
            return struct_equal(a_bod, b_bod);
        },

        // Application: compare both parts
        APP => {
            const a_fun = HVM.heap[a_val];
            const a_arg = HVM.heap[a_val + 1];
            const b_fun = HVM.heap[b_val];
            const b_arg = HVM.heap[b_val + 1];
            return struct_equal(a_fun, b_fun) and struct_equal(a_arg, b_arg);
        },

        // Superposition: compare label and both branches
        SUP => {
            if (a_ext != b_ext) return false;
            const a_lft = HVM.heap[a_val];
            const a_rgt = HVM.heap[a_val + 1];
            const b_lft = HVM.heap[b_val];
            const b_rgt = HVM.heap[b_val + 1];
            return struct_equal(a_lft, b_lft) and struct_equal(a_rgt, b_rgt);
        },

        // Constructors: compare ext (constructor id) and all fields
        C00, C01, C02, C03, C04, C05, C06, C07, C08, C09, C10, C11, C12, C13, C14, C15 => {
            if (a_ext != b_ext) return false;
            const arity = ctr_arity(a_tag);
            for (0..arity) |i| {
                if (!struct_equal(HVM.heap[a_val + i], HVM.heap[b_val + i])) {
                    return false;
                }
            }
            return true;
        },

        // Type universe
        TYP => return true,

        // Default: not equal if we can't determine
        else => return false,
    }
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
        // APP+SUP, CO*+LAM, CO*+CTR, MAT+CTR, SWI+NUM all allocate
        if (p_tag == APP and tag == SUP) return true;
        if ((p_tag == CO0 or p_tag == CO1) and tag == LAM) return true;
        if ((p_tag == CO0 or p_tag == CO1) and tag >= C00 and tag <= C15) return true;
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

    // CO1 interactions
    table[CO1][SUP] = wrap_dup_sup;
    table[CO1][NUM] = wrap_dup_num;
    table[CO1][ERA] = wrap_dup_era;
    table[CO1][LAM] = wrap_dup_lam;

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
pub noinline fn reduce(term: Term) Term {
    const heap = HVM.heap;
    const stack = HVM.stack;
    const stop = HVM.stack_pos;
    var spos = stop;
    var next = term;

    while (true) {
        // Handle substitution
        if ((next & SUB_BIT) != 0) {
            @branchHint(.likely);
            const loc = term_val(next);
            next = heap[loc];
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
                if (spos > stop) {
                    spos -= 1;
                    const prev = stack[spos];
                    heap[term_val(prev)] = next;
                    next = prev;
                } else {
                    break;
                }
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

                // F_ANN + value: Annotation decay - extract the term
                if (p_tag == F_ANN) {
                    HVM.interactions += 1;

                    // SUP requires special handling: distribute annotation
                    if (tag == SUP) {
                        // {SUP(a,b) : T} => SUP({a:T}, {b:T})
                        const sup_loc = val;
                        const sup_ext = ext;
                        const typ = heap[p_val + 1];

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
// Normal Form (Full Reduction)
// =============================================================================

pub fn normal(term: Term) Term {
    const result = reduce(term);
    const tag = term_tag(result);

    switch (tag) {
        LAM => {
            const loc = term_val(result);
            const bod = normal(got(loc));
            set(loc, bod);
        },
        APP => {
            const loc = term_val(result);
            const fun = normal(got(loc));
            const arg = normal(got(loc + 1));
            set(loc, fun);
            set(loc + 1, arg);
        },
        SUP => {
            const loc = term_val(result);
            const lft = normal(got(loc));
            const rgt = normal(got(loc + 1));
            set(loc, lft);
            set(loc + 1, rgt);
        },
        else => {
            if (tag >= C00 and tag <= C15) {
                const loc = term_val(result);
                const arity = ctr_arity(tag);
                for (0..arity) |i| {
                    const field = normal(got(loc + i));
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

/// Get fresh label for auto-dup
pub fn fresh_label() Ext {
    const lab = HVM.fresh_label;
    HVM.fresh_label += 1;
    return lab;
}

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

// =============================================================================
// Parallel Execution (Multi-threaded SIMD)
// =============================================================================

pub const NUM_WORKERS = 12;

var worker_states: [NUM_WORKERS]?*State = [_]?*State{null} ** NUM_WORKERS;
var worker_allocator: ?std.mem.Allocator = null;

pub fn parallel_init(allocator: std.mem.Allocator) !void {
    worker_allocator = allocator;
    for (0..NUM_WORKERS) |i| {
        worker_states[i] = try State.init(allocator);
    }
}

pub fn parallel_free() void {
    if (worker_allocator) |allocator| {
        for (0..NUM_WORKERS) |i| {
            if (worker_states[i]) |state| {
                state.deinit(allocator);
                worker_states[i] = null;
            }
        }
    }
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
    const chunk_size = (a.len + NUM_WORKERS - 1) / NUM_WORKERS;
    var threads: [NUM_WORKERS]?std.Thread = [_]?std.Thread{null} ** NUM_WORKERS;

    for (0..NUM_WORKERS) |i| {
        const start = i * chunk_size;
        if (start >= a.len) break;
        const end = @min(start + chunk_size, a.len);
        threads[i] = std.Thread.spawn(.{}, struct {
            fn work(a_s: []const u32, b_s: []const u32, r_s: []u32) void {
                batch_add(a_s, b_s, r_s);
            }
        }.work, .{ a[start..end], b[start..end], results[start..end] }) catch null;
    }

    for (threads) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
}

/// Parallel SIMD batch multiply
pub fn parallel_batch_mul(a: []const u32, b: []const u32, results: []u32) void {
    const chunk_size = (a.len + NUM_WORKERS - 1) / NUM_WORKERS;
    var threads: [NUM_WORKERS]?std.Thread = [_]?std.Thread{null} ** NUM_WORKERS;

    for (0..NUM_WORKERS) |i| {
        const start = i * chunk_size;
        if (start >= a.len) break;
        const end = @min(start + chunk_size, a.len);
        threads[i] = std.Thread.spawn(.{}, struct {
            fn work(a_s: []const u32, b_s: []const u32, r_s: []u32) void {
                batch_mul(a_s, b_s, r_s);
            }
        }.work, .{ a[start..end], b[start..end], results[start..end] }) catch null;
    }

    for (threads) |maybe_thread| {
        if (maybe_thread) |thread| thread.join();
    }
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
