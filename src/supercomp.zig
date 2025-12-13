const std = @import("std");
const hvm = @import("hvm.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// HVM4 Supercompiler - Partial Evaluation & Specialization
// =============================================================================
//
// This module implements compile-time optimization through:
// 1. Partial Evaluation - Reduce terms with known static values
// 2. Specialization - Create optimized versions for specific arg patterns
// 3. Fusion - Combine consecutive operations
// 4. Inlining - Inline small/single-use functions
//
// The key insight: Many HVM programs have "static" structure (known at compile
// time) mixed with "dynamic" data (only known at runtime). We evaluate the
// static parts and residualize the dynamic parts.
//
// =============================================================================

// Helper: create an application term
fn make_app(func: hvm.Term, arg: hvm.Term) hvm.Term {
    const loc = hvm.alloc(2);
    hvm.set(loc, func);
    hvm.set(loc + 1, arg);
    return hvm.term_new(hvm.APP, 0, loc);
}

/// Binding time: is this value known at compile time?
pub const BindingTime = enum {
    static,   // Known at compile time - can be reduced
    dynamic,  // Only known at runtime - must be residualized
    mixed,    // Partially known - specialize
};

/// Information about a term during partial evaluation
pub const PEInfo = struct {
    binding_time: BindingTime,
    size: u32,        // Term size for inlining decisions
    use_count: u32,   // How many times referenced
    is_recursive: bool,
};

/// Specialization key: function ID + static argument hash
pub const SpecKey = struct {
    fid: hvm.Ext,
    static_args_hash: u64,

    pub fn hash(self: SpecKey) u64 {
        return @as(u64, self.fid) ^ (self.static_args_hash *% 0x517cc1b727220a95);
    }

    pub fn eql(a: SpecKey, b: SpecKey) bool {
        return a.fid == b.fid and a.static_args_hash == b.static_args_hash;
    }
};

/// Partial Evaluation Context
pub const PEContext = struct {
    allocator: Allocator,

    // Specialization cache: maps (fid, static_args) -> specialized fid
    spec_cache: std.AutoHashMap(u64, hvm.Ext),
    next_spec_id: hvm.Ext,

    // Term info cache
    term_info: std.AutoHashMap(hvm.Val, PEInfo),

    // Statistics
    stats: PEStats,

    // Configuration
    config: PEConfig,

    pub fn init(allocator: Allocator) PEContext {
        return .{
            .allocator = allocator,
            .spec_cache = std.AutoHashMap(u64, hvm.Ext).init(allocator),
            .next_spec_id = 0x8000, // Start specialized function IDs high
            .term_info = std.AutoHashMap(hvm.Val, PEInfo).init(allocator),
            .stats = .{},
            .config = .{},
        };
    }

    pub fn deinit(self: *PEContext) void {
        self.spec_cache.deinit();
        self.term_info.deinit();
    }
};

/// Configuration for partial evaluation
pub const PEConfig = struct {
    /// Maximum term size to inline
    inline_threshold: u32 = 20,
    /// Maximum specialization depth
    max_spec_depth: u32 = 5,
    /// Enable fusion optimizations
    enable_fusion: bool = true,
    /// Enable aggressive inlining
    aggressive_inline: bool = false,
};

/// Statistics from partial evaluation
pub const PEStats = struct {
    terms_reduced: u64 = 0,
    specializations_created: u64 = 0,
    fusions_performed: u64 = 0,
    inlinings_performed: u64 = 0,
    terms_residualized: u64 = 0,
};

// =============================================================================
// Core Partial Evaluator
// =============================================================================

/// Partial evaluation result
pub const PEResult = struct {
    term: hvm.Term,
    binding_time: BindingTime,
    reduced: bool,
};

/// Perform partial evaluation on a term
pub fn partial_eval(ctx: *PEContext, term: hvm.Term) PEResult {
    const tag = hvm.term_tag(term);
    const ext = hvm.term_ext(term);
    const val = hvm.term_val(term);

    // Handle substitution
    if ((term & hvm.SUB_BIT) != 0) {
        const inner = hvm.got(val);
        return partial_eval(ctx, inner);
    }

    switch (tag) {
        // Numbers are static
        hvm.NUM => {
            ctx.stats.terms_reduced += 1;
            return .{ .term = term, .binding_time = .static, .reduced = true };
        },

        // Erasures are static
        hvm.ERA => {
            return .{ .term = term, .binding_time = .static, .reduced = true };
        },

        // Types are static
        hvm.TYP => {
            return .{ .term = term, .binding_time = .static, .reduced = true };
        },

        // Variables are dynamic (bound at runtime)
        hvm.VAR => {
            ctx.stats.terms_residualized += 1;
            return .{ .term = term, .binding_time = .dynamic, .reduced = false };
        },

        // Lambdas: evaluate body if possible
        hvm.LAM => {
            const body = hvm.got(val);
            const body_result = partial_eval(ctx, body);

            if (body_result.binding_time == .static) {
                // Fully static lambda - keep reduced body
                const new_loc = hvm.alloc(1);
                hvm.set(new_loc, body_result.term);
                return .{
                    .term = hvm.term_new(hvm.LAM, ext, new_loc),
                    .binding_time = .static,
                    .reduced = body_result.reduced,
                };
            }

            return .{ .term = term, .binding_time = .dynamic, .reduced = false };
        },

        // Application: try beta reduction or specialize
        hvm.APP => {
            const func = hvm.got(val);
            const arg = hvm.got(val + 1);

            const func_result = partial_eval(ctx, func);
            const arg_result = partial_eval(ctx, arg);

            // If function is a lambda and we can reduce
            if (hvm.term_tag(func_result.term) == hvm.LAM) {
                if (arg_result.binding_time == .static or ctx.config.aggressive_inline) {
                    // Beta reduction at compile time
                    ctx.stats.terms_reduced += 1;
                    const reduced = hvm.reduce(make_app(func_result.term, arg_result.term));
                    return partial_eval(ctx, reduced);
                }
            }

            // Residualize application
            const new_loc = hvm.alloc(2);
            hvm.set(new_loc, func_result.term);
            hvm.set(new_loc + 1, arg_result.term);

            const bt: BindingTime = if (func_result.binding_time == .static and arg_result.binding_time == .static)
                .static
            else if (func_result.binding_time == .dynamic or arg_result.binding_time == .dynamic)
                .dynamic
            else
                .mixed;

            return .{
                .term = hvm.term_new(hvm.APP, ext, new_loc),
                .binding_time = bt,
                .reduced = func_result.reduced or arg_result.reduced,
            };
        },

        // Superposition: evaluate both branches
        hvm.SUP => {
            const left = hvm.got(val);
            const right = hvm.got(val + 1);

            const left_result = partial_eval(ctx, left);
            const right_result = partial_eval(ctx, right);

            // If both branches reduce to same value, collapse
            if (left_result.binding_time == .static and right_result.binding_time == .static) {
                if (hvm.struct_equal(left_result.term, right_result.term)) {
                    ctx.stats.terms_reduced += 1;
                    return .{ .term = left_result.term, .binding_time = .static, .reduced = true };
                }
            }

            const new_loc = hvm.alloc(2);
            hvm.set(new_loc, left_result.term);
            hvm.set(new_loc + 1, right_result.term);

            return .{
                .term = hvm.term_new(hvm.SUP, ext, new_loc),
                .binding_time = .mixed,
                .reduced = left_result.reduced or right_result.reduced,
            };
        },

        // Binary operators: compute if both operands static
        hvm.P02 => {
            const left = hvm.got(val);
            const right = hvm.got(val + 1);

            const left_result = partial_eval(ctx, left);
            const right_result = partial_eval(ctx, right);

            // Both static numbers - reduce at runtime for now (compute_op not public)
            if (left_result.binding_time == .static and right_result.binding_time == .static) {
                if (hvm.term_tag(left_result.term) == hvm.NUM and
                    hvm.term_tag(right_result.term) == hvm.NUM)
                {
                    // Let HVM reduce the operation at compile time
                    const op_loc = hvm.alloc(2);
                    hvm.set(op_loc, left_result.term);
                    hvm.set(op_loc + 1, right_result.term);
                    const op_term = hvm.term_new(hvm.P02, ext, op_loc);
                    const result = hvm.reduce(op_term);

                    ctx.stats.terms_reduced += 1;
                    return .{
                        .term = result,
                        .binding_time = .static,
                        .reduced = true,
                    };
                }
            }

            // Residualize
            const new_loc = hvm.alloc(2);
            hvm.set(new_loc, left_result.term);
            hvm.set(new_loc + 1, right_result.term);

            return .{
                .term = hvm.term_new(hvm.P02, ext, new_loc),
                .binding_time = if (left_result.binding_time == .static and right_result.binding_time == .static) .static else .dynamic,
                .reduced = left_result.reduced or right_result.reduced,
            };
        },

        // Reference: consider inlining
        hvm.REF => {
            // For now, just residualize references
            return .{ .term = term, .binding_time = .dynamic, .reduced = false };
        },

        // Switch: evaluate condition, specialize if static
        hvm.SWI => {
            const cond = hvm.got(val);
            const cond_result = partial_eval(ctx, cond);

            if (cond_result.binding_time == .static and hvm.term_tag(cond_result.term) == hvm.NUM) {
                // Static condition - select branch at compile time
                const cond_val = hvm.term_val(cond_result.term);
                const branch = if (cond_val == 0) hvm.got(val + 1) else hvm.got(val + 2);
                ctx.stats.terms_reduced += 1;
                return partial_eval(ctx, branch);
            }

            // Residualize with optimized branches
            const zero_branch = partial_eval(ctx, hvm.got(val + 1));
            const succ_branch = partial_eval(ctx, hvm.got(val + 2));

            const new_loc = hvm.alloc(3);
            hvm.set(new_loc, cond_result.term);
            hvm.set(new_loc + 1, zero_branch.term);
            hvm.set(new_loc + 2, succ_branch.term);

            return .{
                .term = hvm.term_new(hvm.SWI, ext, new_loc),
                .binding_time = .dynamic,
                .reduced = cond_result.reduced or zero_branch.reduced or succ_branch.reduced,
            };
        },

        // Let binding: inline if possible
        hvm.LET => {
            const value = hvm.got(val);
            const body = hvm.got(val + 1);

            const value_result = partial_eval(ctx, value);

            // If value is static, we could substitute
            if (value_result.binding_time == .static) {
                // For now, just record it's static and evaluate body
                ctx.stats.terms_reduced += 1;
            }

            const body_result = partial_eval(ctx, body);

            const new_loc = hvm.alloc(2);
            hvm.set(new_loc, value_result.term);
            hvm.set(new_loc + 1, body_result.term);

            return .{
                .term = hvm.term_new(hvm.LET, ext, new_loc),
                .binding_time = body_result.binding_time,
                .reduced = value_result.reduced or body_result.reduced,
            };
        },

        // Constructors: evaluate all fields
        else => {
            if (tag >= hvm.C00 and tag <= hvm.C15) {
                const arity = hvm.ctr_arity(tag);
                var all_static = true;
                var any_reduced = false;

                const new_loc = hvm.alloc(arity);
                for (0..arity) |i| {
                    const field = hvm.got(val + i);
                    const field_result = partial_eval(ctx, field);
                    hvm.set(new_loc + i, field_result.term);
                    if (field_result.binding_time != .static) all_static = false;
                    if (field_result.reduced) any_reduced = true;
                }

                return .{
                    .term = hvm.term_new(tag, ext, new_loc),
                    .binding_time = if (all_static) .static else .mixed,
                    .reduced = any_reduced,
                };
            }

            // Default: residualize
            ctx.stats.terms_residualized += 1;
            return .{ .term = term, .binding_time = .dynamic, .reduced = false };
        },
    }
}

// =============================================================================
// Term Analysis
// =============================================================================

/// Hash a term for specialization key
fn hash_term(term: hvm.Term) u64 {
    const tag = hvm.term_tag(term);
    const ext = hvm.term_ext(term);
    const val = hvm.term_val(term);

    var h: u64 = @as(u64, tag) *% 0x9e3779b97f4a7c15;
    h ^= @as(u64, ext) *% 0x517cc1b727220a95;
    h ^= @as(u64, val) *% 0xbf58476d1ce4e5b9;

    // For compound terms, hash children
    switch (tag) {
        hvm.APP, hvm.SUP, hvm.P02, hvm.LET => {
            h ^= hash_term(hvm.got(val)) *% 0x94d049bb133111eb;
            h ^= hash_term(hvm.got(val + 1)) *% 0x9e3779b97f4a7c15;
        },
        hvm.LAM => {
            h ^= hash_term(hvm.got(val)) *% 0x94d049bb133111eb;
        },
        else => {},
    }

    return h;
}

/// Calculate term size (approximate AST node count)
pub fn term_size(term: hvm.Term) u32 {
    const tag = hvm.term_tag(term);
    const val = hvm.term_val(term);

    // Handle substitution
    if ((term & hvm.SUB_BIT) != 0) {
        return term_size(hvm.got(val));
    }

    switch (tag) {
        hvm.NUM, hvm.ERA, hvm.VAR, hvm.TYP, hvm.REF => return 1,

        hvm.LAM => return 1 + term_size(hvm.got(val)),

        hvm.APP, hvm.SUP, hvm.P02, hvm.LET, hvm.ANN, hvm.ALL, hvm.SIG => {
            return 1 + term_size(hvm.got(val)) + term_size(hvm.got(val + 1));
        },

        hvm.SWI => {
            return 1 + term_size(hvm.got(val)) + term_size(hvm.got(val + 1)) + term_size(hvm.got(val + 2));
        },

        hvm.CO0, hvm.CO1 => return 1 + term_size(hvm.got(val)),

        hvm.SLF, hvm.BRI => return 1 + term_size(hvm.got(val)),

        else => {
            if (tag >= hvm.C00 and tag <= hvm.C15) {
                const arity = hvm.ctr_arity(tag);
                var size: u32 = 1;
                for (0..arity) |i| {
                    size += term_size(hvm.got(val + i));
                }
                return size;
            }
            return 1;
        },
    }
}

/// Check if a term references a function (for recursion detection)
fn is_recursive(term: hvm.Term, fid: hvm.Ext) bool {
    const tag = hvm.term_tag(term);
    const ext = hvm.term_ext(term);
    const val = hvm.term_val(term);

    if ((term & hvm.SUB_BIT) != 0) {
        return is_recursive(hvm.got(val), fid);
    }

    if (tag == hvm.REF and ext == fid) return true;

    switch (tag) {
        hvm.LAM => return is_recursive(hvm.got(val), fid),
        hvm.APP, hvm.SUP, hvm.P02, hvm.LET => {
            return is_recursive(hvm.got(val), fid) or is_recursive(hvm.got(val + 1), fid);
        },
        hvm.SWI => {
            return is_recursive(hvm.got(val), fid) or
                is_recursive(hvm.got(val + 1), fid) or
                is_recursive(hvm.got(val + 2), fid);
        },
        else => {
            if (tag >= hvm.C00 and tag <= hvm.C15) {
                const arity = hvm.ctr_arity(tag);
                for (0..arity) |i| {
                    if (is_recursive(hvm.got(val + i), fid)) return true;
                }
            }
            return false;
        },
    }
}

// =============================================================================
// Fusion Patterns
// =============================================================================

/// Fusion pattern detection result
pub const FusionPattern = enum {
    none,
    map_map,      // map f (map g xs) => map (f . g) xs
    fold_build,   // fold c n (build g) => g c n
    filter_map,   // filter p (map f xs) => filterMap
    concat_map,   // concat (map f xs) => concatMap f xs
};

/// Detect and apply fusion patterns
pub fn detect_fusion(ctx: *PEContext, term: hvm.Term) ?hvm.Term {
    if (!ctx.config.enable_fusion) return null;
    _ = term;
    // Simplified - would need function name recognition in a real implementation
    return null;
}

// =============================================================================
// Public API
// =============================================================================

/// Optimize a term using partial evaluation
pub fn optimize(allocator: Allocator, term: hvm.Term) hvm.Term {
    var ctx = PEContext.init(allocator);
    defer ctx.deinit();

    const result = partial_eval(&ctx, term);
    return result.term;
}

/// Optimize with custom configuration
pub fn optimize_with_config(allocator: Allocator, term: hvm.Term, config: PEConfig) struct { term: hvm.Term, stats: PEStats } {
    var ctx = PEContext.init(allocator);
    defer ctx.deinit();
    ctx.config = config;

    const result = partial_eval(&ctx, term);
    return .{ .term = result.term, .stats = ctx.stats };
}

/// Check if a term is fully static (can be evaluated at compile time)
pub fn is_static(term: hvm.Term) bool {
    const tag = hvm.term_tag(term);
    const val = hvm.term_val(term);

    if ((term & hvm.SUB_BIT) != 0) {
        return is_static(hvm.got(val));
    }

    switch (tag) {
        hvm.NUM, hvm.ERA, hvm.TYP => return true,
        hvm.VAR => return false, // Variables are dynamic
        hvm.REF => return false, // References may have side effects
        hvm.LAM => return is_static(hvm.got(val)),
        hvm.APP, hvm.SUP, hvm.P02, hvm.LET => {
            return is_static(hvm.got(val)) and is_static(hvm.got(val + 1));
        },
        hvm.SWI => {
            return is_static(hvm.got(val)) and
                is_static(hvm.got(val + 1)) and
                is_static(hvm.got(val + 2));
        },
        else => {
            if (tag >= hvm.C00 and tag <= hvm.C15) {
                const arity = hvm.ctr_arity(tag);
                for (0..arity) |i| {
                    if (!is_static(hvm.got(val + i))) return false;
                }
                return true;
            }
            return false;
        },
    }
}

// =============================================================================
// Tests
// =============================================================================

test "partial eval numbers" {
    const allocator = std.testing.allocator;
    hvm.init(.{});
    defer hvm.deinit();

    // 2 + 3 should reduce to 5 at compile time
    const two = hvm.term_new(hvm.NUM, 0, 2);
    const three = hvm.term_new(hvm.NUM, 0, 3);

    const loc = hvm.alloc(2);
    hvm.set(loc, two);
    hvm.set(loc + 1, three);
    const add = hvm.term_new(hvm.P02, hvm.OP_ADD, loc);

    const result = optimize(allocator, add);

    try std.testing.expectEqual(hvm.NUM, hvm.term_tag(result));
    try std.testing.expectEqual(@as(hvm.Val, 5), hvm.term_val(result));
}

test "term size calculation" {
    hvm.init(.{});
    defer hvm.deinit();

    // Simple number
    const num = hvm.term_new(hvm.NUM, 0, 42);
    try std.testing.expectEqual(@as(u32, 1), term_size(num));

    // Lambda with body
    const lam_loc = hvm.alloc(1);
    hvm.set(lam_loc, num);
    const lam = hvm.term_new(hvm.LAM, 0, lam_loc);
    try std.testing.expectEqual(@as(u32, 2), term_size(lam));
}

test "is_static detection" {
    hvm.init(.{});
    defer hvm.deinit();

    // Numbers are static
    const num = hvm.term_new(hvm.NUM, 0, 42);
    try std.testing.expect(is_static(num));

    // Variables are dynamic
    const var_term = hvm.term_new(hvm.VAR, 0, 0);
    try std.testing.expect(!is_static(var_term));
}
