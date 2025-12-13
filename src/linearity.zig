const std = @import("std");
const hvm = @import("hvm.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// Linearity Checker - Static analysis for Oracle Problem prevention
// =============================================================================
//
// The Oracle Problem in HVM occurs when DUP+SUP interactions with different
// labels cause exponential blowup through commutation. This checker prevents
// such blowup by enforcing linear/affine type discipline:
//
// - Linear variables MUST be used exactly once
// - Affine variables MAY be used at most once
// - Duplication of linear terms is forbidden
// - Different-label DUP+SUP is flagged as oracle risk
//
// =============================================================================

/// Result of linearity analysis
pub const LinearityResult = struct {
    is_safe: bool,
    errors: []hvm.LinearityError,
    warnings: []Warning,
    linear_vars_unused: u32,
    affine_vars_dropped: u32,
};

/// Non-fatal warnings during analysis
pub const Warning = struct {
    kind: WarningKind,
    location: hvm.Val,
    message: []const u8,
};

pub const WarningKind = enum {
    /// Affine variable was dropped (allowed but noted)
    affine_dropped,
    /// Deep superposition nesting detected
    deep_sup_nesting,
    /// Many distinct labels in scope
    many_labels,
};

/// Tracks variable bindings during analysis
const VarInfo = struct {
    loc: hvm.Val,
    linearity: hvm.Linearity,
    label: ?hvm.Ext,
    usage_count: u32,
    defined_at_depth: u32,
};

/// Linearity Checker state
pub const LinearityChecker = struct {
    allocator: Allocator,
    errors: ArrayList(hvm.LinearityError),
    warnings: ArrayList(Warning),
    var_stack: ArrayList(VarInfo),
    label_stack: ArrayList(hvm.Ext),
    current_depth: u32,
    in_dup_context: bool,
    dup_label: ?hvm.Ext,

    pub fn init(allocator: Allocator) LinearityChecker {
        return .{
            .allocator = allocator,
            .errors = ArrayList(hvm.LinearityError).empty,
            .warnings = ArrayList(Warning).empty,
            .var_stack = ArrayList(VarInfo).empty,
            .label_stack = ArrayList(hvm.Ext).empty,
            .current_depth = 0,
            .in_dup_context = false,
            .dup_label = null,
        };
    }

    pub fn deinit(self: *LinearityChecker) void {
        self.errors.deinit(self.allocator);
        self.warnings.deinit(self.allocator);
        self.var_stack.deinit(self.allocator);
        self.label_stack.deinit(self.allocator);
    }

    /// Check a term for linearity violations
    /// Returns true if the term is safe (no linearity errors)
    pub fn check(self: *LinearityChecker, term: hvm.Term) !bool {
        try self.checkTerm(term);
        return self.errors.items.len == 0;
    }

    /// Get the analysis result
    pub fn getResult(self: *LinearityChecker) LinearityResult {
        var linear_unused: u32 = 0;
        var affine_dropped: u32 = 0;

        // Check for unused linear variables
        for (self.var_stack.items) |v| {
            if (v.linearity == .linear and v.usage_count == 0) {
                linear_unused += 1;
            } else if (v.linearity == .affine and v.usage_count == 0) {
                affine_dropped += 1;
            }
        }

        return .{
            .is_safe = self.errors.items.len == 0,
            .errors = self.errors.items,
            .warnings = self.warnings.items,
            .linear_vars_unused = linear_unused,
            .affine_vars_dropped = affine_dropped,
        };
    }

    fn checkTerm(self: *LinearityChecker, term: hvm.Term) !void {
        // Handle substitution
        if ((term & hvm.SUB_BIT) != 0) {
            const loc = hvm.term_val(term);
            const inner = hvm.got(loc);
            try self.checkTerm(inner);
            return;
        }

        const tag = hvm.term_tag(term);
        const ext = hvm.term_ext(term);
        const val = hvm.term_val(term);

        switch (tag) {
            hvm.VAR => {
                // Variable usage - find and increment count
                try self.useVariable(val);
            },

            hvm.LAM => {
                // Lambda - introduces a binding
                self.current_depth += 1;
                try self.checkTerm(hvm.got(val));
                self.current_depth -= 1;
            },

            hvm.APP => {
                // Application - check both parts
                try self.checkTerm(hvm.got(val));
                try self.checkTerm(hvm.got(val + 1));
            },

            hvm.SUP => {
                // Superposition - track label and check branches
                try self.label_stack.append(self.allocator, ext);

                // Warn if nesting is deep
                if (self.label_stack.items.len > 5) {
                    try self.warnings.append(self.allocator, .{
                        .kind = .deep_sup_nesting,
                        .location = val,
                        .message = "Deep superposition nesting may cause performance issues",
                    });
                }

                try self.checkTerm(hvm.got(val));
                try self.checkTerm(hvm.got(val + 1));

                _ = self.label_stack.pop();
            },

            hvm.CO0, hvm.CO1 => {
                // Collapse projection - this is where duplication happens
                const dup_label = ext;
                const old_in_dup = self.in_dup_context;
                const old_dup_label = self.dup_label;

                self.in_dup_context = true;
                self.dup_label = dup_label;

                // Check if we're duplicating something with a different SUP label
                try self.checkDuplicationSafety(val, dup_label);

                try self.checkTerm(hvm.got(val));

                self.in_dup_context = old_in_dup;
                self.dup_label = old_dup_label;
            },

            hvm.LIN => {
                // Linear wrapper - the inner term must be used exactly once
                // Register this as a linear binding point
                try self.var_stack.append(self.allocator, .{
                    .loc = val,
                    .linearity = .linear,
                    .label = if (ext != 0) ext else null,
                    .usage_count = 0,
                    .defined_at_depth = self.current_depth,
                });

                try self.checkTerm(hvm.got(val));

                // Check if linear term was used
                if (self.var_stack.items.len > 0) {
                    const last = self.var_stack.items[self.var_stack.items.len - 1];
                    if (last.loc == val and last.usage_count == 0) {
                        try self.errors.append(self.allocator, .{
                            .kind = .linear_not_used,
                            .location = val,
                            .message = "Linear term was not used",
                        });
                    }
                }
            },

            hvm.AFF => {
                // Affine wrapper - the inner term may be used at most once
                try self.var_stack.append(self.allocator, .{
                    .loc = val,
                    .linearity = .affine,
                    .label = null,
                    .usage_count = 0,
                    .defined_at_depth = self.current_depth,
                });

                try self.checkTerm(hvm.got(val));

                // Affine can be dropped, just note it
                if (self.var_stack.items.len > 0) {
                    const last = self.var_stack.items[self.var_stack.items.len - 1];
                    if (last.loc == val and last.usage_count == 0) {
                        try self.warnings.append(self.allocator, .{
                            .kind = .affine_dropped,
                            .location = val,
                            .message = "Affine term was dropped without use",
                        });
                    }
                }
            },

            hvm.LET => {
                // Let binding
                try self.checkTerm(hvm.got(val));
                try self.checkTerm(hvm.got(val + 1));
            },

            hvm.ANN => {
                // Type annotation
                try self.checkTerm(hvm.got(val));
                try self.checkTerm(hvm.got(val + 1));
            },

            hvm.P02 => {
                // Binary operator
                try self.checkTerm(hvm.got(val));
                try self.checkTerm(hvm.got(val + 1));
            },

            hvm.SWI => {
                // Switch
                try self.checkTerm(hvm.got(val));
                try self.checkTerm(hvm.got(val + 1));
                try self.checkTerm(hvm.got(val + 2));
            },

            hvm.MAT => {
                // Pattern match - check scrutinee and all branches
                try self.checkTerm(hvm.got(val));
                // Branches stored after scrutinee based on ext (number of cases)
                for (1..ext + 1) |i| {
                    try self.checkTerm(hvm.got(val + i));
                }
            },

            hvm.ALL, hvm.SIG => {
                // Dependent types
                try self.checkTerm(hvm.got(val));
                try self.checkTerm(hvm.got(val + 1));
            },

            hvm.SLF, hvm.BRI => {
                // Self type, Bridge
                try self.checkTerm(hvm.got(val));
            },

            // Terminals - no recursion needed
            hvm.ERA, hvm.NUM, hvm.TYP, hvm.REF => {},

            // Constructors
            else => {
                if (tag >= hvm.C00 and tag <= hvm.C15) {
                    const arity = hvm.ctr_arity(tag);
                    for (0..arity) |i| {
                        try self.checkTerm(hvm.got(val + i));
                    }
                }
            },
        }
    }

    fn useVariable(self: *LinearityChecker, loc: hvm.Val) !void {
        // Find the variable in our stack and increment usage
        for (self.var_stack.items) |*v| {
            if (v.loc == loc) {
                v.usage_count += 1;

                // Check for linearity violations
                if (v.linearity == .linear and v.usage_count > 1) {
                    try self.errors.append(self.allocator, .{
                        .kind = .linear_used_twice,
                        .location = loc,
                        .message = "Linear variable used more than once",
                    });
                } else if (v.linearity == .affine and v.usage_count > 1) {
                    try self.errors.append(self.allocator, .{
                        .kind = .affine_used_twice,
                        .location = loc,
                        .message = "Affine variable used more than once",
                    });
                }
                return;
            }
        }
    }

    fn checkDuplicationSafety(self: *LinearityChecker, loc: hvm.Val, dup_label: hvm.Ext) !void {
        // Check if we're duplicating a linear term
        for (self.var_stack.items) |v| {
            if (v.loc == loc and v.linearity == .linear) {
                // Linear term being duplicated
                // Check if it's label-safe
                if (v.label) |safe_label| {
                    if (safe_label != dup_label) {
                        try self.errors.append(self.allocator, .{
                            .kind = .oracle_risk,
                            .location = loc,
                            .dup_label = dup_label,
                            .sup_label = safe_label,
                            .message = "Linear term duplicated with different label - oracle risk",
                        });
                    }
                    // Same label - safe to duplicate (will annihilate)
                } else {
                    // No label - always unsafe to duplicate linear
                    try self.errors.append(self.allocator, .{
                        .kind = .linear_duplicated,
                        .location = loc,
                        .dup_label = dup_label,
                        .message = "Linear term cannot be duplicated",
                    });
                }
            }
        }

        // Check if there's a SUP with different label in scope
        for (self.label_stack.items) |sup_label| {
            if (sup_label != dup_label) {
                // Different label SUP in duplication context - oracle risk
                try self.errors.append(self.allocator, .{
                    .kind = .oracle_risk,
                    .location = loc,
                    .dup_label = dup_label,
                    .sup_label = sup_label,
                    .message = "DUP with different label than enclosing SUP - potential oracle explosion",
                });
            }
        }
    }
};

// =============================================================================
// Public API
// =============================================================================

/// Check a term for linearity violations
/// Returns a result with all errors and warnings
pub fn checkLinearity(allocator: Allocator, term: hvm.Term) !LinearityResult {
    var checker = LinearityChecker.init(allocator);
    defer checker.deinit();

    _ = try checker.check(term);
    return checker.getResult();
}

/// Quick check - returns true if term is linearity-safe
pub fn isLinearitySafe(allocator: Allocator, term: hvm.Term) !bool {
    var checker = LinearityChecker.init(allocator);
    defer checker.deinit();

    return checker.check(term);
}

/// Format linearity errors for display
pub fn formatErrors(allocator: Allocator, errors: []const hvm.LinearityError) ![]const u8 {
    var buf = ArrayList(u8).empty;

    for (errors, 0..) |err, i| {
        if (i > 0) try buf.appendSlice(allocator, "\n");

        const kind_str = switch (err.kind) {
            .linear_used_twice => "Linear variable used twice",
            .linear_not_used => "Linear variable not used",
            .affine_used_twice => "Affine variable used twice",
            .oracle_risk => "Oracle explosion risk",
            .linear_duplicated => "Linear term duplicated",
            .linear_dropped => "Linear variable dropped",
        };

        try std.fmt.format(buf.writer(allocator), "Error at loc {d}: {s}", .{ err.location, kind_str });

        if (err.dup_label) |dl| {
            try std.fmt.format(buf.writer(allocator), " (DUP label: {d}", .{dl});
            if (err.sup_label) |sl| {
                try std.fmt.format(buf.writer(allocator), ", SUP label: {d}", .{sl});
            }
            try buf.append(allocator, ')');
        }
    }

    return buf.toOwnedSlice(allocator);
}

// =============================================================================
// Tests
// =============================================================================

test "linear variable used once is safe" {
    const allocator = std.testing.allocator;

    // Create: \!x.x (linear identity)
    hvm.init(.{});
    defer hvm.deinit();

    const lam_loc = hvm.alloc(1);
    const var_term = hvm.term_new(hvm.VAR, 0, lam_loc);
    hvm.set(lam_loc, var_term);
    const lam = hvm.term_new(hvm.LAM, 0, lam_loc);
    const lin_lam = hvm.linear(lam);

    const result = try checkLinearity(allocator, lin_lam);
    try std.testing.expect(result.is_safe);
    try std.testing.expectEqual(@as(usize, 0), result.errors.len);
}
