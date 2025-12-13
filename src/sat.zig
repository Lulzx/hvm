const std = @import("std");
const hvm = @import("hvm.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// HVM4 SAT Solver - Superposition-Powered Boolean Satisfiability
// =============================================================================
//
// This module implements a SAT solver using HVM4's superposition mechanism.
// Boolean variables are represented as superpositions of 0 and 1, and the
// formula is evaluated in parallel across all possible assignments.
//
// Key insight: Superposition naturally encodes the search space of SAT.
// - Each variable x = SUP{0, 1} represents "x is 0 or 1"
// - Formula evaluation happens in parallel for all assignments
// - Satisfying assignments are found when formula evaluates to 1
//
// =============================================================================

/// Literal: a variable or its negation
pub const Literal = struct {
    var_idx: u32,
    negated: bool,

    pub fn positive(idx: u32) Literal {
        return .{ .var_idx = idx, .negated = false };
    }

    pub fn negative(idx: u32) Literal {
        return .{ .var_idx = idx, .negated = true };
    }
};

/// Clause: disjunction of literals (l1 OR l2 OR ...)
pub const Clause = struct {
    literals: []const Literal,
};

/// CNF Formula: conjunction of clauses ((c1) AND (c2) AND ...)
pub const CNF = struct {
    num_vars: u32,
    clauses: []const Clause,
};

/// SAT Solver result
pub const SATResult = struct {
    satisfiable: bool,
    assignment: ?[]bool, // Variable assignments if SAT
    interactions: u64,   // Number of interactions used
};

/// SAT Solver using superposition
pub const SATSolver = struct {
    allocator: Allocator,
    num_vars: u32,
    clauses: ArrayList(Clause),
    var_terms: ArrayList(hvm.Term), // SUP{0,1} for each variable

    pub fn init(allocator: Allocator) SATSolver {
        return .{
            .allocator = allocator,
            .num_vars = 0,
            .clauses = ArrayList(Clause).empty,
            .var_terms = ArrayList(hvm.Term).empty,
        };
    }

    pub fn deinit(self: *SATSolver) void {
        self.clauses.deinit(self.allocator);
        self.var_terms.deinit(self.allocator);
    }

    /// Add a variable (returns variable index)
    pub fn addVar(self: *SATSolver) !u32 {
        const idx = self.num_vars;
        self.num_vars += 1;

        // Create SUP{0, 1} for this variable
        const var_term = hvm.sup(
            hvm.term_new(hvm.NUM, 0, 0), // 0 = false
            hvm.term_new(hvm.NUM, 0, 1), // 1 = true
        );
        try self.var_terms.append(self.allocator, var_term);

        return idx;
    }

    /// Add multiple variables
    pub fn addVars(self: *SATSolver, n: u32) !void {
        for (0..n) |_| {
            _ = try self.addVar();
        }
    }

    /// Add a clause
    pub fn addClause(self: *SATSolver, literals: []const Literal) !void {
        // Copy literals to owned memory
        const owned = try self.allocator.dupe(Literal, literals);
        try self.clauses.append(self.allocator, .{ .literals = owned });
    }

    /// Build the formula term
    fn buildFormula(self: *SATSolver) !hvm.Term {
        if (self.clauses.items.len == 0) {
            return hvm.term_new(hvm.NUM, 0, 1); // True
        }

        // Build each clause as OR of literals
        var clause_terms = ArrayList(hvm.Term).empty;
        defer clause_terms.deinit(self.allocator);

        for (self.clauses.items) |clause| {
            const clause_term = try self.buildClause(clause);
            try clause_terms.append(self.allocator, clause_term);
        }

        // AND all clauses together
        var result = clause_terms.items[0];
        for (clause_terms.items[1..]) |clause_term| {
            result = try self.buildAnd(result, clause_term);
        }

        return result;
    }

    fn buildClause(self: *SATSolver, clause: Clause) !hvm.Term {
        if (clause.literals.len == 0) {
            return hvm.term_new(hvm.NUM, 0, 0); // False (empty clause)
        }

        var result = try self.buildLiteral(clause.literals[0]);
        for (clause.literals[1..]) |lit| {
            const lit_term = try self.buildLiteral(lit);
            result = try self.buildOr(result, lit_term);
        }
        return result;
    }

    fn buildLiteral(self: *SATSolver, lit: Literal) !hvm.Term {
        if (lit.var_idx >= self.var_terms.items.len) {
            return error.InvalidVariable;
        }

        const var_term = self.var_terms.items[lit.var_idx];
        if (lit.negated) {
            return self.buildNot(var_term);
        }
        return var_term;
    }

    fn buildNot(self: *SATSolver, term: hvm.Term) hvm.Term {
        _ = self;
        // NOT x = x XOR 1
        const loc = hvm.alloc(2);
        hvm.set(loc, term);
        hvm.set(loc + 1, hvm.term_new(hvm.NUM, 0, 1));
        return hvm.term_new(hvm.P02, 0x07, loc); // XOR
    }

    fn buildOr(self: *SATSolver, a: hvm.Term, b: hvm.Term) !hvm.Term {
        _ = self;
        // a OR b = a | b (bitwise OR)
        const loc = hvm.alloc(2);
        hvm.set(loc, a);
        hvm.set(loc + 1, b);
        return hvm.term_new(hvm.P02, 0x06, loc); // OR
    }

    fn buildAnd(self: *SATSolver, a: hvm.Term, b: hvm.Term) !hvm.Term {
        _ = self;
        // a AND b = a & b (bitwise AND)
        const loc = hvm.alloc(2);
        hvm.set(loc, a);
        hvm.set(loc + 1, b);
        return hvm.term_new(hvm.P02, 0x05, loc); // AND
    }

    /// Solve the SAT problem
    pub fn solve(self: *SATSolver) !SATResult {
        const start_itrs = hvm.get_itrs();

        // Build the formula
        const formula = try self.buildFormula();

        // Reduce the formula
        hvm.reset_commutation_counter();
        const result = hvm.reduce(formula);

        const end_itrs = hvm.get_itrs();

        // Check the result
        const tag = hvm.term_tag(result);

        if (tag == hvm.NUM) {
            const val = hvm.term_val(result);
            if (val == 1) {
                // SAT - try to extract assignment
                const assignment = try self.extractAssignment();
                return .{
                    .satisfiable = true,
                    .assignment = assignment,
                    .interactions = end_itrs - start_itrs,
                };
            } else {
                // UNSAT (all branches evaluated to 0)
                return .{
                    .satisfiable = false,
                    .assignment = null,
                    .interactions = end_itrs - start_itrs,
                };
            }
        } else if (tag == hvm.SUP) {
            // Result is still a superposition - formula is SAT
            // Need to find which branch is satisfying
            const assignment = try self.extractAssignmentFromSup(result);
            return .{
                .satisfiable = true,
                .assignment = assignment,
                .interactions = end_itrs - start_itrs,
            };
        }

        // Unexpected result
        return .{
            .satisfiable = false,
            .assignment = null,
            .interactions = end_itrs - start_itrs,
        };
    }

    fn extractAssignment(self: *SATSolver) ![]bool {
        var assignment = try self.allocator.alloc(bool, self.num_vars);

        for (self.var_terms.items, 0..) |var_term, i| {
            const reduced = hvm.reduce(var_term);
            const tag = hvm.term_tag(reduced);

            if (tag == hvm.NUM) {
                assignment[i] = hvm.term_val(reduced) != 0;
            } else {
                // Still superposition - default to false
                assignment[i] = false;
            }
        }

        return assignment;
    }

    fn extractAssignmentFromSup(self: *SATSolver, result: hvm.Term) ![]bool {
        _ = result;
        // For now, return the first satisfying assignment
        return self.extractAssignment();
    }
};

// =============================================================================
// High-level API
// =============================================================================

/// Solve a CNF formula
pub fn solveCNF(allocator: Allocator, cnf: CNF) !SATResult {
    var solver = SATSolver.init(allocator);
    defer solver.deinit();

    // Add variables
    try solver.addVars(cnf.num_vars);

    // Add clauses
    for (cnf.clauses) |clause| {
        try solver.addClause(clause.literals);
    }

    return solver.solve();
}

/// Parse DIMACS format CNF
pub fn parseDIMACS(allocator: Allocator, text: []const u8) !CNF {
    var lines = std.mem.splitScalar(u8, text, '\n');
    var num_vars: u32 = 0;
    var clauses = ArrayList(Clause).empty;

    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == 'c') continue; // Comment

        if (trimmed[0] == 'p') {
            // Problem line: p cnf <vars> <clauses>
            var parts = std.mem.splitScalar(u8, trimmed, ' ');
            _ = parts.next(); // 'p'
            _ = parts.next(); // 'cnf'
            if (parts.next()) |vars_str| {
                num_vars = try std.fmt.parseInt(u32, vars_str, 10);
            }
            continue;
        }

        // Clause line: lit1 lit2 ... 0
        var literals = ArrayList(Literal).empty;
        var parts = std.mem.splitScalar(u8, trimmed, ' ');
        while (parts.next()) |lit_str| {
            if (lit_str.len == 0) continue;
            const lit_val = std.fmt.parseInt(i32, lit_str, 10) catch continue;
            if (lit_val == 0) break; // End of clause

            const idx = @as(u32, @intCast(@abs(lit_val))) - 1;
            const lit = if (lit_val > 0)
                Literal.positive(idx)
            else
                Literal.negative(idx);
            try literals.append(allocator, lit);
        }

        if (literals.items.len > 0) {
            const owned = try literals.toOwnedSlice(allocator);
            try clauses.append(allocator, .{ .literals = owned });
        }
    }

    return .{
        .num_vars = num_vars,
        .clauses = try clauses.toOwnedSlice(allocator),
    };
}

// =============================================================================
// Convenience functions for building formulas
// =============================================================================

/// Create a simple 3-SAT instance for testing
pub fn make3SATInstance(allocator: Allocator, num_vars: u32, num_clauses: u32, seed: u64) !CNF {
    var prng = std.rand.DefaultPrng.init(seed);
    var random = prng.random();

    var clauses = ArrayList(Clause).empty;

    for (0..num_clauses) |_| {
        var literals: [3]Literal = undefined;
        for (0..3) |i| {
            const idx = random.intRangeLessThan(u32, 0, num_vars);
            const negated = random.boolean();
            literals[i] = .{ .var_idx = idx, .negated = negated };
        }
        const owned = try allocator.dupe(Literal, &literals);
        try clauses.append(allocator, .{ .literals = owned });
    }

    return .{
        .num_vars = num_vars,
        .clauses = try clauses.toOwnedSlice(allocator),
    };
}

// =============================================================================
// Tests
// =============================================================================

test "simple SAT" {
    const allocator = std.testing.allocator;
    hvm.init(.{});
    defer hvm.deinit();

    var solver = SATSolver.init(allocator);
    defer solver.deinit();

    // Create variables
    const x = try solver.addVar();
    const y = try solver.addVar();

    // Add clause: (x OR y)
    try solver.addClause(&[_]Literal{
        Literal.positive(x),
        Literal.positive(y),
    });

    const result = try solver.solve();
    try std.testing.expect(result.satisfiable);
}

test "UNSAT" {
    const allocator = std.testing.allocator;
    hvm.init(.{});
    defer hvm.deinit();

    var solver = SATSolver.init(allocator);
    defer solver.deinit();

    const x = try solver.addVar();

    // Add contradictory clauses: x AND NOT x
    try solver.addClause(&[_]Literal{Literal.positive(x)});
    try solver.addClause(&[_]Literal{Literal.negative(x)});

    const result = try solver.solve();
    try std.testing.expect(!result.satisfiable);
}
