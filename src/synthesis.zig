const std = @import("std");
const hvm = @import("hvm.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// HVM4 Program Synthesis - Superposition-Powered Program Generation
// =============================================================================
//
// This module implements program synthesis using HVM4's superposition mechanism.
// The key insight is that superposition naturally encodes the search space of
// possible programs:
//
// - SUP{prog1, prog2} represents "either prog1 or prog2 could be the answer"
// - Grammar rules become superposition constructors
// - Evaluation prunes invalid programs through collapse
// - The first satisfying program is found via sup_find
//
// =============================================================================

/// Specification: what the synthesized program should do
pub const Spec = struct {
    /// Input-output examples
    examples: []const Example,
    /// Maximum program depth
    max_depth: u32,
    /// Type constraints (optional)
    type_hint: ?hvm.Term,

    pub const Example = struct {
        input: hvm.Term,
        expected_output: hvm.Term,
    };
};

/// Grammar for program generation
pub const Grammar = struct {
    /// Terminal symbols (constants, variables)
    terminals: []const Terminal,
    /// Non-terminal production rules
    rules: []const Rule,
    /// Start symbol
    start: u32,

    pub const Terminal = struct {
        name: []const u8,
        term: hvm.Term,
    };

    pub const Rule = struct {
        name: []const u8,
        arity: u32,
        /// Build function: takes child terms, returns combined term
        builder: *const fn ([]const hvm.Term) hvm.Term,
    };
};

/// Synthesis result
pub const SynthesisResult = struct {
    success: bool,
    program: ?hvm.Term,
    candidates_explored: u64,
    interactions: u64,
};

/// Program Synthesizer using superposition
pub const Synthesizer = struct {
    allocator: Allocator,
    spec: Spec,
    grammar: Grammar,
    stats: SynthesisStats,

    pub const SynthesisStats = struct {
        candidates_generated: u64 = 0,
        candidates_pruned: u64 = 0,
        examples_tested: u64 = 0,
    };

    pub fn init(allocator: Allocator, spec: Spec, grammar: Grammar) Synthesizer {
        return .{
            .allocator = allocator,
            .spec = spec,
            .grammar = grammar,
            .stats = .{},
        };
    }

    /// Generate all candidate programs up to max_depth as a superposition
    pub fn enumerate(self: *Synthesizer, depth: u32) hvm.Term {
        return self.enumerateFrom(self.grammar.start, depth);
    }

    fn enumerateFrom(self: *Synthesizer, rule_idx: u32, depth: u32) hvm.Term {
        if (depth == 0) {
            // Base case: enumerate terminals
            return self.enumerateTerminals();
        }

        // Enumerate all applicable rules
        var candidates = ArrayList(hvm.Term).init(self.allocator);
        defer candidates.deinit();

        // Add terminals
        for (self.grammar.terminals) |terminal| {
            candidates.append(terminal.term) catch continue;
            self.stats.candidates_generated += 1;
        }

        // Add rule applications with recursive children
        for (self.grammar.rules, 0..) |rule, idx| {
            _ = rule_idx;
            if (rule.arity == 0) {
                // Nullary rule
                var empty_args: [0]hvm.Term = .{};
                const term = rule.builder(&empty_args);
                candidates.append(term) catch continue;
                self.stats.candidates_generated += 1;
            } else {
                // Generate children recursively
                const children = self.enumerateChildren(rule.arity, @as(u32, @intCast(idx)), depth - 1);
                if (children) |c| {
                    candidates.append(c) catch continue;
                    self.stats.candidates_generated += 1;
                }
            }
        }

        // Combine all candidates into a superposition
        return self.combineToSup(candidates.items);
    }

    fn enumerateTerminals(self: *Synthesizer) hvm.Term {
        if (self.grammar.terminals.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        var result = self.grammar.terminals[0].term;
        for (self.grammar.terminals[1..]) |terminal| {
            result = hvm.sup(result, terminal.term);
        }
        return result;
    }

    fn enumerateChildren(self: *Synthesizer, arity: u32, rule_idx: u32, depth: u32) ?hvm.Term {
        if (arity == 0) return null;

        // For simplicity, enumerate each child position independently
        // A more sophisticated version would enumerate tuples
        var children = self.allocator.alloc(hvm.Term, arity) catch return null;
        defer self.allocator.free(children);

        for (0..arity) |i| {
            children[i] = self.enumerateFrom(rule_idx, depth);
        }

        // Apply the rule builder
        const rule = self.grammar.rules[rule_idx];
        return rule.builder(children);
    }

    fn combineToSup(self: *Synthesizer, terms: []const hvm.Term) hvm.Term {
        _ = self;
        if (terms.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        var result = terms[0];
        for (terms[1..]) |t| {
            result = hvm.sup(result, t);
        }
        return result;
    }

    /// Synthesize a program that satisfies the specification
    pub fn synthesize(self: *Synthesizer) SynthesisResult {
        const start_itrs = hvm.get_itrs();

        // Generate candidate programs as superposition
        const candidates = self.enumerate(self.spec.max_depth);

        // Find a program that satisfies all examples
        const result = self.findSatisfying(candidates);

        const end_itrs = hvm.get_itrs();

        return .{
            .success = result != null,
            .program = result,
            .candidates_explored = self.stats.candidates_generated,
            .interactions = end_itrs - start_itrs,
        };
    }

    fn findSatisfying(self: *Synthesizer, candidates: hvm.Term) ?hvm.Term {
        // Apply the program to each example and check output
        // Using sup_find to find a satisfying assignment

        // For each example, create a test
        const test_term = candidates;
        for (self.spec.examples) |example| {
            // Apply candidate to input
            const applied = makeApp(test_term, example.input);
            // Reduce
            const result = hvm.reduce(applied);
            // Check if it matches expected
            if (!checkOutput(result, example.expected_output)) {
                self.stats.candidates_pruned += 1;
                return null;
            }
            self.stats.examples_tested += 1;
        }

        // If we get here with a concrete term, we found a solution
        const tag = hvm.term_tag(test_term);
        if (tag != hvm.SUP) {
            return test_term;
        }

        // Still a superposition - need to collapse further
        // Use sup_find to find satisfying branch
        return hvm.sup_find(test_term, self.spec.examples[0].expected_output);
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

fn makeApp(func: hvm.Term, arg: hvm.Term) hvm.Term {
    const loc = hvm.alloc(2);
    hvm.set(loc, func);
    hvm.set(loc + 1, arg);
    return hvm.term_new(hvm.APP, 0, loc);
}

fn checkOutput(result: hvm.Term, expected: hvm.Term) bool {
    // Simple structural equality check
    return hvm.struct_equal(result, expected);
}

// =============================================================================
// Built-in Grammars
// =============================================================================

/// Simple arithmetic grammar for synthesizing numeric functions
pub fn arithmeticGrammar() Grammar {
    return .{
        .terminals = &[_]Grammar.Terminal{
            .{ .name = "0", .term = hvm.term_new(hvm.NUM, 0, 0) },
            .{ .name = "1", .term = hvm.term_new(hvm.NUM, 0, 1) },
            .{ .name = "x", .term = hvm.term_new(hvm.VAR, 0, 0) }, // Input variable
        },
        .rules = &[_]Grammar.Rule{
            .{ .name = "add", .arity = 2, .builder = &buildAdd },
            .{ .name = "mul", .arity = 2, .builder = &buildMul },
            .{ .name = "sub", .arity = 2, .builder = &buildSub },
        },
        .start = 0,
    };
}

fn buildAdd(args: []const hvm.Term) hvm.Term {
    if (args.len < 2) return hvm.term_new(hvm.ERA, 0, 0);
    const loc = hvm.alloc(2);
    hvm.set(loc, args[0]);
    hvm.set(loc + 1, args[1]);
    return hvm.term_new(hvm.P02, hvm.OP_ADD, loc);
}

fn buildMul(args: []const hvm.Term) hvm.Term {
    if (args.len < 2) return hvm.term_new(hvm.ERA, 0, 0);
    const loc = hvm.alloc(2);
    hvm.set(loc, args[0]);
    hvm.set(loc + 1, args[1]);
    return hvm.term_new(hvm.P02, hvm.OP_MUL, loc);
}

fn buildSub(args: []const hvm.Term) hvm.Term {
    if (args.len < 2) return hvm.term_new(hvm.ERA, 0, 0);
    const loc = hvm.alloc(2);
    hvm.set(loc, args[0]);
    hvm.set(loc + 1, args[1]);
    return hvm.term_new(hvm.P02, hvm.OP_SUB, loc);
}

/// Simple lambda grammar for synthesizing small functions
pub fn lambdaGrammar() Grammar {
    return .{
        .terminals = &[_]Grammar.Terminal{
            .{ .name = "x", .term = hvm.term_new(hvm.VAR, 0, 0) },
        },
        .rules = &[_]Grammar.Rule{
            .{ .name = "lam", .arity = 1, .builder = &buildLam },
            .{ .name = "app", .arity = 2, .builder = &buildApp },
        },
        .start = 0,
    };
}

fn buildLam(args: []const hvm.Term) hvm.Term {
    if (args.len < 1) return hvm.term_new(hvm.ERA, 0, 0);
    const loc = hvm.alloc(1);
    hvm.set(loc, args[0]);
    return hvm.term_new(hvm.LAM, 0, loc);
}

fn buildApp(args: []const hvm.Term) hvm.Term {
    if (args.len < 2) return hvm.term_new(hvm.ERA, 0, 0);
    return makeApp(args[0], args[1]);
}

// =============================================================================
// High-Level API
// =============================================================================

/// Synthesize a function from input-output examples
pub fn synthesizeFromExamples(
    allocator: Allocator,
    examples: []const Spec.Example,
    max_depth: u32,
) SynthesisResult {
    const spec = Spec{
        .examples = examples,
        .max_depth = max_depth,
        .type_hint = null,
    };

    var synthesizer = Synthesizer.init(allocator, spec, arithmeticGrammar());
    return synthesizer.synthesize();
}

/// Synthesize a program matching a type signature
pub fn synthesizeFromType(
    allocator: Allocator,
    type_sig: hvm.Term,
    max_depth: u32,
) SynthesisResult {
    const spec = Spec{
        .examples = &[_]Spec.Example{},
        .max_depth = max_depth,
        .type_hint = type_sig,
    };

    var synthesizer = Synthesizer.init(allocator, spec, lambdaGrammar());
    return synthesizer.synthesize();
}

// =============================================================================
// Enumeration Utilities
// =============================================================================

/// Enumerate all programs of a given size as a superposition
pub fn enumeratePrograms(grammar: Grammar, depth: u32) hvm.Term {
    // Use HVM's built-in enum_term for basic enumeration
    // Number of variables depends on grammar terminals
    const num_vars: u32 = @intCast(grammar.terminals.len);
    return hvm.enum_term(depth, num_vars);
}

/// Create a superposition of all values in a range
pub fn enumerateRange(min: u32, max: u32) hvm.Term {
    if (min >= max) return hvm.term_new(hvm.NUM, 0, min);

    var result = hvm.term_new(hvm.NUM, 0, max - 1);
    var i: u32 = max - 1;
    while (i > min) {
        i -= 1;
        result = hvm.sup(hvm.term_new(hvm.NUM, 0, i), result);
    }
    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "enumerate range" {
    hvm.init(.{});
    defer hvm.deinit();

    // Enumerate 0..3
    const range = enumerateRange(0, 3);

    // Should be SUP{0, SUP{1, 2}}
    try std.testing.expectEqual(hvm.SUP, hvm.term_tag(range));
}

test "arithmetic grammar construction" {
    hvm.init(.{});
    defer hvm.deinit();

    const grammar = arithmeticGrammar();
    try std.testing.expectEqual(@as(usize, 3), grammar.terminals.len);
    try std.testing.expectEqual(@as(usize, 3), grammar.rules.len);
}

test "build add" {
    hvm.init(.{});
    defer hvm.deinit();

    const two = hvm.term_new(hvm.NUM, 0, 2);
    const three = hvm.term_new(hvm.NUM, 0, 3);
    const args = [_]hvm.Term{ two, three };

    const add = buildAdd(&args);
    const result = hvm.reduce(add);

    try std.testing.expectEqual(hvm.NUM, hvm.term_tag(result));
    try std.testing.expectEqual(@as(hvm.Val, 5), hvm.term_val(result));
}
