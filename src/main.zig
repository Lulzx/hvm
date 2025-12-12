const std = @import("std");
const hvm = @import("hvm.zig");
const parser = @import("parser.zig");
const print = std.debug.print;

// =============================================================================
// Example Programs
// =============================================================================

// Example: Identity function (lambda x. x)
fn identity() void {
    const lam_loc = hvm.alloc_node(1);
    hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, lam_loc));

    const app_loc = hvm.alloc_node(2);
    hvm.set(app_loc + 0, hvm.term_new(hvm.LAM, 0, lam_loc));
    hvm.set(app_loc + 1, hvm.term_new(hvm.W32, 0, 42));

    const app = hvm.term_new(hvm.APP, 0, app_loc);

    print("Example: Identity function\n", .{});
    print("  Input: (lambda x. x) 42\n", .{});

    const result = hvm.reduce(app);
    print("  Result: ", .{});
    hvm.show_term(result);
    print("\n", .{});
}

// Example: Constant function (lambda x. lambda y. x)
fn constant_fn() void {
    const inner_lam_loc = hvm.alloc_node(1);
    const outer_lam_loc = hvm.alloc_node(1);
    hvm.set(outer_lam_loc, hvm.term_new(hvm.LAM, 0, inner_lam_loc));
    hvm.set(inner_lam_loc, hvm.term_new(hvm.VAR, 0, outer_lam_loc));

    const app1_loc = hvm.alloc_node(2);
    hvm.set(app1_loc + 0, hvm.term_new(hvm.LAM, 0, outer_lam_loc));
    hvm.set(app1_loc + 1, hvm.term_new(hvm.W32, 0, 1));

    const app2_loc = hvm.alloc_node(2);
    hvm.set(app2_loc + 0, hvm.term_new(hvm.APP, 0, app1_loc));
    hvm.set(app2_loc + 1, hvm.term_new(hvm.W32, 0, 2));

    const app = hvm.term_new(hvm.APP, 0, app2_loc);

    print("\nExample: Constant function\n", .{});
    print("  Input: ((lambda x. lambda y. x) 1) 2\n", .{});
    print("  Expected: 1\n", .{});

    const result = hvm.reduce(app);
    print("  Result: ", .{});
    hvm.show_term(result);
    print("\n", .{});
}

// Example: Arithmetic (3 + 4)
fn arithmetic() void {
    const op_loc = hvm.alloc_node(2);
    hvm.set(op_loc + 0, hvm.term_new(hvm.W32, 0, 3));
    hvm.set(op_loc + 1, hvm.term_new(hvm.W32, 0, 4));

    const opx = hvm.term_new(hvm.OPX, hvm.OP_ADD, op_loc);

    print("\nExample: Arithmetic\n", .{});
    print("  Input: 3 + 4\n", .{});

    const result = hvm.reduce(opx);
    print("  Result: ", .{});
    hvm.show_term(result);
    print("\n", .{});
}

// Example: Superposition and duplication
fn superposition() void {
    const sup_loc = hvm.alloc_node(2);
    hvm.set(sup_loc + 0, hvm.term_new(hvm.W32, 0, 1));
    hvm.set(sup_loc + 1, hvm.term_new(hvm.W32, 0, 2));

    const dup_loc = hvm.alloc_node(1);
    hvm.set(dup_loc + 0, hvm.term_new(hvm.SUP, 0, sup_loc));

    const dp0 = hvm.term_new(hvm.DP0, 0, dup_loc);

    print("\nExample: Superposition\n", .{});
    print("  Input: !0{{a, b}} = {{1, 2}}; a\n", .{});
    print("  Expected: 1\n", .{});

    const result = hvm.reduce(dp0);
    print("  Result: ", .{});
    hvm.show_term(result);
    print("\n", .{});
}

// Example: Erasure
fn erasure() void {
    const lam_loc = hvm.alloc_node(1);
    hvm.set(lam_loc, hvm.term_new(hvm.ERA, 0, 0));

    const app_loc = hvm.alloc_node(2);
    hvm.set(app_loc + 0, hvm.term_new(hvm.LAM, 0, lam_loc));
    hvm.set(app_loc + 1, hvm.term_new(hvm.W32, 0, 42));

    const app = hvm.term_new(hvm.APP, 0, app_loc);

    print("\nExample: Erasure\n", .{});
    print("  Input: (lambda x. *) 42\n", .{});
    print("  Expected: *\n", .{});

    const result = hvm.reduce(app);
    print("  Result: ", .{});
    hvm.show_term(result);
    print("\n", .{});
}

// Example: All operators
fn all_operators() void {
    print("\nExample: All arithmetic operators\n", .{});

    const ops = [_]struct { op: hvm.Lab, name: []const u8, a: u32, b: u32, expected: u32 }{
        .{ .op = hvm.OP_ADD, .name = "+", .a = 10, .b = 3, .expected = 13 },
        .{ .op = hvm.OP_SUB, .name = "-", .a = 10, .b = 3, .expected = 7 },
        .{ .op = hvm.OP_MUL, .name = "*", .a = 10, .b = 3, .expected = 30 },
        .{ .op = hvm.OP_DIV, .name = "/", .a = 10, .b = 3, .expected = 3 },
        .{ .op = hvm.OP_MOD, .name = "%", .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_EQ, .name = "==", .a = 10, .b = 10, .expected = 1 },
        .{ .op = hvm.OP_NE, .name = "!=", .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_LT, .name = "<", .a = 3, .b = 10, .expected = 1 },
        .{ .op = hvm.OP_GT, .name = ">", .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_AND, .name = "&", .a = 0xFF, .b = 0x0F, .expected = 0x0F },
        .{ .op = hvm.OP_OR, .name = "|", .a = 0xF0, .b = 0x0F, .expected = 0xFF },
        .{ .op = hvm.OP_XOR, .name = "^", .a = 0xFF, .b = 0x0F, .expected = 0xF0 },
        .{ .op = hvm.OP_LSH, .name = "<<", .a = 1, .b = 4, .expected = 16 },
        .{ .op = hvm.OP_RSH, .name = ">>", .a = 16, .b = 2, .expected = 4 },
    };

    for (ops) |op_info| {
        const op_loc = hvm.alloc_node(2);
        hvm.set(op_loc + 0, hvm.term_new(hvm.W32, 0, op_info.a));
        hvm.set(op_loc + 1, hvm.term_new(hvm.W32, 0, op_info.b));

        const opx = hvm.term_new(hvm.OPX, op_info.op, op_loc);
        const result = hvm.reduce(opx);
        const result_val = hvm.term_loc(result);

        const status = if (result_val == op_info.expected) "OK" else "FAIL";
        print("  {d} {s} {d} = {d} [{s}]\n", .{ op_info.a, op_info.name, op_info.b, result_val, status });
    }
}

// =============================================================================
// Parser Tests
// =============================================================================

fn test_parser(allocator: std.mem.Allocator) !void {
    print("\n=== Parser Tests ===\n\n", .{});

    const tests = [_]struct { name: []const u8, code: []const u8, expected: ?u64 }{
        .{ .name = "Number", .code = "#42", .expected = 42 },
        .{ .name = "Erasure", .code = "*", .expected = null },
        .{ .name = "Character", .code = "'x'", .expected = null },
        .{ .name = "Addition", .code = "(+ #3 #4)", .expected = 7 },
        .{ .name = "Subtraction", .code = "(- #10 #3)", .expected = 7 },
        .{ .name = "Multiplication", .code = "(* #6 #7)", .expected = 42 },
        .{ .name = "Identity", .code = "((\\x.x) #42)", .expected = 42 },
        .{ .name = "Constant", .code = "(((\\x.\\y.x) #1) #2)", .expected = 1 },
        .{ .name = "Superposition", .code = "&0{#1,#2}", .expected = null },
        .{ .name = "Switch zero", .code = "(?#0 #100 \\p.#200)", .expected = 100 },
        .{ .name = "Switch succ", .code = "(?#5 #100 \\p.#200)", .expected = 200 },
    };

    var passed: u32 = 0;
    for (tests) |t| {
        // Reset heap for each test
        hvm.set_len(1);
        hvm.set_itr(0);

        const term = parser.parse(allocator, t.code) catch |err| {
            print("  {s}: PARSE ERROR ({any})\n", .{ t.name, err });
            continue;
        };

        const result = hvm.reduce(term);
        const result_val = hvm.term_loc(result);

        if (t.expected) |exp| {
            if (result_val == exp) {
                print("  {s}: {s} => #{d} [OK]\n", .{ t.name, t.code, result_val });
                passed += 1;
            } else {
                print("  {s}: {s} => #{d} (expected #{d}) [FAIL]\n", .{ t.name, t.code, result_val, exp });
            }
        } else {
            print("  {s}: {s} => ", .{ t.name, t.code });
            hvm.show_term(result);
            print(" [OK]\n", .{});
            passed += 1;
        }
    }

    print("\nPassed: {d}/{d}\n", .{ passed, tests.len });
}

// =============================================================================
// File Runner
// =============================================================================

fn run_file(allocator: std.mem.Allocator, path: []const u8) !void {
    // Read file
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        print("Error: cannot open file '{s}': {any}\n", .{ path, err });
        return;
    };
    defer file.close();

    const content = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch |err| {
        print("Error: cannot read file: {any}\n", .{err});
        return;
    };
    defer allocator.free(content);

    print("Running: {s}\n", .{path});
    print("---\n", .{});

    const start = std.time.microTimestamp();

    // Parse and run
    const term = parser.parse(allocator, content) catch |err| {
        print("Parse error: {any}\n", .{err});
        return;
    };

    const result = hvm.reduce(term);

    const end = std.time.microTimestamp();

    // Print result
    print("Result: ", .{});
    const pretty_result = parser.pretty(allocator, result) catch {
        hvm.show_term(result);
        print("\n", .{});
        return;
    };
    defer allocator.free(pretty_result);
    print("{s}\n", .{pretty_result});

    print("---\n", .{});
    print("Time: {d} us\n", .{end - start});
    print("Interactions: {d}\n", .{hvm.get_itr()});
}

// =============================================================================
// Eval (single expression)
// =============================================================================

fn run_eval(allocator: std.mem.Allocator, code: []const u8) !void {
    const term = parser.parse(allocator, code) catch |err| {
        print("Parse error: {any}\n", .{err});
        return;
    };

    const result = hvm.reduce(term);

    const pretty_result = parser.pretty(allocator, result) catch {
        hvm.show_term(result);
        print("\n", .{});
        return;
    };
    defer allocator.free(pretty_result);
    print("{s}\n", .{pretty_result});
}

// =============================================================================
// Main
// =============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize HVM
    try hvm.hvm_init(allocator);
    defer hvm.hvm_free(allocator);

    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len > 1) {
        const cmd = args[1];

        // Check for .hvm file
        if (std.mem.endsWith(u8, cmd, ".hvm")) {
            try run_file(allocator, cmd);
            return;
        }

        if (std.mem.eql(u8, cmd, "run") and args.len > 2) {
            try run_file(allocator, args[2]);
        } else if (std.mem.eql(u8, cmd, "eval") and args.len > 2) {
            try run_eval(allocator, args[2]);
        } else if (std.mem.eql(u8, cmd, "test")) {
            print_banner();
            run_tests();
        } else if (std.mem.eql(u8, cmd, "parse")) {
            print_banner();
            try test_parser(allocator);
        } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
            print_help();
        } else if (std.mem.eql(u8, cmd, "bench")) {
            print_banner();
            run_benchmark();
        } else if (std.mem.eql(u8, cmd, "examples")) {
            print_banner();
            run_examples();
        } else {
            print("Unknown command: {s}\n\n", .{cmd});
            print_help();
        }
    } else {
        // Default: show help
        print_help();
    }
}

fn print_banner() void {
    print("===========================================\n", .{});
    print("HVM3 - Higher-Order Virtual Machine (Zig)\n", .{});
    print("===========================================\n", .{});
}

fn run_examples() void {
    print("\nRunning examples...\n", .{});

    identity();
    constant_fn();
    arithmetic();
    superposition();
    erasure();

    print("\n", .{});
    hvm.show_stats();
}

fn run_tests() void {
    print("\nRunning tests...\n", .{});

    identity();
    constant_fn();
    arithmetic();
    superposition();
    erasure();
    all_operators();

    print("\nAll tests completed!\n", .{});
    hvm.show_stats();
}

fn run_benchmark() void {
    print("\nRunning benchmark...\n", .{});

    // Benchmark 1: Arithmetic operations
    print("\n1. Arithmetic benchmark (100K ops):\n", .{});
    const arith_iterations: u32 = 100000;
    var timer = std.time.Timer.start() catch {
        print("Timer not available\n", .{});
        return;
    };

    var i: u32 = 0;
    while (i < arith_iterations) : (i += 1) {
        const op_loc = hvm.alloc_node(2);
        hvm.set(op_loc + 0, hvm.term_new(hvm.W32, 0, i));
        hvm.set(op_loc + 1, hvm.term_new(hvm.W32, 0, i + 1));
        const opx = hvm.term_new(hvm.OPX, hvm.OP_ADD, op_loc);
        _ = hvm.reduce(opx);
    }

    var elapsed = timer.read();
    var elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(arith_iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 2: Beta reduction (lambda applications)
    print("\n2. Beta reduction benchmark (100K apps):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < arith_iterations) : (i += 1) {
        // Create (λx.x) i - identity application
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, lam_loc));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.LAM, 0, lam_loc));
        hvm.set(app_loc + 1, hvm.term_new(hvm.W32, 0, i));
        _ = hvm.reduce(hvm.term_new(hvm.APP, 0, app_loc));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(arith_iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 3: DUP+SUP annihilation
    print("\n3. DUP+SUP annihilation benchmark (100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < arith_iterations) : (i += 1) {
        // Create !0{a,b} = &0{1,2}; a
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.W32, 0, 1));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.W32, 0, 2));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 0, sup_loc));
        _ = hvm.reduce(hvm.term_new(hvm.DP0, 0, dup_loc));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(arith_iterations)) / (elapsed_ms / 1000.0)});

    print("\n", .{});
    hvm.show_stats();
}

fn print_help() void {
    const help =
        \\HVM3 - Higher-Order Virtual Machine (Zig)
        \\
        \\Usage: hvm3 [command] [options]
        \\
        \\Commands:
        \\  <file.hvm>      Run an HVM file directly
        \\  run <file>      Run an HVM file
        \\  eval "<expr>"   Evaluate a single expression
        \\  parse           Run parser tests
        \\  test            Run runtime tests
        \\  bench           Run performance benchmark
        \\  examples        Run example programs
        \\  help            Show this help message
        \\
        \\HVM Syntax:
        \\  #42           Number
        \\  'c'           Character
        \\  *             Erasure
        \\  \x.body       Lambda
        \\  (f x)         Application
        \\  (+ a b)       Operators: + - * / % == != < > <= >= & | ^ << >>
        \\  &L{a,b}       Superposition with label L
        \\  !&L{x,y}=v;k  Duplication
        \\  (?n z s)      Switch on number
        \\
        \\Examples:
        \\  hvm3 example.hvm
        \\  hvm3 eval "(+ #3 #4)"
        \\  hvm3 bench
        \\
        \\For more information: https://github.com/HigherOrderCO/HVM3
        \\
    ;
    print("{s}", .{help});
}

// =============================================================================
// Tests
// =============================================================================

test "term operations" {
    const term = hvm.term_new(hvm.LAM, 42, 12345);
    try std.testing.expectEqual(hvm.LAM, hvm.term_tag(term));
    try std.testing.expectEqual(@as(hvm.Lab, 42), hvm.term_lab(term));
    try std.testing.expectEqual(@as(u64, 12345), hvm.term_loc(term));
}

test "term is atom" {
    try std.testing.expect(hvm.term_is_atom(hvm.term_new(hvm.ERA, 0, 0)));
    try std.testing.expect(hvm.term_is_atom(hvm.term_new(hvm.W32, 0, 42)));
    try std.testing.expect(!hvm.term_is_atom(hvm.term_new(hvm.LAM, 0, 0)));
}
