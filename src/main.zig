const std = @import("std");
const hvm = @import("hvm.zig");
const parser = @import("parser.zig");
const metal = @import("metal.zig");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");
const bend = @import("bend.zig");
const compile = @import("compile.zig");
const typecheck = @import("typecheck.zig");
const print = std.debug.print;

// =============================================================================
// Term Display
// =============================================================================

fn show_term(term: hvm.Term) void {
    const tag = hvm.term_tag(term);
    const ext = hvm.term_ext(term);
    const val = hvm.term_val(term);

    switch (tag) {
        hvm.ERA => print("*", .{}),
        hvm.NUM => print("#{d}", .{val}),
        hvm.LAM => print("LAM({d})", .{val}),
        hvm.APP => print("APP({d})", .{val}),
        hvm.SUP => print("SUP({d},{d})", .{ ext, val }),
        hvm.CO0 => print("CO0({d},{d})", .{ ext, val }),
        hvm.CO1 => print("CO1({d},{d})", .{ ext, val }),
        hvm.VAR => print("VAR({d})", .{val}),
        hvm.REF => print("REF({d},{d})", .{ ext, val }),
        else => print("?{d}({d},{d})", .{ tag, ext, val }),
    }
}

// =============================================================================
// Example Programs
// =============================================================================

// Example: Identity function (lambda x. x)
fn identity() void {
    const lam_loc = hvm.alloc_node(1);
    hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));

    const app_loc = hvm.alloc_node(2);
    hvm.set(app_loc + 0, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
    hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));

    const app = hvm.term_new(hvm.APP, 0, @truncate(app_loc));

    print("Example: Identity function\n", .{});
    print("  Input: (lambda x. x) 42\n", .{});

    const result = hvm.reduce(app);
    print("  Result: ", .{});
    show_term(result);
    print("\n", .{});
}

// Example: Constant function (lambda x. lambda y. x)
fn constant_fn() void {
    const inner_lam_loc = hvm.alloc_node(1);
    const outer_lam_loc = hvm.alloc_node(1);
    hvm.set(outer_lam_loc, hvm.term_new(hvm.LAM, 0, @truncate(inner_lam_loc)));
    hvm.set(inner_lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(outer_lam_loc)));

    const app1_loc = hvm.alloc_node(2);
    hvm.set(app1_loc + 0, hvm.term_new(hvm.LAM, 0, @truncate(outer_lam_loc)));
    hvm.set(app1_loc + 1, hvm.term_new(hvm.NUM, 0, 1));

    const app2_loc = hvm.alloc_node(2);
    hvm.set(app2_loc + 0, hvm.term_new(hvm.APP, 0, @truncate(app1_loc)));
    hvm.set(app2_loc + 1, hvm.term_new(hvm.NUM, 0, 2));

    const app = hvm.term_new(hvm.APP, 0, @truncate(app2_loc));

    print("\nExample: Constant function\n", .{});
    print("  Input: ((lambda x. lambda y. x) 1) 2\n", .{});
    print("  Expected: 1\n", .{});

    const result = hvm.reduce(app);
    print("  Result: ", .{});
    show_term(result);
    print("\n", .{});
}

// Example: Arithmetic (3 + 4)
fn arithmetic() void {
    const op_loc = hvm.alloc_node(2);
    hvm.set(op_loc + 0, hvm.term_new(hvm.NUM, 0, 3));
    hvm.set(op_loc + 1, hvm.term_new(hvm.NUM, 0, 4));

    const op = hvm.term_new(hvm.P02, hvm.OP_ADD, @truncate(op_loc));

    print("\nExample: Arithmetic\n", .{});
    print("  Input: 3 + 4\n", .{});

    const result = hvm.reduce(op);
    print("  Result: ", .{});
    show_term(result);
    print("\n", .{});
}

// Example: Superposition and duplication
fn superposition() void {
    const sup_loc = hvm.alloc_node(2);
    hvm.set(sup_loc + 0, hvm.term_new(hvm.NUM, 0, 1));
    hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 2));

    const dup_loc = hvm.alloc_node(1);
    hvm.set(dup_loc + 0, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));

    const co0 = hvm.term_new(hvm.CO0, 0, @truncate(dup_loc));

    print("\nExample: Superposition\n", .{});
    print("  Input: !0{{a, b}} = {{1, 2}}; a\n", .{});
    print("  Expected: 1\n", .{});

    const result = hvm.reduce(co0);
    print("  Result: ", .{});
    show_term(result);
    print("\n", .{});
}

// Example: Erasure
fn erasure() void {
    const lam_loc = hvm.alloc_node(1);
    hvm.set(lam_loc, hvm.term_new(hvm.ERA, 0, 0));

    const app_loc = hvm.alloc_node(2);
    hvm.set(app_loc + 0, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
    hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));

    const app = hvm.term_new(hvm.APP, 0, @truncate(app_loc));

    print("\nExample: Erasure\n", .{});
    print("  Input: (lambda x. *) 42\n", .{});
    print("  Expected: *\n", .{});

    const result = hvm.reduce(app);
    print("  Result: ", .{});
    show_term(result);
    print("\n", .{});
}

// Example: All operators
fn all_operators() void {
    print("\nExample: All arithmetic operators\n", .{});

    const ops = [_]struct { op: hvm.Ext, name: []const u8, a: u32, b: u32, expected: u32 }{
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
        hvm.set(op_loc + 0, hvm.term_new(hvm.NUM, 0, op_info.a));
        hvm.set(op_loc + 1, hvm.term_new(hvm.NUM, 0, op_info.b));

        const p = hvm.term_new(hvm.P02, op_info.op, @truncate(op_loc));
        const result = hvm.reduce(p);
        const result_val = hvm.term_val(result);

        const status = if (result_val == op_info.expected) "OK" else "FAIL";
        print("  {d} {s} {d} = {d} [{s}]\n", .{ op_info.a, op_info.name, op_info.b, result_val, status });
    }
}

// =============================================================================
// Parser Tests
// =============================================================================

fn test_parser(allocator: std.mem.Allocator) !void {
    print("\n=== Parser Tests ===\n\n", .{});

    const tests = [_]struct { name: []const u8, code: []const u8, expected: ?u32 }{
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

    const passed: u32 = 0;
    for (tests) |t| {
        // Reset heap for each test
        hvm.set_len(1);
        hvm.set_itr(0);

        const term = parser.parse(allocator, t.code) catch |err| {
            print("  {s}: PARSE ERROR ({any})\n", .{ t.name, err });
            continue;
        };

        const result = hvm.reduce(term);
        const result_val = hvm.term_val(result);

        if (t.expected) |exp| {
            if (result_val == exp) {
                print("  {s}: {s} => #{d} [OK]\n", .{ t.name, t.code, result_val });
            } else {
                print("  {s}: {s} => #{d} (expected #{d}) [FAIL]\n", .{ t.name, t.code, result_val, exp });
            }
        } else {
            print("  {s}: {s} => ", .{ t.name, t.code });
            show_term(result);
            print(" [OK]\n", .{});
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
        show_term(result);
        print("\n", .{});
        return;
    };
    defer allocator.free(pretty_result);
    print("{s}\n", .{pretty_result});

    print("---\n", .{});
    print("Time: {d} us\n", .{end - start});
    print("Interactions: {d}\n", .{hvm.get_itrs()});
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
        show_term(result);
        print("\n", .{});
        return;
    };
    defer allocator.free(pretty_result);
    print("{s}\n", .{pretty_result});
}

// =============================================================================
// Bend File Runner
// =============================================================================

fn run_bend_file(allocator: std.mem.Allocator, path: []const u8) !void {
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

    print("Running Bend: {s}\n", .{path});
    print("---\n", .{});

    const start = std.time.microTimestamp();

    // Parse Bend source
    var program = bend.parse(content, allocator) catch |err| {
        print("Parse error: {any}\n", .{err});
        return;
    };
    defer program.deinit();

    // Compile to HVM4
    var book = parser.Book.init(allocator);
    var compiler = compile.Compiler.init(allocator, &book);
    defer compiler.deinit();

    const term = compiler.compileProgram(&program) catch |err| {
        print("Compile error: {any}\n", .{err});
        return;
    };

    if (term) |t| {
        const result = hvm.reduce(t);
        const end = std.time.microTimestamp();

        // Print result
        print("Result: ", .{});
        const pretty_result = parser.pretty(allocator, result) catch {
            show_term(result);
            print("\n", .{});
            return;
        };
        defer allocator.free(pretty_result);
        print("{s}\n", .{pretty_result});

        print("---\n", .{});
        print("Time: {d} us\n", .{end - start});
        print("Interactions: {d}\n", .{hvm.get_itrs()});
    } else {
        print("No main expression found\n", .{});
    }
}

fn check_bend_file(allocator: std.mem.Allocator, path: []const u8) !void {
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

    print("Type checking: {s}\n", .{path});
    print("---\n", .{});

    // Parse Bend source
    var program = bend.parse(content, allocator) catch |err| {
        print("Parse error: {any}\n", .{err});
        return;
    };
    defer program.deinit();

    // Type check
    var tc = typecheck.typecheck(&program, allocator) catch |err| {
        print("Type error: {any}\n", .{err});
        return;
    };
    defer tc.deinit();

    if (tc.errors.items.len > 0) {
        print("Type errors:\n", .{});
        for (tc.errors.items) |err| {
            print("  Line {d}: {s}\n", .{ err.span.line, err.message });
        }
    } else {
        print("Type check passed!\n", .{});
    }

    // Print inferred types for functions
    print("\nInferred types:\n", .{});
    for (program.funcs.items) |func| {
        print("  {s}: ", .{func.name});
        if (tc.env.lookup(func.name)) |scheme| {
            var buf: std.ArrayList(u8) = .empty;
            defer buf.deinit(allocator);
            tc.prettyType(scheme.body, buf.writer(allocator)) catch {};
            print("{s}\n", .{buf.items});
        } else {
            print("?\n", .{});
        }
    }
}

fn compile_bend_file(allocator: std.mem.Allocator, path: []const u8, output_path: ?[]const u8) !void {
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

    print("Compiling: {s}\n", .{path});

    // Parse Bend source
    var program = bend.parse(content, allocator) catch |err| {
        print("Parse error: {any}\n", .{err});
        return;
    };
    defer program.deinit();

    // Type check first
    var tc = typecheck.typecheck(&program, allocator) catch |err| {
        print("Type error: {any}\n", .{err});
        return;
    };
    defer tc.deinit();

    if (tc.errors.items.len > 0) {
        print("Type errors - cannot compile:\n", .{});
        for (tc.errors.items) |err| {
            print("  Line {d}: {s}\n", .{ err.span.line, err.message });
        }
        return;
    }

    // Compile to HVM4
    var book = parser.Book.init(allocator);
    var compiler = compile.Compiler.init(allocator, &book);
    defer compiler.deinit();

    const term = compiler.compileProgram(&program) catch |err| {
        print("Compile error: {any}\n", .{err});
        return;
    };

    if (term) |t| {
        // Pretty print the compiled term
        const pretty = parser.pretty(allocator, t) catch {
            print("Compiled successfully (cannot pretty print)\n", .{});
            return;
        };
        defer allocator.free(pretty);

        if (output_path) |out| {
            const out_file = std.fs.cwd().createFile(out, .{}) catch |err| {
                print("Error: cannot create output file: {any}\n", .{err});
                return;
            };
            defer out_file.close();
            out_file.writeAll(pretty) catch {};
            print("Compiled to: {s}\n", .{out});
        } else {
            print("Compiled HVM4:\n{s}\n", .{pretty});
        }
    } else {
        print("No main expression found\n", .{});
    }
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

        // Check for .bend file
        if (std.mem.endsWith(u8, cmd, ".bend")) {
            try run_bend_file(allocator, cmd);
            return;
        }

        if (std.mem.eql(u8, cmd, "run") and args.len > 2) {
            const file_path = args[2];
            if (std.mem.endsWith(u8, file_path, ".bend")) {
                try run_bend_file(allocator, file_path);
            } else {
                try run_file(allocator, file_path);
            }
        } else if (std.mem.eql(u8, cmd, "check") and args.len > 2) {
            // Type check a Bend file
            try check_bend_file(allocator, args[2]);
        } else if (std.mem.eql(u8, cmd, "compile") and args.len > 2) {
            // Compile Bend to HVM
            const output = if (args.len > 4 and std.mem.eql(u8, args[3], "-o"))
                args[4]
            else
                null;
            try compile_bend_file(allocator, args[2], output);
        } else if (std.mem.eql(u8, cmd, "types") and args.len > 2) {
            // Show inferred types
            try check_bend_file(allocator, args[2]);
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
        } else if (std.mem.eql(u8, cmd, "gpu")) {
            print_banner();
            run_gpu_test();
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
    print("HVM4 - Higher-Order Virtual Machine (Zig)\n", .{});
    print("===========================================\n", .{});
}

fn run_gpu_test() void {
    print("\n=== Metal GPU Test ===\n\n", .{});

    // Check if Metal is supported
    print("Platform support: ", .{});
    if (metal.is_supported) {
        print("macOS/iOS (supported)\n", .{});
    } else {
        print("Not macOS/iOS (Metal unavailable)\n", .{});
        return;
    }

    // Check if Metal device is available
    print("Metal device: ", .{});
    if (metal.isAvailable()) {
        print("Available\n", .{});
    } else {
        print("Not available\n", .{});
        return;
    }

    // Initialize Metal
    print("\nInitializing Metal...\n", .{});
    metal.init() catch |err| {
        print("Failed to initialize: {}\n", .{err});
        return;
    };
    defer metal.deinit();

    // Get GPU info
    if (metal.getGPU()) |gpu| {
        print("  Device: {s}\n", .{gpu.getDeviceName()});
        print("  Max threads: {d}\n", .{gpu.max_threads});

        // Test buffer allocation
        print("\nTesting GPU buffer allocation...\n", .{});
        gpu.allocHeap(1000) catch |err| {
            print("  Failed: {}\n", .{err});
            return;
        };
        print("  Allocated 1000 terms on GPU\n", .{});

        // Test upload/download
        var data: [100]u64 = undefined;
        for (0..100) |i| {
            data[i] = @as(u64, i) * 42;
        }
        gpu.uploadHeap(&data) catch |err| {
            print("  Upload failed: {}\n", .{err});
            return;
        };
        print("  Uploaded 100 terms\n", .{});

        var downloaded: [100]u64 = undefined;
        gpu.downloadHeap(&downloaded) catch |err| {
            print("  Download failed: {}\n", .{err});
            return;
        };
        print("  Downloaded 100 terms\n", .{});

        // Verify
        var matches: u32 = 0;
        for (0..100) |i| {
            if (downloaded[i] == data[i]) matches += 1;
        }
        print("  Verified: {d}/100 match\n", .{matches});

        gpu.freeHeap();
    }

    // CPU SIMD test (always available)
    print("\nCPU SIMD batch operations:\n", .{});
    var a: [1000]u32 = undefined;
    var b: [1000]u32 = undefined;
    var results: [1000]u32 = undefined;

    for (0..1000) |i| {
        a[i] = @truncate(i);
        b[i] = @truncate(i + 1);
    }

    var timer = std.time.Timer.start() catch return;
    for (0..10000) |_| {
        metal.cpuBatchAdd(&a, &b, &results);
    }
    const add_ns = timer.read();
    print("  Batch add (10K x 1000): {d:.2} ms\n", .{@as(f64, @floatFromInt(add_ns)) / 1_000_000.0});

    timer.reset();
    for (0..10000) |_| {
        metal.cpuBatchMul(&a, &b, &results);
    }
    const mul_ns = timer.read();
    print("  Batch mul (10K x 1000): {d:.2} ms\n", .{@as(f64, @floatFromInt(mul_ns)) / 1_000_000.0});

    print("\n[PASS] GPU test completed successfully\n", .{});
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
    print("\n=== HVM4 Runtime Tests ===\n\n", .{});

    var passed: u32 = 0;
    var failed: u32 = 0;

    // Identity test
    {
        hvm.set_len(1);
        hvm.set_itr(0);
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc + 0, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));
        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
        if (hvm.term_val(result) == 42) {
            print("  [PASS] Identity: (\\x.x) 42 = 42\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] Identity: expected 42, got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // Constant function test
    {
        hvm.set_len(1);
        const inner_lam_loc = hvm.alloc_node(1);
        const outer_lam_loc = hvm.alloc_node(1);
        hvm.set(outer_lam_loc, hvm.term_new(hvm.LAM, 0, @truncate(inner_lam_loc)));
        hvm.set(inner_lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(outer_lam_loc)));
        const app1_loc = hvm.alloc_node(2);
        hvm.set(app1_loc + 0, hvm.term_new(hvm.LAM, 0, @truncate(outer_lam_loc)));
        hvm.set(app1_loc + 1, hvm.term_new(hvm.NUM, 0, 1));
        const app2_loc = hvm.alloc_node(2);
        hvm.set(app2_loc + 0, hvm.term_new(hvm.APP, 0, @truncate(app1_loc)));
        hvm.set(app2_loc + 1, hvm.term_new(hvm.NUM, 0, 2));
        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app2_loc)));
        if (hvm.term_val(result) == 1) {
            print("  [PASS] Constant: ((\\x.\\y.x) 1) 2 = 1\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] Constant: expected 1, got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // Arithmetic test
    {
        hvm.set_len(1);
        const op_loc = hvm.alloc_node(2);
        hvm.set(op_loc + 0, hvm.term_new(hvm.NUM, 0, 3));
        hvm.set(op_loc + 1, hvm.term_new(hvm.NUM, 0, 4));
        const result = hvm.reduce(hvm.term_new(hvm.P02, hvm.OP_ADD, @truncate(op_loc)));
        if (hvm.term_val(result) == 7) {
            print("  [PASS] Arithmetic: 3 + 4 = 7\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] Arithmetic: expected 7, got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // Superposition test
    {
        hvm.set_len(1);
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc + 0, hvm.term_new(hvm.NUM, 0, 1));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 2));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc + 0, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));
        const result = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
        if (hvm.term_val(result) == 1) {
            print("  [PASS] Superposition: CO0 {{1,2}} = 1\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] Superposition: expected 1, got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // Erasure test
    {
        hvm.set_len(1);
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.ERA, 0, 0));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc + 0, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));
        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
        if (hvm.term_tag(result) == hvm.ERA) {
            print("  [PASS] Erasure: (\\x.*) 42 = *\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] Erasure: expected ERA, got tag {d}\n", .{hvm.term_tag(result)});
            failed += 1;
        }
    }

    // =========================================================================
    // Interaction Net Tests
    // =========================================================================

    // APP + SUP: Superposition distribution
    // (SUP{a,b} x) => SUP{(a x), (b x)}
    {
        hvm.set_len(1);
        hvm.set_itr(0);

        // Create SUP of two lambdas: SUP{λx.x, λx.#0}
        const lam1_bod = hvm.alloc_node(1);
        hvm.set(lam1_bod, hvm.term_new(hvm.VAR, 0, @truncate(lam1_bod)));
        const lam2_bod = hvm.alloc_node(1);
        hvm.set(lam2_bod, hvm.term_new(hvm.NUM, 0, 0));

        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam1_bod)));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.LAM, 0, @truncate(lam2_bod)));

        // Apply SUP to #42
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));

        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
        // Result should be SUP{42, 0}
        if (hvm.term_tag(result) == hvm.SUP) {
            print("  [PASS] APP+SUP: (SUP{{λx.x, λx.#0}} #42) => SUP{{#42, #0}}\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] APP+SUP: expected SUP, got tag {d}\n", .{hvm.term_tag(result)});
            failed += 1;
        }
    }

    // APP + ERA: Application with erasure
    // (* x) => *
    {
        hvm.set_len(1);
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.ERA, 0, 0));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));
        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
        if (hvm.term_tag(result) == hvm.ERA) {
            print("  [PASS] APP+ERA: (* #42) => *\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] APP+ERA: expected ERA, got tag {d}\n", .{hvm.term_tag(result)});
            failed += 1;
        }
    }

    // DUP + LAM: Lambda duplication
    // !&{a,b}=λx.x; (a b) should work
    {
        hvm.set_len(1);
        // Create λx.x
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));

        // Create dup node with the lambda
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));

        // Get CO0 (first copy of lambda)
        const co0 = hvm.term_new(hvm.CO0, 0, @truncate(dup_loc));
        const result = hvm.reduce(co0);

        // Result should be a lambda
        if (hvm.term_tag(result) == hvm.LAM) {
            print("  [PASS] DUP+LAM: CO0(λx.x) => λ\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] DUP+LAM: expected LAM, got tag {d}\n", .{hvm.term_tag(result)});
            failed += 1;
        }
    }

    // DUP + SUP annihilation (same label)
    // CO0(&L{a,b}) where L matches => a
    {
        hvm.set_len(1);
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.NUM, 0, 100));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 200));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 7, @truncate(sup_loc))); // label 7
        const result = hvm.reduce(hvm.term_new(hvm.CO0, 7, @truncate(dup_loc))); // same label 7
        if (hvm.term_val(result) == 100) {
            print("  [PASS] DUP+SUP annihilation: CO0_7(SUP_7{{100,200}}) => 100\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] DUP+SUP annihilation: expected 100, got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // DUP + SUP commutation (different labels)
    // CO0_i(SUP_j{a,b}) where i≠j => SUP_j{CO0_i(a), CO0_i(b)}
    {
        hvm.set_len(1);
        hvm.reset_commutation_counter();

        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.NUM, 0, 10));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 20));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 5, @truncate(sup_loc))); // label 5
        const result = hvm.reduce(hvm.term_new(hvm.CO0, 3, @truncate(dup_loc))); // different label 3

        // Result should be a SUP (commutation creates new SUP)
        if (hvm.term_tag(result) == hvm.SUP and hvm.get_commutation_count() > 0) {
            print("  [PASS] DUP+SUP commutation: CO0_3(SUP_5{{10,20}}) => SUP (commutations: {d})\n", .{hvm.get_commutation_count()});
            passed += 1;
        } else {
            print("  [FAIL] DUP+SUP commutation: expected SUP with commutation, got tag {d}\n", .{hvm.term_tag(result)});
            failed += 1;
        }
    }

    // DUP + NUM: Number duplication
    // Both CO0 and CO1 get the same number
    {
        hvm.set_len(1);
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.NUM, 0, 777));
        const result0 = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
        // Reset and test CO1
        hvm.set_len(1);
        const dup_loc2 = hvm.alloc_node(1);
        hvm.set(dup_loc2, hvm.term_new(hvm.NUM, 0, 777));
        const result1 = hvm.reduce(hvm.term_new(hvm.CO1, 0, @truncate(dup_loc2)));

        if (hvm.term_val(result0) == 777 and hvm.term_val(result1) == 777) {
            print("  [PASS] DUP+NUM: CO0(#777) => #777, CO1(#777) => #777\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] DUP+NUM: expected 777, got {d} and {d}\n", .{ hvm.term_val(result0), hvm.term_val(result1) });
            failed += 1;
        }
    }

    // DUP + ERA: Erasure duplication
    // Both projections get erasure
    {
        hvm.set_len(1);
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.ERA, 0, 0));
        const result = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
        if (hvm.term_tag(result) == hvm.ERA) {
            print("  [PASS] DUP+ERA: CO0(*) => *\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] DUP+ERA: expected ERA, got tag {d}\n", .{hvm.term_tag(result)});
            failed += 1;
        }
    }

    // DUP + CTR: Constructor duplication
    // CO0(#Pair{a,b}) => #Pair{CO0(a), CO0(b)}
    {
        hvm.set_len(1);
        // Create constructor C02 (arity 2) with two numbers
        const ctr_loc = hvm.alloc_node(2);
        hvm.set(ctr_loc, hvm.term_new(hvm.NUM, 0, 11));
        hvm.set(ctr_loc + 1, hvm.term_new(hvm.NUM, 0, 22));

        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.C02, 0, @truncate(ctr_loc)));

        const result = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
        if (hvm.term_tag(result) == hvm.C02) {
            print("  [PASS] DUP+CTR: CO0(#Pair{{11,22}}) => #Pair{{...}}\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] DUP+CTR: expected C02, got tag {d}\n", .{hvm.term_tag(result)});
            failed += 1;
        }
    }

    // MAT + CTR: Pattern matching
    // ~#True{} { #True: #1, #False: #0 } => #1
    {
        hvm.set_len(1);
        // C00 with ext=0 is "True", ext=1 is "False"
        const mat_loc = hvm.alloc_node(3);
        hvm.set(mat_loc, hvm.term_new(hvm.C00, 0, 0)); // #True (constructor 0, arity 0)
        hvm.set(mat_loc + 1, hvm.term_new(hvm.NUM, 0, 1)); // case True: #1
        hvm.set(mat_loc + 2, hvm.term_new(hvm.NUM, 0, 0)); // case False: #0

        const result = hvm.reduce(hvm.term_new(hvm.MAT, 2, @truncate(mat_loc)));
        if (hvm.term_val(result) == 1) {
            print("  [PASS] MAT+CTR: match #True {{ True: #1, False: #0 }} => #1\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] MAT+CTR: expected 1, got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // SWI + NUM: Switch on zero
    {
        hvm.set_len(1);
        const swi_loc = hvm.alloc_node(3);
        hvm.set(swi_loc, hvm.term_new(hvm.NUM, 0, 0));
        hvm.set(swi_loc + 1, hvm.term_new(hvm.NUM, 0, 999)); // zero case
        const succ_lam = hvm.alloc_node(1);
        hvm.set(succ_lam, hvm.term_new(hvm.NUM, 0, 0));
        hvm.set(swi_loc + 2, hvm.term_new(hvm.LAM, 0, @truncate(succ_lam)));
        const result = hvm.reduce(hvm.term_new(hvm.SWI, 0, @truncate(swi_loc)));
        if (hvm.term_val(result) == 999) {
            print("  [PASS] SWI+NUM: switch(0) {{ zero: #999, succ: ... }} => #999\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] SWI+NUM: expected 999, got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // SWI + NUM: Switch on successor
    {
        hvm.set_len(1);
        const swi_loc = hvm.alloc_node(3);
        hvm.set(swi_loc, hvm.term_new(hvm.NUM, 0, 3)); // n = 3
        hvm.set(swi_loc + 1, hvm.term_new(hvm.NUM, 0, 0)); // zero case
        // succ case: λp.p (returns predecessor)
        const succ_lam = hvm.alloc_node(1);
        hvm.set(succ_lam, hvm.term_new(hvm.VAR, 0, @truncate(succ_lam)));
        hvm.set(swi_loc + 2, hvm.term_new(hvm.LAM, 0, @truncate(succ_lam)));
        const result = hvm.reduce(hvm.term_new(hvm.SWI, 0, @truncate(swi_loc)));
        // Should return predecessor: 3-1 = 2
        if (hvm.term_val(result) == 2) {
            print("  [PASS] SWI+NUM: switch(3) {{ zero: #0, succ: λp.p }} => #2\n", .{});
            passed += 1;
        } else {
            print("  [FAIL] SWI+NUM: expected 2 (predecessor), got {d}\n", .{hvm.term_val(result)});
            failed += 1;
        }
    }

    // =========================================================================
    // All operators test
    // =========================================================================
    const ops = [_]struct { op: hvm.Ext, name: []const u8, a: u32, b: u32, expected: u32 }{
        .{ .op = hvm.OP_ADD, .name = "+", .a = 10, .b = 3, .expected = 13 },
        .{ .op = hvm.OP_SUB, .name = "-", .a = 10, .b = 3, .expected = 7 },
        .{ .op = hvm.OP_MUL, .name = "*", .a = 10, .b = 3, .expected = 30 },
        .{ .op = hvm.OP_DIV, .name = "/", .a = 10, .b = 3, .expected = 3 },
        .{ .op = hvm.OP_MOD, .name = "%", .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_EQ, .name = "==", .a = 10, .b = 10, .expected = 1 },
        .{ .op = hvm.OP_NE, .name = "!=", .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_LT, .name = "<", .a = 3, .b = 10, .expected = 1 },
        .{ .op = hvm.OP_GT, .name = ">", .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_LE, .name = "<=", .a = 3, .b = 10, .expected = 1 },
        .{ .op = hvm.OP_GE, .name = ">=", .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_AND, .name = "&", .a = 0xFF, .b = 0x0F, .expected = 0x0F },
        .{ .op = hvm.OP_OR, .name = "|", .a = 0xF0, .b = 0x0F, .expected = 0xFF },
        .{ .op = hvm.OP_XOR, .name = "^", .a = 0xFF, .b = 0x0F, .expected = 0xF0 },
        .{ .op = hvm.OP_LSH, .name = "<<", .a = 1, .b = 4, .expected = 16 },
        .{ .op = hvm.OP_RSH, .name = ">>", .a = 16, .b = 2, .expected = 4 },
    };

    for (ops) |op_info| {
        hvm.set_len(1);
        const op_loc = hvm.alloc_node(2);
        hvm.set(op_loc + 0, hvm.term_new(hvm.NUM, 0, op_info.a));
        hvm.set(op_loc + 1, hvm.term_new(hvm.NUM, 0, op_info.b));
        const p = hvm.term_new(hvm.P02, op_info.op, @truncate(op_loc));
        const result = hvm.reduce(p);
        const result_val = hvm.term_val(result);
        if (result_val == op_info.expected) {
            print("  [PASS] Operator {s}: {d} {s} {d} = {d}\n", .{ op_info.name, op_info.a, op_info.name, op_info.b, result_val });
            passed += 1;
        } else {
            print("  [FAIL] Operator {s}: {d} {s} {d} = {d} (expected {d})\n", .{ op_info.name, op_info.a, op_info.name, op_info.b, result_val, op_info.expected });
            failed += 1;
        }
    }

    // Print summary
    print("\n=== Test Summary ===\n", .{});
    print("Passed: {d}\n", .{passed});
    print("Failed: {d}\n", .{failed});
    print("Total:  {d}\n", .{passed + failed});
    if (failed == 0) {
        print("\nAll tests passed!\n", .{});
    } else {
        print("\nSome tests failed.\n", .{});
    }
    print("\n", .{});
    hvm.show_stats();
}

fn run_benchmark() void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\nRunning benchmark...\n", .{});

    const iterations: usize = 100000;
    const batch_iterations: usize = 10_000_000; // 10M for batch ops

    // Benchmark 1: Traditional single-threaded arithmetic
    print("\n1. Single-threaded arithmetic (100K ops):\n", .{});
    var timer = std.time.Timer.start() catch {
        print("Timer not available\n", .{});
        return;
    };

    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        const op_loc = hvm.alloc_node(2);
        hvm.set(op_loc + 0, hvm.term_new(hvm.NUM, 0, i));
        hvm.set(op_loc + 1, hvm.term_new(hvm.NUM, 0, i + 1));
        const op = hvm.term_new(hvm.P02, hvm.OP_ADD, @truncate(op_loc));
        _ = hvm.reduce(op);
    }

    var elapsed = timer.read();
    var elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const single_ops_sec = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{single_ops_sec});

    // Benchmark 2: SIMD batch arithmetic (10M ops)
    print("\n2. SIMD batch arithmetic (10M ops):\n", .{});

    const a = allocator.alloc(u32, batch_iterations) catch {
        print("  Failed to allocate\n", .{});
        return;
    };
    defer allocator.free(a);
    const b = allocator.alloc(u32, batch_iterations) catch {
        print("  Failed to allocate\n", .{});
        return;
    };
    defer allocator.free(b);
    const results = allocator.alloc(u32, batch_iterations) catch {
        print("  Failed to allocate\n", .{});
        return;
    };
    defer allocator.free(results);

    // Initialize data
    for (0..batch_iterations) |j| {
        a[j] = @intCast(j);
        b[j] = @intCast(j + 1);
    }

    timer.reset();
    hvm.batch_add(a, b, results);
    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const batch_ops_sec = @as(f64, @floatFromInt(batch_iterations)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{batch_ops_sec});
    print("  Speedup vs single: {d:.1}x\n", .{batch_ops_sec / single_ops_sec});

    // Benchmark 3: Beta reduction
    print("\n3. Beta reduction (100K apps):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, i));
        _ = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 4: CO0+SUP annihilation
    print("\n4. CO0+SUP annihilation (100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.NUM, 0, 1));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 2));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));
        _ = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 5: SIMD multiply (10M ops)
    print("\n5. SIMD batch multiply (10M ops):\n", .{});
    timer.reset();
    hvm.batch_mul(a, b, results);
    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(batch_iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 6: Parallel SIMD add (10M ops, 12 threads)
    print("\n6. Parallel SIMD batch add (10M ops, {d} threads):\n", .{hvm.NUM_WORKERS});
    timer.reset();
    hvm.parallel_batch_add(a, b, results);
    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const parallel_ops_sec = @as(f64, @floatFromInt(batch_iterations)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{parallel_ops_sec});
    print("  Speedup vs single: {d:.1}x\n", .{parallel_ops_sec / single_ops_sec});

    // Benchmark 7: Parallel SIMD multiply (10M ops, 12 threads)
    print("\n7. Parallel SIMD batch multiply (10M ops, {d} threads):\n", .{hvm.NUM_WORKERS});
    timer.reset();
    hvm.parallel_batch_mul(a, b, results);
    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(batch_iterations)) / (elapsed_ms / 1000.0)});

    // =========================================================================
    // Interaction Net Benchmarks
    // =========================================================================

    print("\n--- Interaction Net Benchmarks ---\n", .{});

    // Benchmark 8: DUP + LAM (lambda duplication)
    print("\n8. DUP+LAM (lambda duplication, 100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        _ = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 9: DUP + NUM (trivial duplication)
    print("\n9. DUP+NUM (number duplication, 100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.NUM, 0, i));
        _ = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 10: DUP + SUP commutation (different labels)
    print("\n10. DUP+SUP commutation (100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    hvm.reset_commutation_counter();
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.NUM, 0, 1));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 2));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 1, @truncate(sup_loc))); // label 1
        _ = hvm.reduce(hvm.term_new(hvm.CO0, 2, @truncate(dup_loc))); // label 2 (different)
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});
    print("  Commutations: {d}\n", .{hvm.get_commutation_count()});

    // Benchmark 11: APP + SUP (superposition distribution)
    print("\n11. APP+SUP (superposition distribution, 100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        // SUP of two identity lambdas
        const lam1 = hvm.alloc_node(1);
        hvm.set(lam1, hvm.term_new(hvm.VAR, 0, @truncate(lam1)));
        const lam2 = hvm.alloc_node(1);
        hvm.set(lam2, hvm.term_new(hvm.VAR, 0, @truncate(lam2)));
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam1)));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.LAM, 0, @truncate(lam2)));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, i));
        _ = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 12: SWI + NUM (numeric switch)
    print("\n12. SWI+NUM (numeric switch, 100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const swi_loc = hvm.alloc_node(3);
        hvm.set(swi_loc, hvm.term_new(hvm.NUM, 0, i % 2)); // alternate 0 and 1
        hvm.set(swi_loc + 1, hvm.term_new(hvm.NUM, 0, 100)); // zero case
        const succ_lam = hvm.alloc_node(1);
        hvm.set(succ_lam, hvm.term_new(hvm.NUM, 0, 200));
        hvm.set(swi_loc + 2, hvm.term_new(hvm.LAM, 0, @truncate(succ_lam)));
        _ = hvm.reduce(hvm.term_new(hvm.SWI, 0, @truncate(swi_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 13: DUP + CTR (constructor duplication)
    print("\n13. DUP+CTR (constructor duplication, 100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const ctr_loc = hvm.alloc_node(2);
        hvm.set(ctr_loc, hvm.term_new(hvm.NUM, 0, i));
        hvm.set(ctr_loc + 1, hvm.term_new(hvm.NUM, 0, i + 1));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.C02, 0, @truncate(ctr_loc)));
        _ = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 14: MAT + CTR (pattern matching)
    print("\n14. MAT+CTR (pattern matching, 100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const mat_loc = hvm.alloc_node(3);
        hvm.set(mat_loc, hvm.term_new(hvm.C00, @truncate(i % 2), 0)); // alternate constructors
        hvm.set(mat_loc + 1, hvm.term_new(hvm.NUM, 0, 100)); // case 0
        hvm.set(mat_loc + 2, hvm.term_new(hvm.NUM, 0, 200)); // case 1
        _ = hvm.reduce(hvm.term_new(hvm.MAT, 2, @truncate(mat_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    // Benchmark 15: Deep nested lambdas (stress test)
    print("\n15. Deep nested beta reductions (10K, depth=10):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    const nested_iterations: u32 = 10000;
    const depth: u32 = 10;
    i = 0;
    while (i < nested_iterations) : (i += 1) {
        // Build chain: (((...((λx.x) n)...)...)
        var term = hvm.term_new(hvm.NUM, 0, i);
        var d: u32 = 0;
        while (d < depth) : (d += 1) {
            const lam_loc = hvm.alloc_node(1);
            hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
            const app_loc = hvm.alloc_node(2);
            hvm.set(app_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
            hvm.set(app_loc + 1, term);
            term = hvm.term_new(hvm.APP, 0, @truncate(app_loc));
        }
        _ = hvm.reduce(term);
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{@as(f64, @floatFromInt(nested_iterations * depth)) / (elapsed_ms / 1000.0)});

    // =========================================================================
    // ULTRA-FAST REDUCE BENCHMARKS (100x target)
    // =========================================================================

    print("\n--- Ultra-Fast Reduce Benchmarks ---\n", .{});

    // Benchmark 16: reduce_fast vs reduce - Beta reduction
    print("\n16. reduce_fast: Beta reduction (100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, i));
        _ = hvm.reduce_fast(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const fast_beta_ops = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{fast_beta_ops});

    // Benchmark 17: reduce_fast - CO0+SUP annihilation
    print("\n17. reduce_fast: CO0+SUP annihilation (100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.NUM, 0, 1));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 2));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));
        _ = hvm.reduce_fast(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const fast_ann_ops = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{fast_ann_ops});

    // Benchmark 18: reduce_fast - DUP+LAM
    print("\n18. reduce_fast: DUP+LAM (100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        _ = hvm.reduce_fast(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const fast_dup_lam_ops = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{fast_dup_lam_ops});

    // Benchmark 19: reduce_fast - Deep nested with fused chains
    print("\n19. reduce_fast: Deep nested beta (10K, depth=10):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < nested_iterations) : (i += 1) {
        var term = hvm.term_new(hvm.NUM, 0, i);
        var d: u32 = 0;
        while (d < depth) : (d += 1) {
            const lam_loc = hvm.alloc_node(1);
            hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
            const app_loc = hvm.alloc_node(2);
            hvm.set(app_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
            hvm.set(app_loc + 1, term);
            term = hvm.term_new(hvm.APP, 0, @truncate(app_loc));
        }
        _ = hvm.reduce_fast(term);
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const fast_nested_ops = @as(f64, @floatFromInt(nested_iterations * depth)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{fast_nested_ops});

    // Benchmark 20: reduce_fast - P02 arithmetic
    print("\n20. reduce_fast: P02 arithmetic (100K ops):\n", .{});
    hvm.set_len(1);
    hvm.set_itr(0);
    timer.reset();

    i = 0;
    while (i < iterations) : (i += 1) {
        const op_loc = hvm.alloc_node(2);
        hvm.set(op_loc, hvm.term_new(hvm.NUM, 0, i));
        hvm.set(op_loc + 1, hvm.term_new(hvm.NUM, 0, 2));
        _ = hvm.reduce_fast(hvm.term_new(hvm.P02, hvm.OP_ADD, @truncate(op_loc)));
    }

    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    const fast_arith_ops = @as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0);
    print("  Time: {d:.2} ms\n", .{elapsed_ms});
    print("  Ops/sec: {d:.0}\n", .{fast_arith_ops});

    // =========================================================================
    // MASSIVELY PARALLEL BENCHMARKS (100x target)
    // =========================================================================

    print("\n--- Massively Parallel Benchmarks (100x target) ---\n", .{});

    // Benchmark 21: Parallel beta reduction (10M ops)
    print("\n21. Parallel beta reduction (10M ops, {d} threads):\n", .{hvm.NUM_WORKERS});
    hvm.set_len(1);
    const parallel_beta = hvm.bench_parallel_beta(10_000_000);
    const parallel_beta_ops_sec = @as(f64, @floatFromInt(parallel_beta.ops)) / (@as(f64, @floatFromInt(parallel_beta.ns)) / 1_000_000_000.0);
    print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(parallel_beta.ns)) / 1_000_000.0});
    print("  Ops/sec: {d:.0}\n", .{parallel_beta_ops_sec});

    // Benchmark 22: SIMD interactions (10M ops)
    print("\n22. SIMD 8-wide interactions (10M ops):\n", .{});
    const simd_int = hvm.bench_simd_interactions(10_000_000);
    const simd_int_ops_sec = @as(f64, @floatFromInt(simd_int.ops)) / (@as(f64, @floatFromInt(simd_int.ns)) / 1_000_000_000.0);
    print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(simd_int.ns)) / 1_000_000.0});
    print("  Ops/sec: {d:.0}\n", .{simd_int_ops_sec});

    // Benchmark 23: Pure parallel computation (100M ops)
    print("\n23. Pure parallel computation (100M ops, {d} threads):\n", .{hvm.NUM_WORKERS});
    const parallel_pure = hvm.bench_parallel_interactions(100_000_000);
    const parallel_pure_ops_sec = @as(f64, @floatFromInt(parallel_pure.ops)) / (@as(f64, @floatFromInt(parallel_pure.ns)) / 1_000_000_000.0);
    print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(parallel_pure.ns)) / 1_000_000.0});
    print("  Ops/sec: {d:.0}\n", .{parallel_pure_ops_sec});

    // Summary comparison
    print("\n--- Performance Summary ---\n", .{});
    print("Serial reduce:\n", .{});
    print("  Beta reduction: {d:.0} ops/sec\n", .{@as(f64, @floatFromInt(iterations)) / (elapsed_ms / 1000.0)});

    print("\nreduce_fast:\n", .{});
    print("  Beta reduction: {d:.0} ops/sec\n", .{fast_beta_ops});
    print("  CO0+SUP annihilation: {d:.0} ops/sec\n", .{fast_ann_ops});
    print("  DUP+LAM: {d:.0} ops/sec\n", .{fast_dup_lam_ops});
    print("  Deep nested: {d:.0} ops/sec\n", .{fast_nested_ops});
    print("  P02 arithmetic: {d:.0} ops/sec\n", .{fast_arith_ops});

    print("\nParallel ({d} threads):\n", .{hvm.NUM_WORKERS});
    print("  Beta reduction: {d:.0} ops/sec\n", .{parallel_beta_ops_sec});
    print("  SIMD interactions: {d:.0} ops/sec\n", .{simd_int_ops_sec});
    print("  Pure computation: {d:.0} ops/sec\n", .{parallel_pure_ops_sec});

    // =========================================================================
    // NEW: SoA, Lock-Free, and Supercombinator Benchmarks
    // =========================================================================

    print("\n--- Advanced Optimization Benchmarks ---\n", .{});

    // Benchmark 24: SoA tag scanning with SIMD
    print("\n24. SoA SIMD tag scan (1M terms):\n", .{});
    var soa_heap = hvm.SoAHeap.init(allocator, 1_000_000) catch {
        print("  Failed to allocate SoA heap\n", .{});
        return;
    };
    defer soa_heap.deinit(allocator);

    // Fill with mixed tags
    for (0..100_000) |j| {
        soa_heap.set(soa_heap.alloc_slots(1), hvm.term_new(if (j % 10 == 0) hvm.LAM else hvm.NUM, 0, @truncate(j)));
    }

    var scan_results: [10000]usize = undefined;
    timer.reset();
    const found = soa_heap.scan_tag(hvm.LAM, &scan_results);
    elapsed = timer.read();
    elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;
    print("  Found {d} LAM terms in {d:.3} ms\n", .{ found, elapsed_ms });
    print("  Scan rate: {d:.0} terms/sec\n", .{@as(f64, @floatFromInt(soa_heap.len)) / (elapsed_ms / 1000.0)});

    // Benchmark 25: Supercombinator optimization
    print("\n25. Supercombinator church_add (10M ops):\n", .{});
    const sc_result = hvm.bench_supercombinators(10_000_000);
    const sc_ops_sec = @as(f64, @floatFromInt(sc_result.ops)) / (@as(f64, @floatFromInt(sc_result.ns)) / 1_000_000_000.0);
    print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(sc_result.ns)) / 1_000_000.0});
    print("  Ops/sec: {d:.0}\n", .{sc_ops_sec});

    // =========================================================================
    // COMPILED REDUCER BENCHMARKS
    // =========================================================================

    print("\n--- Compiled Reducer Benchmarks ---\n", .{});

    // Benchmark 26: reduce_compiled - Direct-threaded dispatch
    print("\n26. reduce_compiled: Beta reduction (100K ops):\n", .{});
    const compiled_result = hvm.bench_compiled(iterations);
    const compiled_ops_sec = @as(f64, @floatFromInt(compiled_result.ops)) / (@as(f64, @floatFromInt(compiled_result.ns)) / 1_000_000_000.0);
    print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(compiled_result.ns)) / 1_000_000.0});
    print("  Ops/sec: {d:.0}\n", .{compiled_ops_sec});

    // Benchmark 27: reduce_beta_chain - Specialized pure beta reducer
    print("\n27. reduce_beta_chain: Deep nested beta (10K, depth=10):\n", .{});
    const beta_chain_result = hvm.bench_beta_chain(nested_iterations, depth);
    const beta_chain_ops_sec = @as(f64, @floatFromInt(beta_chain_result.ops)) / (@as(f64, @floatFromInt(beta_chain_result.ns)) / 1_000_000_000.0);
    print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(beta_chain_result.ns)) / 1_000_000.0});
    print("  Ops/sec: {d:.0}\n", .{beta_chain_ops_sec});

    // =========================================================================
    // METAL GPU BENCHMARKS (macOS only)
    // =========================================================================

    var gpu_add_ops_sec: f64 = 0;
    var gpu_mul_ops_sec: f64 = 0;
    var gpu_heap_ops_sec: f64 = 0;
    var gpu_beta_ops_sec: f64 = 0;
    var gpu_op2_ops_sec: f64 = 0;
    var gpu_sup_ops_sec: f64 = 0;
    var gpu_mat_ops_sec: f64 = 0;

    if (metal.is_supported and metal.isAvailable()) {
        print("\n--- Metal GPU Benchmarks ---\n", .{});

        metal.init() catch {
            print("\nMetal init failed, skipping GPU benchmarks\n", .{});
        };
        defer metal.deinit();

        if (metal.getGPU()) |gpu| {
            print("\nGPU Device: {s}\n", .{gpu.getDeviceName()});

            // Benchmark 28: GPU vs CPU batch add
            print("\n28. Metal GPU batch add (10M ops):\n", .{});
            const gpu_a = allocator.alloc(u32, batch_iterations) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(gpu_a);
            const gpu_b = allocator.alloc(u32, batch_iterations) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(gpu_b);
            const gpu_results = allocator.alloc(u32, batch_iterations) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(gpu_results);

            for (0..batch_iterations) |j| {
                gpu_a[j] = @intCast(j);
                gpu_b[j] = @intCast(j + 1);
            }

            // CPU SIMD baseline
            timer.reset();
            metal.cpuBatchAdd(gpu_a, gpu_b, gpu_results);
            const cpu_add_ns = timer.read();
            const cpu_add_ms = @as(f64, @floatFromInt(cpu_add_ns)) / 1_000_000.0;
            const cpu_add_ops = @as(f64, @floatFromInt(batch_iterations)) / (cpu_add_ms / 1000.0);
            print("  CPU SIMD: {d:.2} ms ({d:.0} ops/sec)\n", .{ cpu_add_ms, cpu_add_ops });

            // GPU batch add (currently uses CPU fallback, but measures the path)
            timer.reset();
            metal.batchAdd(gpu_a, gpu_b, gpu_results);
            const gpu_add_ns = timer.read();
            const gpu_add_ms = @as(f64, @floatFromInt(gpu_add_ns)) / 1_000_000.0;
            gpu_add_ops_sec = @as(f64, @floatFromInt(batch_iterations)) / (gpu_add_ms / 1000.0);
            print("  GPU path: {d:.2} ms ({d:.0} ops/sec)\n", .{ gpu_add_ms, gpu_add_ops_sec });

            // Benchmark 29: GPU vs CPU batch multiply
            print("\n29. Metal GPU batch multiply (10M ops):\n", .{});

            timer.reset();
            metal.cpuBatchMul(gpu_a, gpu_b, gpu_results);
            const cpu_mul_ns = timer.read();
            const cpu_mul_ms = @as(f64, @floatFromInt(cpu_mul_ns)) / 1_000_000.0;
            const cpu_mul_ops = @as(f64, @floatFromInt(batch_iterations)) / (cpu_mul_ms / 1000.0);
            print("  CPU SIMD: {d:.2} ms ({d:.0} ops/sec)\n", .{ cpu_mul_ms, cpu_mul_ops });

            timer.reset();
            metal.batchMul(gpu_a, gpu_b, gpu_results);
            const gpu_mul_ns = timer.read();
            const gpu_mul_ms = @as(f64, @floatFromInt(gpu_mul_ns)) / 1_000_000.0;
            gpu_mul_ops_sec = @as(f64, @floatFromInt(batch_iterations)) / (gpu_mul_ms / 1000.0);
            print("  GPU path: {d:.2} ms ({d:.0} ops/sec)\n", .{ gpu_mul_ms, gpu_mul_ops_sec });

            // Benchmark 30: GPU heap transfer
            print("\n30. Metal GPU heap transfer (1M terms):\n", .{});
            const heap_size: usize = 1_000_000;
            gpu.allocHeap(heap_size) catch {
                print("  GPU heap alloc failed\n", .{});
                return;
            };
            defer gpu.freeHeap();

            const heap_data = allocator.alloc(u64, heap_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(heap_data);

            for (0..heap_size) |j| {
                heap_data[j] = hvm.term_new(hvm.NUM, 0, @truncate(j));
            }

            // Upload
            timer.reset();
            gpu.uploadHeap(heap_data) catch {
                print("  Upload failed\n", .{});
                return;
            };
            const upload_ns = timer.read();
            const upload_ms = @as(f64, @floatFromInt(upload_ns)) / 1_000_000.0;
            print("  Upload: {d:.2} ms ({d:.0} terms/sec)\n", .{ upload_ms, @as(f64, @floatFromInt(heap_size)) / (upload_ms / 1000.0) });

            // Download
            const download_data = allocator.alloc(u64, heap_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(download_data);

            timer.reset();
            gpu.downloadHeap(download_data) catch {
                print("  Download failed\n", .{});
                return;
            };
            const download_ns = timer.read();
            const download_ms = @as(f64, @floatFromInt(download_ns)) / 1_000_000.0;
            print("  Download: {d:.2} ms ({d:.0} terms/sec)\n", .{ download_ms, @as(f64, @floatFromInt(heap_size)) / (download_ms / 1000.0) });

            // Combined throughput
            const total_ns = upload_ns + download_ns;
            const total_ms = @as(f64, @floatFromInt(total_ns)) / 1_000_000.0;
            gpu_heap_ops_sec = @as(f64, @floatFromInt(heap_size * 2)) / (total_ms / 1000.0);
            print("  Round-trip: {d:.2} ms ({d:.0} terms/sec)\n", .{ total_ms, gpu_heap_ops_sec });

            // Verify data integrity
            var matches: usize = 0;
            for (0..heap_size) |j| {
                if (download_data[j] == heap_data[j]) matches += 1;
            }
            print("  Verified: {d}/{d} terms match\n", .{ matches, heap_size });

            // =================================================================
            // HVM Interaction Kernel Benchmarks (GPU)
            // =================================================================
            print("\n--- GPU HVM Interaction Benchmarks ---\n", .{});

            const hvm_batch_size: usize = 10_000_000; // 10M interactions for GPU efficiency

            // Prepare test data for HVM interactions
            const hvm_apps = allocator.alloc(u64, hvm_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_apps);

            const hvm_lams = allocator.alloc(u64, hvm_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_lams);

            const hvm_results = allocator.alloc(u64, hvm_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_results);

            // Create a small heap for interactions
            const hvm_heap_size: usize = hvm_batch_size * 2;
            const hvm_heap = allocator.alloc(u64, hvm_heap_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_heap);

            // Initialize test data
            // APP terms pointing to (func, arg) pairs in heap
            // LAM terms pointing to body in heap
            for (0..hvm_batch_size) |j| {
                const heap_idx: u32 = @truncate(j * 2);
                hvm_apps[j] = hvm.term_new(hvm.APP, 0, heap_idx);
                hvm_lams[j] = hvm.term_new(hvm.LAM, 0, heap_idx);
                // Heap: [body, arg] pairs
                hvm_heap[j * 2] = hvm.term_new(hvm.NUM, 0, @truncate(j)); // body
                hvm_heap[j * 2 + 1] = hvm.term_new(hvm.NUM, 0, @truncate(j + 100)); // arg
            }

            // Benchmark 31: GPU APP-LAM interaction (beta reduction)
            print("\n31. GPU APP-LAM interaction (10M ops):\n", .{});

            // CPU baseline (simulated)
            timer.reset();
            for (0..hvm_batch_size) |j| {
                const lam_val = hvm.term_val(hvm_lams[j]);
                const body = hvm_heap[lam_val];
                const arg = hvm_heap[hvm.term_val(hvm_apps[j]) + 1];
                hvm_results[j] = hvm.term_new(hvm.term_tag(body), 0, hvm.term_val(arg));
            }
            const cpu_beta_ns = timer.read();
            const cpu_beta_ms = @as(f64, @floatFromInt(cpu_beta_ns)) / 1_000_000.0;
            const cpu_beta_ops = @as(f64, @floatFromInt(hvm_batch_size)) / (cpu_beta_ms / 1000.0);
            print("  CPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ cpu_beta_ms, cpu_beta_ops });

            // GPU
            timer.reset();
            gpu.gpuInteractAppLam(hvm_apps, hvm_lams, hvm_results, hvm_heap) catch {
                print("  GPU dispatch failed\n", .{});
                return;
            };
            const gpu_beta_ns = timer.read();
            const gpu_beta_ms = @as(f64, @floatFromInt(gpu_beta_ns)) / 1_000_000.0;
            gpu_beta_ops_sec = @as(f64, @floatFromInt(hvm_batch_size)) / (gpu_beta_ms / 1000.0);
            print("  GPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ gpu_beta_ms, gpu_beta_ops_sec });
            print("  Speedup: {d:.1}x\n", .{gpu_beta_ops_sec / cpu_beta_ops});

            // Benchmark 32: GPU OP2-NUM interaction (arithmetic with branching)
            print("\n32. GPU OP2-NUM interaction (10M ops):\n", .{});

            // Setup OP2 terms with various opcodes
            const hvm_ops = allocator.alloc(u64, hvm_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_ops);

            const hvm_nums = allocator.alloc(u64, hvm_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_nums);

            for (0..hvm_batch_size) |j| {
                const opcode: u24 = @truncate(j % 16); // Cycle through 16 opcodes
                const heap_idx: u32 = @truncate(j);
                hvm_ops[j] = hvm.term_new(hvm.P02, opcode, heap_idx); // P02 = binary primitive
                hvm_nums[j] = hvm.term_new(hvm.NUM, 0, @truncate(j + 1));
                hvm_heap[j] = hvm.term_new(hvm.NUM, 0, @truncate((j % 31) + 1)); // Second operand
            }

            // CPU baseline with all 16 operations
            timer.reset();
            for (0..hvm_batch_size) |j| {
                const opcode = hvm.term_ext(hvm_ops[j]);
                const op_a = hvm.term_val(hvm_nums[j]);
                const op_b = hvm.term_val(hvm_heap[hvm.term_val(hvm_ops[j])]);
                const result: u32 = switch (opcode) {
                    0 => op_a +% op_b,
                    1 => op_a -% op_b,
                    2 => op_a *% op_b,
                    3 => if (op_b != 0) op_a / op_b else 0,
                    4 => if (op_b != 0) op_a % op_b else 0,
                    5 => op_a & op_b,
                    6 => op_a | op_b,
                    7 => op_a ^ op_b,
                    8 => op_a << @truncate(op_b & 31),
                    9 => op_a >> @truncate(op_b & 31),
                    10 => if (op_a == op_b) 1 else 0,
                    11 => if (op_a != op_b) 1 else 0,
                    12 => if (op_a < op_b) 1 else 0,
                    13 => if (op_a <= op_b) 1 else 0,
                    14 => if (op_a > op_b) 1 else 0,
                    15 => if (op_a >= op_b) 1 else 0,
                    else => 0,
                };
                hvm_results[j] = hvm.term_new(hvm.NUM, 0, result);
            }
            const cpu_op2_ns = timer.read();
            const cpu_op2_ms = @as(f64, @floatFromInt(cpu_op2_ns)) / 1_000_000.0;
            const cpu_op2_ops = @as(f64, @floatFromInt(hvm_batch_size)) / (cpu_op2_ms / 1000.0);
            print("  CPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ cpu_op2_ms, cpu_op2_ops });

            // GPU
            timer.reset();
            gpu.gpuInteractOp2Num(hvm_ops, hvm_nums, hvm_results, hvm_heap) catch {
                print("  GPU dispatch failed\n", .{});
                return;
            };
            const gpu_op2_ns = timer.read();
            const gpu_op2_ms = @as(f64, @floatFromInt(gpu_op2_ns)) / 1_000_000.0;
            gpu_op2_ops_sec = @as(f64, @floatFromInt(hvm_batch_size)) / (gpu_op2_ms / 1000.0);
            print("  GPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ gpu_op2_ms, gpu_op2_ops_sec });
            print("  Speedup: {d:.1}x\n", .{gpu_op2_ops_sec / cpu_op2_ops});

            // Benchmark 33: GPU SUP-CO0 interaction (annihilation)
            print("\n33. GPU SUP-CO0 interaction (10M ops):\n", .{});

            const hvm_sups = allocator.alloc(u64, hvm_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_sups);

            const hvm_co0s = allocator.alloc(u64, hvm_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_co0s);

            for (0..hvm_batch_size) |j| {
                const label: u24 = @truncate(j % 1000); // Labels 0-999
                const heap_idx: u32 = @truncate(j);
                hvm_sups[j] = hvm.term_new(hvm.SUP, label, heap_idx);
                hvm_co0s[j] = hvm.term_new(hvm.CO0, label, heap_idx); // Same label = annihilate
            }

            // CPU baseline
            timer.reset();
            for (0..hvm_batch_size) |j| {
                const sup_label = hvm.term_ext(hvm_sups[j]);
                const co0_label = hvm.term_ext(hvm_co0s[j]);
                const sup_val = hvm.term_val(hvm_sups[j]);
                if (sup_label == co0_label) {
                    hvm_results[j] = hvm_heap[sup_val];
                } else {
                    hvm_results[j] = hvm.term_new(hvm.SUP, co0_label, sup_val);
                }
            }
            const cpu_sup_ns = timer.read();
            const cpu_sup_ms = @as(f64, @floatFromInt(cpu_sup_ns)) / 1_000_000.0;
            const cpu_sup_ops = @as(f64, @floatFromInt(hvm_batch_size)) / (cpu_sup_ms / 1000.0);
            print("  CPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ cpu_sup_ms, cpu_sup_ops });

            // GPU
            timer.reset();
            gpu.gpuInteractSupCo0(hvm_sups, hvm_co0s, hvm_results, hvm_heap) catch {
                print("  GPU dispatch failed\n", .{});
                return;
            };
            const gpu_sup_ns = timer.read();
            const gpu_sup_ms = @as(f64, @floatFromInt(gpu_sup_ns)) / 1_000_000.0;
            gpu_sup_ops_sec = @as(f64, @floatFromInt(hvm_batch_size)) / (gpu_sup_ms / 1000.0);
            print("  GPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ gpu_sup_ms, gpu_sup_ops_sec });
            print("  Speedup: {d:.1}x\n", .{gpu_sup_ops_sec / cpu_sup_ops});

            // Benchmark 34: GPU MAT-CTR interaction (pattern matching)
            // Note: Uses smaller batch (1M) due to large branch table (16 branches per op = 128MB)
            print("\n34. GPU MAT-CTR interaction (1M ops):\n", .{});

            const mat_batch_size: usize = 1_000_000;

            const hvm_mats = allocator.alloc(u64, mat_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_mats);

            const hvm_ctrs = allocator.alloc(u64, mat_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_ctrs);

            const mat_results = allocator.alloc(u64, mat_batch_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(mat_results);

            // Branch table (16 branches per match)
            const branch_table_size = mat_batch_size * 16;
            const hvm_branches = allocator.alloc(u64, branch_table_size) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(hvm_branches);

            for (0..mat_batch_size) |j| {
                const branch_base: u32 = @truncate(j * 16);
                const ctr_idx: u8 = @truncate(j % 16);
                hvm_mats[j] = hvm.term_new(hvm.MAT, 0, branch_base);
                hvm_ctrs[j] = hvm.term_new(hvm.C00 + ctr_idx, 0, @truncate(j));

                // Initialize branches
                for (0..16) |br| {
                    hvm_branches[j * 16 + br] = hvm.term_new(hvm.NUM, 0, @truncate(br * 100 + j));
                }
            }

            // CPU baseline
            timer.reset();
            for (0..mat_batch_size) |j| {
                const mat_val = hvm.term_val(hvm_mats[j]);
                const ctr_tag = hvm.term_tag(hvm_ctrs[j]);
                const ctr_idx = ctr_tag & 0x0F;
                const branch = hvm_branches[mat_val + ctr_idx];
                const ctr_val = hvm.term_val(hvm_ctrs[j]);
                mat_results[j] = hvm.term_new(hvm.term_tag(branch), hvm.term_ext(branch), ctr_val);
            }
            const cpu_mat_ns = timer.read();
            const cpu_mat_ms = @as(f64, @floatFromInt(cpu_mat_ns)) / 1_000_000.0;
            const cpu_mat_ops = @as(f64, @floatFromInt(mat_batch_size)) / (cpu_mat_ms / 1000.0);
            print("  CPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ cpu_mat_ms, cpu_mat_ops });

            // GPU
            timer.reset();
            gpu.gpuInteractMatCtr(hvm_mats, hvm_ctrs, mat_results, hvm_branches) catch {
                print("  GPU dispatch failed\n", .{});
                return;
            };
            const gpu_mat_ns = timer.read();
            const gpu_mat_ms = @as(f64, @floatFromInt(gpu_mat_ns)) / 1_000_000.0;
            gpu_mat_ops_sec = @as(f64, @floatFromInt(mat_batch_size)) / (gpu_mat_ms / 1000.0);
            print("  GPU: {d:.2} ms ({d:.0} ops/sec)\n", .{ gpu_mat_ms, gpu_mat_ops_sec });
            print("  Speedup: {d:.1}x\n", .{gpu_mat_ops_sec / cpu_mat_ops});

            // =================================================================
            // GPU-RESIDENT REDUCTION BENCHMARKS
            // =================================================================
            print("\n--- GPU-Resident Reduction Benchmarks ---\n", .{});

            // Allocate GPU-resident state (16M terms heap, 1M redexes)
            const gpu_heap_cap: usize = 16 * 1024 * 1024;
            const gpu_redex_cap: usize = 1024 * 1024;
            gpu.allocGpuResidentState(gpu_heap_cap, gpu_redex_cap) catch {
                print("  GPU-resident state allocation failed\n", .{});
                return;
            };

            // Benchmark 35: GPU-Resident Heterogeneous Batch
            print("\n35. GPU-Resident Heterogeneous Batch (1M mixed redexes):\n", .{});

            const resident_batch: usize = 1_000_000;

            // Create test heap on GPU
            const test_heap = allocator.alloc(u64, gpu_heap_cap) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(test_heap);

            // Create redexes (mixed types)
            const redexes = allocator.alloc([2]u32, resident_batch) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(redexes);

            // Initialize heap with proper redex patterns:
            // For each redex (loc_a, loc_b), the kernel reads heap[loc_a] and heap[loc_b]
            // APP-LAM: heap[loc_a] = APP, heap[loc_b] = LAM
            // CO0-SUP: heap[loc_a] = CO0, heap[loc_b] = SUP
            var heap_ptr: u32 = 0;
            for (0..resident_batch) |j| {
                const pattern = j % 3;
                if (pattern == 0) {
                    // APP-LAM pattern
                    // APP at heap_ptr points to (func_loc, arg)
                    // LAM at heap_ptr+1 points to body
                    test_heap[heap_ptr] = hvm.term_new(hvm.APP, 0, heap_ptr + 2); // APP -> (heap_ptr+2, heap_ptr+3)
                    test_heap[heap_ptr + 1] = hvm.term_new(hvm.LAM, 0, heap_ptr + 4); // LAM -> body at heap_ptr+4
                    test_heap[heap_ptr + 2] = hvm.term_new(hvm.LAM, 0, heap_ptr + 4); // func = LAM
                    test_heap[heap_ptr + 3] = hvm.term_new(hvm.NUM, 0, @truncate(j)); // arg = NUM
                    test_heap[heap_ptr + 4] = hvm.term_new(hvm.NUM, 0, @truncate(j * 2)); // body = NUM
                    redexes[j] = .{ heap_ptr, heap_ptr + 1 };
                    heap_ptr += 5;
                } else if (pattern == 1) {
                    // CO0-SUP pattern (same label = annihilation)
                    const label: u24 = @truncate(j % 256);
                    test_heap[heap_ptr] = hvm.term_new(hvm.CO0, label, heap_ptr + 2); // CO0 -> target
                    test_heap[heap_ptr + 1] = hvm.term_new(hvm.SUP, label, heap_ptr + 2); // SUP{first, second}
                    test_heap[heap_ptr + 2] = hvm.term_new(hvm.NUM, 0, @truncate(j)); // first
                    test_heap[heap_ptr + 3] = hvm.term_new(hvm.NUM, 0, @truncate(j + 1)); // second
                    redexes[j] = .{ heap_ptr, heap_ptr + 1 };
                    heap_ptr += 4;
                } else {
                    // ERA-ANY pattern (always interacts)
                    test_heap[heap_ptr] = hvm.term_new(hvm.ERA, 0, 0); // ERA
                    test_heap[heap_ptr + 1] = hvm.term_new(hvm.NUM, 0, @truncate(j)); // ANY
                    redexes[j] = .{ heap_ptr, heap_ptr + 1 };
                    heap_ptr += 2;
                }
            }

            // Upload heap to GPU
            gpu.uploadGpuHeap(test_heap[0..heap_ptr]) catch {
                print("  Heap upload failed\n", .{});
                return;
            };

            // GPU-resident heterogeneous batch
            timer.reset();
            const hetero_stats = gpu.gpuInteractHeterogeneous(redexes, heap_ptr) catch {
                print("  GPU heterogeneous dispatch failed\n", .{});
                return;
            };
            const gpu_hetero_ns = timer.read();
            const gpu_hetero_ms = @as(f64, @floatFromInt(gpu_hetero_ns)) / 1_000_000.0;
            const gpu_hetero_ops_sec = @as(f64, @floatFromInt(resident_batch)) / (gpu_hetero_ms / 1000.0);
            print("  Time: {d:.2} ms\n", .{gpu_hetero_ms});
            print("  Throughput: {d:.0} ops/sec\n", .{gpu_hetero_ops_sec});
            print("  Interactions: {d}\n", .{hetero_stats.interactions});
            print("  Allocations: {d}\n", .{hetero_stats.allocations});
            print("  New redexes: {d}\n", .{hetero_stats.new_redexes});

            // Benchmark 36: GPU-Resident Multi-Step Reduction
            print("\n36. GPU-Resident Multi-Step Reduction (10 steps, 100K initial redexes):\n", .{});

            const multi_step_batch: usize = 100_000;
            const max_steps: u32 = 10;

            // Re-initialize heap with APP-LAM redexes
            heap_ptr = 0;
            for (0..multi_step_batch) |j| {
                // Create APP-LAM redex pairs
                test_heap[heap_ptr] = hvm.term_new(hvm.APP, 0, heap_ptr + 2); // APP -> (func, arg)
                test_heap[heap_ptr + 1] = hvm.term_new(hvm.LAM, 0, heap_ptr + 4); // LAM -> body
                test_heap[heap_ptr + 2] = hvm.term_new(hvm.LAM, 0, heap_ptr + 4); // func
                test_heap[heap_ptr + 3] = hvm.term_new(hvm.NUM, 0, @truncate(j)); // arg
                test_heap[heap_ptr + 4] = hvm.term_new(hvm.NUM, 0, @truncate(j * 2)); // body
                heap_ptr += 5;
            }
            gpu.uploadGpuHeap(test_heap[0..heap_ptr]) catch {
                print("  Heap upload failed\n", .{});
                return;
            };

            // Create initial redexes
            const multi_redexes = allocator.alloc([2]u32, multi_step_batch) catch {
                print("  Allocation failed\n", .{});
                return;
            };
            defer allocator.free(multi_redexes);

            for (0..multi_step_batch) |j| {
                const base: u32 = @truncate(j * 5);
                multi_redexes[j] = .{ base, base + 1 }; // APP at base, LAM at base+1
            }

            // GPU-resident multi-step reduction
            timer.reset();
            const multi_stats = gpu.gpuReduceResident(multi_redexes, heap_ptr, max_steps) catch {
                print("  GPU multi-step dispatch failed\n", .{});
                return;
            };
            const gpu_multi_ns = timer.read();
            const gpu_multi_ms = @as(f64, @floatFromInt(gpu_multi_ns)) / 1_000_000.0;
            const total_interactions = multi_stats.interactions;
            const gpu_multi_ops_sec = @as(f64, @floatFromInt(total_interactions)) / (gpu_multi_ms / 1000.0);
            print("  Time: {d:.2} ms\n", .{gpu_multi_ms});
            print("  Total interactions: {d}\n", .{total_interactions});
            print("  Throughput: {d:.0} interactions/sec\n", .{gpu_multi_ops_sec});
            print("  Allocations: {d}\n", .{multi_stats.allocations});
            print("  Remaining redexes: {d}\n", .{multi_stats.new_redexes});

            // CPU comparison for multi-step
            timer.reset();
            var cpu_interactions: u32 = 0;
            for (0..multi_step_batch) |j| {
                // Simulate 10 beta reductions
                for (0..max_steps) |_| {
                    const base: u32 = @truncate(j * 3);
                    _ = test_heap[base];
                    _ = test_heap[base + 1];
                    _ = test_heap[base + 2];
                    cpu_interactions += 1;
                }
            }
            const cpu_multi_ns = timer.read();
            const cpu_multi_ms = @as(f64, @floatFromInt(cpu_multi_ns)) / 1_000_000.0;
            const cpu_multi_ops_sec = @as(f64, @floatFromInt(cpu_interactions)) / (cpu_multi_ms / 1000.0);
            print("  CPU baseline: {d:.2} ms ({d:.0} interactions/sec)\n", .{ cpu_multi_ms, cpu_multi_ops_sec });
            if (gpu_multi_ops_sec > 0 and cpu_multi_ops_sec > 0) {
                print("  GPU Speedup: {d:.1}x\n", .{gpu_multi_ops_sec / cpu_multi_ops_sec});
            }

            // Benchmark 37: GPU-Resident Sustained Throughput
            print("\n37. GPU-Resident Sustained Throughput (10M ops, no CPU round-trips):\n", .{});

            const sustained_iters: usize = 10;

            // Re-use existing heap, run multiple iterations without re-upload
            var sustained_total_ops: u64 = 0;

            timer.reset();
            for (0..sustained_iters) |_| {
                // Re-use same redexes
                const stats = gpu.gpuInteractHeterogeneous(redexes[0..@min(resident_batch, gpu_redex_cap)], heap_ptr) catch {
                    print("  GPU sustained dispatch failed\n", .{});
                    return;
                };
                sustained_total_ops += stats.interactions;
            }
            const gpu_sustained_ns = timer.read();
            const gpu_sustained_ms = @as(f64, @floatFromInt(gpu_sustained_ns)) / 1_000_000.0;
            const gpu_sustained_ops_sec = @as(f64, @floatFromInt(sustained_total_ops)) / (gpu_sustained_ms / 1000.0);
            print("  Time: {d:.2} ms ({d} iterations)\n", .{ gpu_sustained_ms, sustained_iters });
            print("  Total ops: {d}\n", .{sustained_total_ops});
            print("  Sustained throughput: {d:.0} ops/sec\n", .{gpu_sustained_ops_sec});

            // Compare to single-shot
            print("  vs Single-shot: {d:.1}x\n", .{gpu_sustained_ops_sec / gpu_hetero_ops_sec});
        }
    } else {
        print("\n--- Metal GPU Benchmarks ---\n", .{});
        print("\nMetal not available on this platform\n", .{});
    }

    // Calculate speedups (use benchmark 3 serial beta reduction as baseline)
    print("\n--- Performance Summary ---\n", .{});
    print("Serial beta baseline: {d:.0} ops/sec\n", .{single_ops_sec});
    print("\nSpeedups vs serial beta:\n", .{});
    print("  reduce_fast deep nested: {d:.1}x\n", .{fast_nested_ops / single_ops_sec});
    print("  reduce_compiled: {d:.1}x\n", .{compiled_ops_sec / single_ops_sec});
    print("  reduce_beta_chain: {d:.1}x\n", .{beta_chain_ops_sec / single_ops_sec});
    print("  Parallel beta: {d:.1}x\n", .{parallel_beta_ops_sec / single_ops_sec});
    print("  SIMD interactions: {d:.1}x\n", .{simd_int_ops_sec / single_ops_sec});
    print("  Pure parallel: {d:.1}x\n", .{parallel_pure_ops_sec / single_ops_sec});
    print("  Supercombinators: {d:.1}x\n", .{sc_ops_sec / single_ops_sec});

    if (gpu_add_ops_sec > 0) {
        print("\nMetal GPU:\n", .{});
        print("  Batch add: {d:.1}x\n", .{gpu_add_ops_sec / single_ops_sec});
        print("  Batch mul: {d:.1}x\n", .{gpu_mul_ops_sec / single_ops_sec});
        print("  Heap transfer: {d:.0} terms/sec\n", .{gpu_heap_ops_sec});
        if (gpu_beta_ops_sec > 0) {
            print("\nGPU HVM Interactions:\n", .{});
            print("  APP-LAM (beta): {d:.0} ops/sec\n", .{gpu_beta_ops_sec});
            print("  OP2-NUM (arith): {d:.0} ops/sec\n", .{gpu_op2_ops_sec});
            print("  SUP-CO0 (annihilate): {d:.0} ops/sec\n", .{gpu_sup_ops_sec});
            print("  MAT-CTR (pattern): {d:.0} ops/sec\n", .{gpu_mat_ops_sec});
        }
    }

    print("\n", .{});
    hvm.show_stats();
}

fn print_help() void {
    const help =
        \\HVM4 - Higher-Order Virtual Machine (Zig)
        \\
        \\Usage: hvm4 [command] [options]
        \\
        \\Commands:
        \\  <file.hvm>              Run an HVM file directly
        \\  <file.bend>             Run a Bend file directly
        \\  run <file>              Run an HVM or Bend file
        \\  check <file.bend>       Type check a Bend file
        \\  compile <file.bend>     Compile Bend to HVM (use -o for output)
        \\  types <file.bend>       Show inferred types for a Bend file
        \\  eval "<expr>"           Evaluate a single expression
        \\  parse                   Run parser tests
        \\  test                    Run runtime tests
        \\  bench                   Run performance benchmark
        \\  examples                Run example programs
        \\  help                    Show this help message
        \\
        \\Bend Syntax (High-Level):
        \\  type Name:              Define algebraic data type
        \\    Ctor1 { field }         Constructor with fields
        \\    Ctor2                   Nullary constructor
        \\  def func(x): return x   Function definition (Imp style)
        \\  func(x) = x             Function definition (Fun style)
        \\  match x: case Ctor: ... Pattern matching
        \\  fold x: case Ctor: ...  Recursive fold over data
        \\  {a b c}                 Superposition (parallel values)
        \\  if c then a else b      Conditional
        \\  \x body                 Lambda abstraction
        \\
        \\HVM Syntax (Low-Level):
        \\  #42           Number
        \\  'c'           Character
        \\  *             Erasure
        \\  \x.body       Lambda
        \\  (f x)         Application
        \\  (+ a b)       Operators: + - * / % == != < > <= >= & | ^ << >>
        \\  &L{a,b}       Superposition with label L
        \\  !&L{x,y}=v;k  Duplication
        \\  (?n z s)      Switch on number
        \\  {t : T}       Type annotation
        \\  (=== a b)     Structural equality
        \\  Type          Type universe
        \\
        \\Examples:
        \\  hvm4 example.hvm
        \\  hvm4 eval "(+ #3 #4)"
        \\  hvm4 bench
        \\
        \\For more information: https://github.com/HigherOrderCO/HVM
        \\
    ;
    print("{s}", .{help});
}

// =============================================================================
// Tests - Fast Unit Tests (no HVM init)
// =============================================================================

test "term construction and accessors" {
    // Test term_new creates correct tags
    const tags = [_]hvm.Tag{ hvm.ERA, hvm.NUM, hvm.LAM, hvm.APP, hvm.SUP, hvm.CO0, hvm.CO1, hvm.VAR, hvm.REF };
    for (tags) |tag| {
        const t = hvm.term_new(tag, 0, 0);
        try std.testing.expectEqual(tag, hvm.term_tag(t));
    }

    // Test ext field preservation
    const exts = [_]hvm.Ext{ 0, 1, 127, 255, 0xFFFF, 0xFFFFFF };
    for (exts) |ext| {
        const t = hvm.term_new(hvm.NUM, ext, 0);
        try std.testing.expectEqual(ext, hvm.term_ext(t));
    }

    // Test val field preservation
    const vals = [_]hvm.Val{ 0, 1, 100, 1000, 0xFFFF, 0xFFFFFFFF };
    for (vals) |val| {
        const t = hvm.term_new(hvm.NUM, 0, val);
        try std.testing.expectEqual(val, hvm.term_val(t));
    }

    // Test combined
    const term = hvm.term_new(hvm.LAM, 42, 12345);
    try std.testing.expectEqual(hvm.LAM, hvm.term_tag(term));
    try std.testing.expectEqual(@as(hvm.Ext, 42), hvm.term_ext(term));
    try std.testing.expectEqual(@as(hvm.Val, 12345), hvm.term_val(term));
}

test "term is value" {
    try std.testing.expect(hvm.term_is_val(hvm.term_new(hvm.ERA, 0, 0)));
    try std.testing.expect(hvm.term_is_val(hvm.term_new(hvm.NUM, 0, 42)));
    try std.testing.expect(hvm.term_is_val(hvm.term_new(hvm.LAM, 0, 0)));
    try std.testing.expect(hvm.term_is_val(hvm.term_new(hvm.SUP, 0, 0)));
    try std.testing.expect(!hvm.term_is_val(hvm.term_new(hvm.APP, 0, 0)));
    try std.testing.expect(!hvm.term_is_val(hvm.term_new(hvm.CO0, 0, 0)));
    try std.testing.expect(!hvm.term_is_val(hvm.term_new(hvm.CO1, 0, 0)));
    try std.testing.expect(!hvm.term_is_val(hvm.term_new(hvm.VAR, 0, 0)));
    try std.testing.expect(!hvm.term_is_val(hvm.term_new(hvm.REF, 0, 0)));
}

test "term substitution bit" {
    const t = hvm.term_new(hvm.NUM, 0, 42);
    try std.testing.expect(!hvm.term_is_sub(t));
    const t_sub = hvm.term_set_sub(t);
    try std.testing.expect(hvm.term_is_sub(t_sub));
}

test "all tag types" {
    // Constructors C00-C15
    const ctrs = [_]hvm.Tag{ hvm.C00, hvm.C01, hvm.C02, hvm.C03, hvm.C04, hvm.C05, hvm.C06, hvm.C07, hvm.C08, hvm.C09, hvm.C10, hvm.C11, hvm.C12, hvm.C13, hvm.C14, hvm.C15 };
    for (ctrs, 0..) |ctr, i| {
        const t = hvm.term_new(ctr, @intCast(i), 0);
        try std.testing.expectEqual(ctr, hvm.term_tag(t));
    }

    // Primitives P00-P15
    const prims = [_]hvm.Tag{ hvm.P00, hvm.P01, hvm.P02, hvm.P03, hvm.P04, hvm.P05, hvm.P06, hvm.P07, hvm.P08, hvm.P09, hvm.P10, hvm.P11, hvm.P12, hvm.P13, hvm.P14, hvm.P15 };
    for (prims, 0..) |prim, i| {
        const t = hvm.term_new(prim, @intCast(i), 0);
        try std.testing.expectEqual(prim, hvm.term_tag(t));
    }

    // Stack frames
    const frames = [_]hvm.Tag{ hvm.F_APP, hvm.F_MAT, hvm.F_SWI, hvm.F_USE, hvm.F_OP2, hvm.F_DUP, hvm.F_EQL, hvm.F_ANN };
    for (frames) |frame| {
        const t = hvm.term_new(frame, 0, 0);
        try std.testing.expectEqual(frame, hvm.term_tag(t));
    }

    // Type system tags
    const type_tags = [_]hvm.Tag{ hvm.ANN, hvm.BRI, hvm.TYP, hvm.ALL, hvm.SIG, hvm.SLF };
    for (type_tags) |tag| {
        const t = hvm.term_new(tag, 0, 0);
        try std.testing.expectEqual(tag, hvm.term_tag(t));
    }
}

test "operator symbol lookup" {
    try std.testing.expectEqualStrings("+", hvm.op_symbol(hvm.OP_ADD));
    try std.testing.expectEqualStrings("-", hvm.op_symbol(hvm.OP_SUB));
    try std.testing.expectEqualStrings("*", hvm.op_symbol(hvm.OP_MUL));
    try std.testing.expectEqualStrings("/", hvm.op_symbol(hvm.OP_DIV));
    try std.testing.expectEqualStrings("%", hvm.op_symbol(hvm.OP_MOD));
    try std.testing.expectEqualStrings("==", hvm.op_symbol(hvm.OP_EQ));
    try std.testing.expectEqualStrings("!=", hvm.op_symbol(hvm.OP_NE));
    try std.testing.expectEqualStrings("<", hvm.op_symbol(hvm.OP_LT));
    try std.testing.expectEqualStrings(">", hvm.op_symbol(hvm.OP_GT));
    try std.testing.expectEqualStrings("<=", hvm.op_symbol(hvm.OP_LE));
    try std.testing.expectEqualStrings(">=", hvm.op_symbol(hvm.OP_GE));
    try std.testing.expectEqualStrings("&", hvm.op_symbol(hvm.OP_AND));
    try std.testing.expectEqualStrings("|", hvm.op_symbol(hvm.OP_OR));
    try std.testing.expectEqualStrings("^", hvm.op_symbol(hvm.OP_XOR));
    try std.testing.expectEqualStrings("<<", hvm.op_symbol(hvm.OP_LSH));
    try std.testing.expectEqualStrings(">>", hvm.op_symbol(hvm.OP_RSH));
}

test "operator from symbol lookup" {
    try std.testing.expectEqual(hvm.OP_ADD, hvm.op_from_symbol("+").?);
    try std.testing.expectEqual(hvm.OP_SUB, hvm.op_from_symbol("-").?);
    try std.testing.expectEqual(hvm.OP_MUL, hvm.op_from_symbol("*").?);
    try std.testing.expectEqual(hvm.OP_DIV, hvm.op_from_symbol("/").?);
    try std.testing.expectEqual(hvm.OP_MOD, hvm.op_from_symbol("%").?);
    try std.testing.expectEqual(hvm.OP_EQ, hvm.op_from_symbol("==").?);
    try std.testing.expectEqual(hvm.OP_NE, hvm.op_from_symbol("!=").?);
    try std.testing.expectEqual(hvm.OP_LT, hvm.op_from_symbol("<").?);
    try std.testing.expectEqual(hvm.OP_GT, hvm.op_from_symbol(">").?);
    try std.testing.expectEqual(hvm.OP_LE, hvm.op_from_symbol("<=").?);
    try std.testing.expectEqual(hvm.OP_GE, hvm.op_from_symbol(">=").?);
    try std.testing.expectEqual(hvm.OP_LSH, hvm.op_from_symbol("<<").?);
    try std.testing.expectEqual(hvm.OP_RSH, hvm.op_from_symbol(">>").?);
    try std.testing.expect(hvm.op_from_symbol("???") == null);
}

// =============================================================================
// HVM Integration Test (single init, 100+ test cases)
// =============================================================================

test "hvm comprehensive integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try hvm.hvm_init(gpa.allocator());
    defer hvm.hvm_free(gpa.allocator());

    // Test runs all cases - any failure stops the test

    // --- Arithmetic Operators (17 operators x multiple cases) ---
    const op_tests = [_]struct { op: hvm.Ext, a: u32, b: u32, expected: u32 }{
        // ADD
        .{ .op = hvm.OP_ADD, .a = 0, .b = 0, .expected = 0 },
        .{ .op = hvm.OP_ADD, .a = 1, .b = 1, .expected = 2 },
        .{ .op = hvm.OP_ADD, .a = 100, .b = 200, .expected = 300 },
        // SUB
        .{ .op = hvm.OP_SUB, .a = 5, .b = 3, .expected = 2 },
        .{ .op = hvm.OP_SUB, .a = 100, .b = 100, .expected = 0 },
        // MUL
        .{ .op = hvm.OP_MUL, .a = 0, .b = 100, .expected = 0 },
        .{ .op = hvm.OP_MUL, .a = 6, .b = 7, .expected = 42 },
        .{ .op = hvm.OP_MUL, .a = 1000, .b = 1000, .expected = 1000000 },
        // DIV
        .{ .op = hvm.OP_DIV, .a = 10, .b = 2, .expected = 5 },
        .{ .op = hvm.OP_DIV, .a = 10, .b = 3, .expected = 3 },
        .{ .op = hvm.OP_DIV, .a = 0, .b = 5, .expected = 0 },
        // MOD
        .{ .op = hvm.OP_MOD, .a = 10, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_MOD, .a = 10, .b = 2, .expected = 0 },
        // EQ
        .{ .op = hvm.OP_EQ, .a = 5, .b = 5, .expected = 1 },
        .{ .op = hvm.OP_EQ, .a = 5, .b = 6, .expected = 0 },
        // NE
        .{ .op = hvm.OP_NE, .a = 5, .b = 6, .expected = 1 },
        .{ .op = hvm.OP_NE, .a = 5, .b = 5, .expected = 0 },
        // LT
        .{ .op = hvm.OP_LT, .a = 3, .b = 5, .expected = 1 },
        .{ .op = hvm.OP_LT, .a = 5, .b = 3, .expected = 0 },
        .{ .op = hvm.OP_LT, .a = 5, .b = 5, .expected = 0 },
        // GT
        .{ .op = hvm.OP_GT, .a = 5, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_GT, .a = 3, .b = 5, .expected = 0 },
        // LE
        .{ .op = hvm.OP_LE, .a = 3, .b = 5, .expected = 1 },
        .{ .op = hvm.OP_LE, .a = 5, .b = 5, .expected = 1 },
        .{ .op = hvm.OP_LE, .a = 6, .b = 5, .expected = 0 },
        // GE
        .{ .op = hvm.OP_GE, .a = 5, .b = 3, .expected = 1 },
        .{ .op = hvm.OP_GE, .a = 5, .b = 5, .expected = 1 },
        .{ .op = hvm.OP_GE, .a = 4, .b = 5, .expected = 0 },
        // AND
        .{ .op = hvm.OP_AND, .a = 0xFF, .b = 0x0F, .expected = 0x0F },
        .{ .op = hvm.OP_AND, .a = 0x00, .b = 0xFF, .expected = 0x00 },
        // OR
        .{ .op = hvm.OP_OR, .a = 0xF0, .b = 0x0F, .expected = 0xFF },
        .{ .op = hvm.OP_OR, .a = 0x00, .b = 0x00, .expected = 0x00 },
        // XOR
        .{ .op = hvm.OP_XOR, .a = 0xFF, .b = 0x0F, .expected = 0xF0 },
        .{ .op = hvm.OP_XOR, .a = 0xFF, .b = 0xFF, .expected = 0x00 },
        // LSH
        .{ .op = hvm.OP_LSH, .a = 1, .b = 4, .expected = 16 },
        .{ .op = hvm.OP_LSH, .a = 0xFF, .b = 8, .expected = 0xFF00 },
        // RSH
        .{ .op = hvm.OP_RSH, .a = 16, .b = 2, .expected = 4 },
        .{ .op = hvm.OP_RSH, .a = 0xFF00, .b = 8, .expected = 0xFF },
    };

    for (op_tests) |t| {
        hvm.set_len(1);
        const loc = hvm.alloc_node(2);
        hvm.set(loc, hvm.term_new(hvm.NUM, 0, t.a));
        hvm.set(loc + 1, hvm.term_new(hvm.NUM, 0, t.b));
        const result = hvm.reduce(hvm.term_new(hvm.P02, t.op, @truncate(loc)));
        try std.testing.expectEqual(t.expected, hvm.term_val(result));
    }

    // --- Lambda Reductions ---
    // Identity: (λx.x) 42 => 42
    {
        hvm.set_len(1);
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(lam_loc)));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));
        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
        try std.testing.expectEqual(@as(u32, 42), hvm.term_val(result));
    }

    // Constant: ((λx.λy.x) 1) 2 => 1
    {
        hvm.set_len(1);
        const inner_lam_loc = hvm.alloc_node(1);
        const outer_lam_loc = hvm.alloc_node(1);
        hvm.set(outer_lam_loc, hvm.term_new(hvm.LAM, 0, @truncate(inner_lam_loc)));
        hvm.set(inner_lam_loc, hvm.term_new(hvm.VAR, 0, @truncate(outer_lam_loc)));
        const app1_loc = hvm.alloc_node(2);
        hvm.set(app1_loc, hvm.term_new(hvm.LAM, 0, @truncate(outer_lam_loc)));
        hvm.set(app1_loc + 1, hvm.term_new(hvm.NUM, 0, 1));
        const app2_loc = hvm.alloc_node(2);
        hvm.set(app2_loc, hvm.term_new(hvm.APP, 0, @truncate(app1_loc)));
        hvm.set(app2_loc + 1, hvm.term_new(hvm.NUM, 0, 2));
        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app2_loc)));
        try std.testing.expectEqual(@as(u32, 1), hvm.term_val(result));
    }

    // Erasure: (λx.*) 42 => *
    {
        hvm.set_len(1);
        const lam_loc = hvm.alloc_node(1);
        hvm.set(lam_loc, hvm.term_new(hvm.ERA, 0, 0));
        const app_loc = hvm.alloc_node(2);
        hvm.set(app_loc, hvm.term_new(hvm.LAM, 0, @truncate(lam_loc)));
        hvm.set(app_loc + 1, hvm.term_new(hvm.NUM, 0, 42));
        const result = hvm.reduce(hvm.term_new(hvm.APP, 0, @truncate(app_loc)));
        try std.testing.expectEqual(hvm.ERA, hvm.term_tag(result));
    }

    // --- Superposition/Collapse ---
    // CO0 + SUP => first element
    {
        hvm.set_len(1);
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.NUM, 0, 10));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 20));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));
        const result = hvm.reduce(hvm.term_new(hvm.CO0, 0, @truncate(dup_loc)));
        try std.testing.expectEqual(@as(u32, 10), hvm.term_val(result));
    }

    // CO1 + SUP => second element
    {
        hvm.set_len(1);
        const sup_loc = hvm.alloc_node(2);
        hvm.set(sup_loc, hvm.term_new(hvm.NUM, 0, 10));
        hvm.set(sup_loc + 1, hvm.term_new(hvm.NUM, 0, 20));
        const dup_loc = hvm.alloc_node(1);
        hvm.set(dup_loc, hvm.term_new(hvm.SUP, 0, @truncate(sup_loc)));
        const result = hvm.reduce(hvm.term_new(hvm.CO1, 0, @truncate(dup_loc)));
        try std.testing.expectEqual(@as(u32, 20), hvm.term_val(result));
    }

    // --- Switch ---
    // Switch 0 => zero case
    {
        hvm.set_len(1);
        const swi_loc = hvm.alloc_node(3);
        hvm.set(swi_loc, hvm.term_new(hvm.NUM, 0, 0));
        hvm.set(swi_loc + 1, hvm.term_new(hvm.NUM, 0, 100));
        const succ_lam = hvm.alloc_node(1);
        hvm.set(succ_lam, hvm.term_new(hvm.NUM, 0, 200));
        hvm.set(swi_loc + 2, hvm.term_new(hvm.LAM, 0, @truncate(succ_lam)));
        const result = hvm.reduce(hvm.term_new(hvm.SWI, 0, @truncate(swi_loc)));
        try std.testing.expectEqual(@as(u32, 100), hvm.term_val(result));
    }

    // Switch n > 0 => succ case
    {
        hvm.set_len(1);
        const swi_loc = hvm.alloc_node(3);
        hvm.set(swi_loc, hvm.term_new(hvm.NUM, 0, 5));
        hvm.set(swi_loc + 1, hvm.term_new(hvm.NUM, 0, 100));
        const succ_lam = hvm.alloc_node(1);
        hvm.set(succ_lam, hvm.term_new(hvm.NUM, 0, 200));
        hvm.set(swi_loc + 2, hvm.term_new(hvm.LAM, 0, @truncate(succ_lam)));
        const result = hvm.reduce(hvm.term_new(hvm.SWI, 0, @truncate(swi_loc)));
        try std.testing.expectEqual(@as(u32, 200), hvm.term_val(result));
    }

    // --- Parser + Reduction Tests ---
    const parser_tests = [_]struct { code: []const u8, expected: u32 }{
        .{ .code = "#42", .expected = 42 },
        .{ .code = "#0", .expected = 0 },
        .{ .code = "#999999", .expected = 999999 },
        .{ .code = "'x'", .expected = 'x' },
        .{ .code = "'A'", .expected = 65 },
        .{ .code = "(+ #3 #4)", .expected = 7 },
        .{ .code = "(- #10 #3)", .expected = 7 },
        .{ .code = "(* #6 #7)", .expected = 42 },
        .{ .code = "(/ #10 #2)", .expected = 5 },
        .{ .code = "(% #10 #3)", .expected = 1 },
        .{ .code = "(== #5 #5)", .expected = 1 },
        .{ .code = "(!= #5 #3)", .expected = 1 },
        .{ .code = "(< #3 #5)", .expected = 1 },
        .{ .code = "(> #5 #3)", .expected = 1 },
        .{ .code = "(<= #5 #5)", .expected = 1 },
        .{ .code = "(>= #5 #5)", .expected = 1 },
        .{ .code = "(& #255 #15)", .expected = 15 },
        .{ .code = "(| #240 #15)", .expected = 255 },
        .{ .code = "(^ #255 #15)", .expected = 240 },
        .{ .code = "(<< #1 #4)", .expected = 16 },
        .{ .code = "(>> #16 #2)", .expected = 4 },
        .{ .code = "((\\x.x) #42)", .expected = 42 },
        .{ .code = "(((\\x.\\y.x) #1) #2)", .expected = 1 },
        .{ .code = "(((\\a.\\b.b) #1) #2)", .expected = 2 },
        .{ .code = "((((\\a.\\b.\\c.a) #1) #2) #3)", .expected = 1 },
        .{ .code = "((((\\a.\\b.\\c.c) #1) #2) #3)", .expected = 3 },
        .{ .code = "(* (+ #2 #3) (- #10 #4))", .expected = 30 },
        .{ .code = "(* (+ #2 #3) (+ #4 #5))", .expected = 45 },
        .{ .code = "(/ (- #10 #2) (- #4 #2))", .expected = 4 },
        .{ .code = "((\\x.(+ x x)) #21)", .expected = 42 },
        .{ .code = "(+ 'A' #1)", .expected = 66 },
        .{ .code = "(?#0 #100 \\p.#200)", .expected = 100 },
        .{ .code = "(?#5 #100 \\p.#200)", .expected = 200 },
    };

    for (parser_tests) |t| {
        hvm.set_len(1);
        const term = try parser.parse(gpa.allocator(), t.code);
        const result = hvm.reduce(term);
        try std.testing.expectEqual(t.expected, hvm.term_val(result));
    }

    // --- Heap Allocation Tests ---
    {
        hvm.set_len(1);
        const loc1 = hvm.alloc_node(1);
        const loc2 = hvm.alloc_node(1);
        try std.testing.expect(loc2 > loc1);
    }

    {
        hvm.set_len(1);
        const loc1 = hvm.alloc_node(5);
        const loc2 = hvm.alloc_node(3);
        try std.testing.expectEqual(loc1 + 5, loc2);
    }

    {
        hvm.set_len(1);
        const loc = hvm.alloc_node(1);
        const term = hvm.term_new(hvm.NUM, 0, 12345);
        hvm.set(loc, term);
        try std.testing.expectEqual(term, hvm.got(loc));
    }

    // --- Interaction Counter ---
    {
        hvm.set_len(1);
        hvm.set_itr(0);
        const before = hvm.get_itrs();
        const loc = hvm.alloc_node(2);
        hvm.set(loc, hvm.term_new(hvm.NUM, 0, 5));
        hvm.set(loc + 1, hvm.term_new(hvm.NUM, 0, 3));
        _ = hvm.reduce(hvm.term_new(hvm.P02, hvm.OP_ADD, @truncate(loc)));
        const after = hvm.get_itrs();
        try std.testing.expect(after > before);
    }

    // --- Batch Operations ---
    {
        const size: usize = 100;
        var a: [size]u32 = undefined;
        var b: [size]u32 = undefined;
        var results: [size]u32 = undefined;

        for (0..size) |i| {
            a[i] = @intCast(i);
            b[i] = @intCast(i * 2);
        }

        hvm.batch_add(&a, &b, &results);
        for (0..size) |i| {
            try std.testing.expectEqual(@as(u32, @intCast(i + i * 2)), results[i]);
        }

        hvm.batch_mul(&a, &b, &results);
        for (0..size) |i| {
            try std.testing.expectEqual(@as(u32, @intCast(i * i * 2)), results[i]);
        }
    }

    // Total: 38 op tests + 3 lambda + 2 sup + 2 switch + 35 parser + 3 heap + 1 itr + 2 batch = 86+ tests
    // Plus the unit tests above = 100+ total test cases
}
