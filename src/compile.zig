const std = @import("std");
const ast = @import("ast.zig");
const hvm = @import("hvm.zig");
const parser = @import("parser.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// =============================================================================
// Bend to HVM4 Compiler
// =============================================================================
// Transforms Bend AST into HVM4 interaction net terms

pub const CompileError = error{
    UndefinedVariable,
    UndefinedConstructor,
    UndefinedFunction,
    InvalidPattern,
    TooManyConstructors,
    TooManyFields,
    OutOfMemory,
};

// =============================================================================
// Constructor Registry
// =============================================================================

pub const CtorInfo = struct {
    id: u16, // Constructor ID (ext field)
    arity: u8, // Number of fields
    type_name: []const u8, // Parent type name
    fields: []const []const u8, // Field names for access
    recursive_fields: []const bool, // Which fields are recursive (~)
};

// =============================================================================
// Compiler State
// =============================================================================

pub const Compiler = struct {
    allocator: Allocator,

    // Constructor registry
    ctors: StringHashMap(CtorInfo),
    type_ctors: StringHashMap(ArrayList([]const u8)), // Type name -> constructor names
    next_ctor_id: u16,

    // Function registry (using HVM4's book)
    funcs: StringHashMap(u16), // Function name -> FID
    book: *parser.Book,
    next_fid: u16,

    // Variable scope
    scope: StringHashMap(u64), // Variable name -> heap location
    scope_stack: ArrayList(StringHashMap(u64)),

    // Label counter for superpositions
    next_label: hvm.Ext,

    pub fn init(allocator: Allocator, book: *parser.Book) Compiler {
        return .{
            .allocator = allocator,
            .ctors = StringHashMap(CtorInfo).init(allocator),
            .type_ctors = StringHashMap(ArrayList([]const u8)).init(allocator),
            .next_ctor_id = 0,
            .funcs = StringHashMap(u16).init(allocator),
            .book = book,
            .next_fid = 0x10,
            .scope = StringHashMap(u64).init(allocator),
            .scope_stack = .empty,
            .next_label = 0,
        };
    }

    pub fn deinit(self: *Compiler) void {
        // Free allocated ctor info - track which IDs we've freed to avoid double-free
        // (same CtorInfo is stored under both full name and short name)
        var freed_ids = std.AutoHashMap(u16, void).init(self.allocator);
        defer freed_ids.deinit();

        var ctor_it = self.ctors.iterator();
        while (ctor_it.next()) |entry| {
            const id = entry.value_ptr.id;
            if (freed_ids.get(id) == null) {
                // Free the allocated field arrays only once per constructor
                self.allocator.free(entry.value_ptr.fields);
                self.allocator.free(entry.value_ptr.recursive_fields);
                freed_ids.put(id, {}) catch {};
            }
            // Free the allocated keys (full_name strings like "Type/Ctor")
            if (std.mem.indexOfScalar(u8, entry.key_ptr.*, '/') != null) {
                self.allocator.free(entry.key_ptr.*);
            }
        }
        self.ctors.deinit();

        var it = self.type_ctors.valueIterator();
        while (it.next()) |list| {
            // Note: full_name strings already freed above via ctors keys
            // Just deinit the ArrayList itself
            var l = list.*;
            l.deinit(self.allocator);
        }
        self.type_ctors.deinit();
        self.funcs.deinit();
        self.scope.deinit();
        for (self.scope_stack.items) |*s| {
            s.deinit();
        }
        self.scope_stack.deinit(self.allocator);
    }

    // =========================================================================
    // Scope Management
    // =========================================================================

    fn pushScope(self: *Compiler) !void {
        var new_scope = StringHashMap(u64).init(self.allocator);
        // Copy current scope
        var it = self.scope.iterator();
        while (it.next()) |entry| {
            try new_scope.put(entry.key_ptr.*, entry.value_ptr.*);
        }
        try self.scope_stack.append(self.allocator, self.scope);
        self.scope = new_scope;
    }

    fn popScope(self: *Compiler) void {
        self.scope.deinit();
        if (self.scope_stack.items.len > 0) {
            self.scope = self.scope_stack.pop().?;
        } else {
            self.scope = StringHashMap(u64).init(self.allocator);
        }
    }

    fn bindVar(self: *Compiler, name: []const u8, loc: u64) !void {
        try self.scope.put(name, loc);
    }

    fn lookupVar(self: *Compiler, name: []const u8) ?u64 {
        return self.scope.get(name);
    }

    fn freshLabel(self: *Compiler) hvm.Ext {
        const lab = self.next_label;
        self.next_label += 1;
        return lab;
    }

    // =========================================================================
    // Program Compilation
    // =========================================================================

    pub fn compileProgram(self: *Compiler, program: *const ast.Program) !?hvm.Term {
        // First pass: register all types and constructors
        for (program.types.items) |typedef| {
            try self.registerType(&typedef);
        }
        for (program.objects.items) |objdef| {
            try self.registerObject(&objdef);
        }

        // Second pass: register all functions
        for (program.funcs.items) |funcdef| {
            try self.registerFunction(&funcdef);
        }

        // Third pass: compile all functions
        for (program.funcs.items) |funcdef| {
            try self.compileFunction(&funcdef);
        }

        // Compile main expression if present
        if (program.main) |main_expr| {
            return try self.compileExpr(main_expr);
        }

        // Look for a function named "main" and create a call to it
        for (program.funcs.items) |funcdef| {
            if (std.mem.eql(u8, funcdef.name, "main")) {
                // Create a REF to the main function with no arguments
                if (self.funcs.get("main")) |fid| {
                    // main() with no args - just a REF with loc 0
                    return hvm.term_new(hvm.REF, fid, 0);
                }
            }
        }

        return null;
    }

    // =========================================================================
    // Type Registration
    // =========================================================================

    fn registerType(self: *Compiler, typedef: *const ast.TypeDef) !void {
        var ctor_list: ArrayList([]const u8) = .empty;

        for (typedef.constructors) |ctor| {
            // Create full constructor name: Type/Ctor
            const full_name = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ typedef.name, ctor.name });

            // Extract field info
            var field_names = try self.allocator.alloc([]const u8, ctor.fields.len);
            var recursive_fields = try self.allocator.alloc(bool, ctor.fields.len);
            for (ctor.fields, 0..) |field, i| {
                field_names[i] = field.name;
                recursive_fields[i] = field.is_recursive;
            }

            try self.ctors.put(full_name, .{
                .id = self.next_ctor_id,
                .arity = @intCast(ctor.fields.len),
                .type_name = typedef.name,
                .fields = field_names,
                .recursive_fields = recursive_fields,
            });

            // Also register short name
            try self.ctors.put(ctor.name, .{
                .id = self.next_ctor_id,
                .arity = @intCast(ctor.fields.len),
                .type_name = typedef.name,
                .fields = field_names,
                .recursive_fields = recursive_fields,
            });

            try ctor_list.append(self.allocator, full_name);
            self.next_ctor_id += 1;
        }

        try self.type_ctors.put(typedef.name, ctor_list);
    }

    fn registerObject(self: *Compiler, objdef: *const ast.ObjectDef) !void {
        // Objects are single-constructor types
        var field_names = try self.allocator.alloc([]const u8, objdef.fields.len);
        var recursive_fields = try self.allocator.alloc(bool, objdef.fields.len);
        for (objdef.fields, 0..) |field, i| {
            field_names[i] = field.name;
            recursive_fields[i] = field.is_recursive;
        }

        try self.ctors.put(objdef.name, .{
            .id = self.next_ctor_id,
            .arity = @intCast(objdef.fields.len),
            .type_name = objdef.name,
            .fields = field_names,
            .recursive_fields = recursive_fields,
        });

        self.next_ctor_id += 1;
    }

    // =========================================================================
    // Function Registration and Compilation
    // =========================================================================

    fn registerFunction(self: *Compiler, funcdef: *const ast.FuncDef) !void {
        const fid = self.next_fid;
        self.next_fid += 1;
        try self.funcs.put(funcdef.name, fid);

        // Store arity
        hvm.hvm_get_state().fun_arity[fid] = @intCast(funcdef.params.len);
    }

    fn compileFunction(self: *Compiler, funcdef: *const ast.FuncDef) !void {
        const fid = self.funcs.get(funcdef.name) orelse return CompileError.UndefinedFunction;

        try self.pushScope();
        defer self.popScope();

        // Bind parameters
        for (funcdef.params, 0..) |param, i| {
            const loc = hvm.alloc_node(1);
            try self.bindVar(param.name, loc);
            _ = i;
        }

        // Compile body
        const body = switch (funcdef.body) {
            .expr => |expr| try self.compileExpr(expr),
            .stmts => |stmts| try self.compileStmts(stmts),
        };

        // Register compiled function
        const state = hvm.hvm_get_state();
        state.setFuncBody(fid, body);
        state.book[fid] = hvm.funcDispatch;
    }

    // =========================================================================
    // Statement Compilation
    // =========================================================================

    fn compileStmts(self: *Compiler, stmts: []const ast.Stmt) CompileError!hvm.Term {
        if (stmts.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        // Find return statement and compile accordingly
        for (stmts) |stmt| {
            switch (stmt) {
                .return_ => |ret| {
                    if (ret.value) |value| {
                        return self.compileExpr(value);
                    }
                    return hvm.term_new(hvm.ERA, 0, 0);
                },
                .assign => |assign| {
                    // Compile value and bind
                    const value = try self.compileExpr(assign.value);
                    try self.bindPattern(assign.pattern, value);
                },
                .let_ => |let_| {
                    const value = try self.compileExpr(let_.value);
                    try self.bindPattern(let_.pattern, value);
                },
                .if_ => |if_stmt| {
                    return self.compileIfStmt(&if_stmt);
                },
                .match_ => |match_stmt| {
                    return self.compileMatchStmt(&match_stmt);
                },
                .switch_ => |switch_stmt| {
                    return self.compileSwitchStmt(&switch_stmt);
                },
                .fold_ => |fold_stmt| {
                    return self.compileFoldStmt(&fold_stmt);
                },
                .expr => |expr| {
                    // Expression statement - just compile it
                    _ = try self.compileExpr(expr);
                },
                else => {},
            }
        }

        // No explicit return - return erasure
        return hvm.term_new(hvm.ERA, 0, 0);
    }

    fn compileIfStmt(self: *Compiler, if_stmt: *const ast.IfStmtPayload) CompileError!hvm.Term {
        if (if_stmt.branches.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        // Compile first branch
        const first = if_stmt.branches[0];
        const cond = try self.compileExpr(first.cond);
        const then_body = try self.compileStmts(first.body);

        // Compile else (or remaining branches)
        var else_body: hvm.Term = undefined;
        if (if_stmt.branches.len > 1) {
            // Remaining branches become nested if
            const remaining = ast.IfStmtPayload{
                .branches = if_stmt.branches[1..],
                .else_body = if_stmt.else_body,
                .span = if_stmt.span,
            };
            else_body = try self.compileIfStmt(&remaining);
        } else if (if_stmt.else_body) |eb| {
            else_body = try self.compileStmts(eb);
        } else {
            else_body = hvm.term_new(hvm.ERA, 0, 0);
        }

        // Compile as numeric switch: (?cond then_body else_body)
        // In Bend, 0 is falsy
        const loc = hvm.alloc_node(3);
        hvm.set(loc, cond);
        hvm.set(loc + 1, else_body); // case 0 (false)
        hvm.set(loc + 2, wrapInLambda(then_body)); // case _ (true)

        return hvm.term_new(hvm.SWI, 2, @truncate(loc));
    }

    fn compileMatchStmt(self: *Compiler, match_stmt: *const ast.MatchStmtPayload) CompileError!hvm.Term {
        const scrutinee = try self.compileExpr(match_stmt.scrutinee);

        // Bind scrutinee if named
        if (match_stmt.name) |name| {
            try self.bindVar(name, hvm.term_val(scrutinee));
        }

        // Get the type from the first pattern
        if (match_stmt.cases.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        // Compile cases
        var case_terms = try self.allocator.alloc(hvm.Term, match_stmt.cases.len);
        for (match_stmt.cases, 0..) |case, i| {
            try self.pushScope();
            try self.bindPattern(case.pattern, scrutinee);
            case_terms[i] = try self.compileStmts(case.body);
            self.popScope();
        }

        // Create MAT node
        const loc = hvm.alloc_node(1 + case_terms.len);
        hvm.set(loc, scrutinee);
        for (case_terms, 0..) |ct, i| {
            hvm.set(loc + 1 + i, ct);
        }

        return hvm.term_new(hvm.MAT, @intCast(case_terms.len), @truncate(loc));
    }

    fn compileSwitchStmt(self: *Compiler, switch_stmt: *const ast.SwitchStmtPayload) CompileError!hvm.Term {
        const scrutinee = try self.compileExpr(switch_stmt.scrutinee);

        if (switch_stmt.name) |name| {
            try self.bindVar(name, hvm.term_val(scrutinee));
        }

        if (switch_stmt.cases.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        // Find zero case and successor case
        var zero_case: hvm.Term = hvm.term_new(hvm.ERA, 0, 0);
        var succ_case: hvm.Term = hvm.term_new(hvm.ERA, 0, 0);

        for (switch_stmt.cases) |case| {
            const body = try self.compileStmts(case.body);
            if (case.value) |v| {
                if (v == 0) {
                    zero_case = body;
                }
            } else {
                // Wildcard - successor case
                succ_case = wrapInLambda(body);
            }
        }

        // Create SWI node
        const loc = hvm.alloc_node(3);
        hvm.set(loc, scrutinee);
        hvm.set(loc + 1, zero_case);
        hvm.set(loc + 2, succ_case);

        return hvm.term_new(hvm.SWI, 2, @truncate(loc));
    }

    fn compileFoldStmt(self: *Compiler, fold_stmt: *const ast.FoldStmtPayload) CompileError!hvm.Term {
        const scrutinee = try self.compileExpr(fold_stmt.scrutinee);

        if (fold_stmt.name) |name| {
            try self.bindVar(name, hvm.term_val(scrutinee));
        }

        // Fold is like match, but recursive fields automatically become recursive calls
        // For now, compile similarly to match
        return self.compileMatchStmt(&.{
            .name = fold_stmt.name,
            .scrutinee = fold_stmt.scrutinee,
            .cases = fold_stmt.cases,
            .span = fold_stmt.span,
        });
    }

    // =========================================================================
    // Expression Compilation
    // =========================================================================

    fn compileExpr(self: *Compiler, expr: *const ast.Expr) CompileError!hvm.Term {
        switch (expr.kind) {
            .var_ => |name| {
                if (self.lookupVar(name)) |loc| {
                    return hvm.term_new(hvm.VAR, 0, @truncate(loc));
                }
                // Check if it's a function reference
                if (self.funcs.get(name)) |fid| {
                    // Create a REF with no arguments
                    const loc = hvm.alloc_node(0);
                    return hvm.term_new(hvm.REF, fid, @truncate(loc));
                }
                return CompileError.UndefinedVariable;
            },

            .constructor => |name| {
                return self.compileConstructor(name, &[_]*ast.Expr{});
            },

            .number => |num| {
                const val: u32 = switch (num) {
                    .unsigned => |u| u,
                    .signed => |s| @bitCast(s),
                    .float => |f| @bitCast(f),
                };
                return hvm.term_new(hvm.NUM, 0, val);
            },

            .char_ => |c| {
                return hvm.term_new(hvm.NUM, 0, c);
            },

            .string => |s| {
                // Compile string as linked list of characters
                return self.compileString(s);
            },

            .lambda => |lam| {
                return self.compileLambda(lam.params, lam.body);
            },

            .app => |app| {
                // Check if it's a constructor application
                if (app.func.kind == .constructor) {
                    return self.compileConstructor(app.func.kind.constructor, app.args);
                }
                // Check if it's a function call
                if (app.func.kind == .var_) {
                    if (self.funcs.get(app.func.kind.var_)) |fid| {
                        return self.compileFunctionCall(fid, app.args);
                    }
                }
                // Regular application
                return self.compileApp(app.func, app.args);
            },

            .binop => |binop| {
                return self.compileBinop(binop.op, binop.left, binop.right);
            },

            .unop => |unop| {
                return self.compileUnop(unop.op, unop.operand);
            },

            .if_ => |if_expr| {
                return self.compileIfExpr(if_expr.cond, if_expr.then_, if_expr.else_);
            },

            .let_ => |let_expr| {
                try self.pushScope();
                defer self.popScope();
                const value = try self.compileExpr(let_expr.value);
                try self.bindPattern(let_expr.pattern, value);
                return self.compileExpr(let_expr.body);
            },

            .match_ => |match_expr| {
                return self.compileMatchExpr(&match_expr);
            },

            .fold_ => |fold_expr| {
                return self.compileFoldExpr(&fold_expr);
            },

            .superposition => |elems| {
                return self.compileSuperposition(elems);
            },

            .tuple => |elems| {
                return self.compileTuple(elems);
            },

            .list => |elems| {
                return self.compileList(elems);
            },

            .field_access => |access| {
                return self.compileFieldAccess(access.value, access.field);
            },

            .fork => |arg| {
                // Fork creates a superposition for parallel evaluation
                const compiled = try self.compileExpr(arg);
                const lab = self.freshLabel();
                const loc = hvm.alloc_node(2);
                hvm.set(loc, compiled);
                hvm.set(loc + 1, compiled); // Duplicate for parallel
                return hvm.term_new(hvm.SUP, lab, @truncate(loc));
            },

            .era => {
                return hvm.term_new(hvm.ERA, 0, 0);
            },

            .hole => {
                return hvm.term_new(hvm.ERA, 0, 0);
            },

            else => {
                return hvm.term_new(hvm.ERA, 0, 0);
            },
        }
    }

    fn compileConstructor(self: *Compiler, name: []const u8, args: []const *ast.Expr) CompileError!hvm.Term {
        const info = self.ctors.get(name) orelse {
            // Unknown constructor - create with arity from args
            const arity = args.len;
            const tag: hvm.Tag = @truncate(hvm.C00 + @min(arity, 15));
            const loc = hvm.alloc_node(arity);
            for (args, 0..) |arg, i| {
                const compiled = try self.compileExpr(arg);
                hvm.set(loc + i, compiled);
            }
            return hvm.term_new(tag, 0, @truncate(loc));
        };

        // Use registered constructor
        const tag: hvm.Tag = @truncate(hvm.C00 + @min(info.arity, 15));
        const loc = hvm.alloc_node(info.arity);
        for (args, 0..) |arg, i| {
            const compiled = try self.compileExpr(arg);
            hvm.set(loc + i, compiled);
        }

        return hvm.term_new(tag, info.id, @truncate(loc));
    }

    fn compileLambda(self: *Compiler, params: []const []const u8, body: *const ast.Expr) CompileError!hvm.Term {
        if (params.len == 0) {
            return self.compileExpr(body);
        }

        try self.pushScope();
        defer self.popScope();

        // Bind first parameter
        const lam_loc = hvm.alloc_node(1);
        try self.bindVar(params[0], lam_loc);

        // Compile body (with remaining params as nested lambdas)
        const inner = if (params.len > 1)
            try self.compileLambda(params[1..], body)
        else
            try self.compileExpr(body);

        hvm.set(lam_loc, inner);
        return hvm.term_new(hvm.LAM, 0, @truncate(lam_loc));
    }

    fn compileFunctionCall(self: *Compiler, fid: u16, args: []const *ast.Expr) CompileError!hvm.Term {
        const loc = hvm.alloc_node(args.len);
        for (args, 0..) |arg, i| {
            const compiled = try self.compileExpr(arg);
            hvm.set(loc + i, compiled);
        }
        return hvm.term_new(hvm.REF, fid, @truncate(loc));
    }

    fn compileApp(self: *Compiler, func: *const ast.Expr, args: []const *ast.Expr) CompileError!hvm.Term {
        var result = try self.compileExpr(func);

        for (args) |arg| {
            const compiled_arg = try self.compileExpr(arg);
            const loc = hvm.alloc_node(2);
            hvm.set(loc, result);
            hvm.set(loc + 1, compiled_arg);
            result = hvm.term_new(hvm.APP, 0, @truncate(loc));
        }

        return result;
    }

    fn compileBinop(self: *Compiler, op: ast.BinOp, left: *const ast.Expr, right: *const ast.Expr) CompileError!hvm.Term {
        const left_term = try self.compileExpr(left);
        const right_term = try self.compileExpr(right);

        const hvm_op: hvm.Ext = switch (op) {
            .add => hvm.OP_ADD,
            .sub => hvm.OP_SUB,
            .mul => hvm.OP_MUL,
            .div => hvm.OP_DIV,
            .mod => hvm.OP_MOD,
            .eq => hvm.OP_EQ,
            .ne => hvm.OP_NE,
            .lt => hvm.OP_LT,
            .le => hvm.OP_LE,
            .gt => hvm.OP_GT,
            .ge => hvm.OP_GE,
            .band => hvm.OP_AND,
            .bor => hvm.OP_OR,
            .bxor => hvm.OP_XOR,
            .shl => hvm.OP_LSH,
            .shr => hvm.OP_RSH,
            .land => hvm.OP_AND, // Logical AND as bitwise
            .lor => hvm.OP_OR, // Logical OR as bitwise
            .pow => hvm.OP_MUL, // No power op, approximate
        };

        const loc = hvm.alloc_node(2);
        hvm.set(loc, left_term);
        hvm.set(loc + 1, right_term);

        return hvm.term_new(hvm.P02, hvm_op, @truncate(loc));
    }

    fn compileUnop(self: *Compiler, op: ast.UnOp, operand: *const ast.Expr) CompileError!hvm.Term {
        const operand_term = try self.compileExpr(operand);

        switch (op) {
            .neg => {
                // -x = 0 - x
                const zero = hvm.term_new(hvm.NUM, 0, 0);
                const loc = hvm.alloc_node(2);
                hvm.set(loc, zero);
                hvm.set(loc + 1, operand_term);
                return hvm.term_new(hvm.P02, hvm.OP_SUB, @truncate(loc));
            },
            .not => {
                // !x = x == 0 (boolean not)
                const zero = hvm.term_new(hvm.NUM, 0, 0);
                const loc = hvm.alloc_node(2);
                hvm.set(loc, operand_term);
                hvm.set(loc + 1, zero);
                return hvm.term_new(hvm.P02, hvm.OP_EQ, @truncate(loc));
            },
            .bnot => {
                // ~x = bitwise not via XOR with all 1s
                const ones = hvm.term_new(hvm.NUM, 0, 0xFFFFFFFF);
                const loc = hvm.alloc_node(2);
                hvm.set(loc, operand_term);
                hvm.set(loc + 1, ones);
                return hvm.term_new(hvm.P02, hvm.OP_XOR, @truncate(loc));
            },
        }
    }

    fn compileIfExpr(self: *Compiler, cond: *const ast.Expr, then_: *const ast.Expr, else_: *const ast.Expr) CompileError!hvm.Term {
        const cond_term = try self.compileExpr(cond);
        const then_term = try self.compileExpr(then_);
        const else_term = try self.compileExpr(else_);

        // Use SWI: (?cond else then)
        // In Bend, 0 is falsy, non-zero is truthy
        const loc = hvm.alloc_node(3);
        hvm.set(loc, cond_term);
        hvm.set(loc + 1, else_term); // case 0 (false)
        hvm.set(loc + 2, wrapInLambda(then_term)); // case _ (true)

        return hvm.term_new(hvm.SWI, 2, @truncate(loc));
    }

    fn compileMatchExpr(self: *Compiler, match_expr: *const ast.MatchExprPayload) CompileError!hvm.Term {
        const scrutinee = try self.compileExpr(match_expr.scrutinee);

        if (match_expr.name) |name| {
            try self.bindVar(name, hvm.term_val(scrutinee));
        }

        if (match_expr.cases.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        var case_terms = try self.allocator.alloc(hvm.Term, match_expr.cases.len);
        for (match_expr.cases, 0..) |case, i| {
            try self.pushScope();
            try self.bindPattern(case.pattern, scrutinee);
            case_terms[i] = try self.compileExpr(case.body);
            self.popScope();
        }

        const loc = hvm.alloc_node(1 + case_terms.len);
        hvm.set(loc, scrutinee);
        for (case_terms, 0..) |ct, i| {
            hvm.set(loc + 1 + i, ct);
        }

        return hvm.term_new(hvm.MAT, @intCast(case_terms.len), @truncate(loc));
    }

    fn compileFoldExpr(self: *Compiler, fold_expr: *const ast.FoldExprPayload) CompileError!hvm.Term {
        // For now, compile like match
        return self.compileMatchExpr(&.{
            .name = fold_expr.name,
            .scrutinee = fold_expr.scrutinee,
            .cases = fold_expr.cases,
        });
    }

    fn compileSuperposition(self: *Compiler, elems: []const *ast.Expr) CompileError!hvm.Term {
        if (elems.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }
        if (elems.len == 1) {
            return self.compileExpr(elems[0]);
        }

        // Build nested superpositions: {a, {b, {c, d}}}
        var result = try self.compileExpr(elems[elems.len - 1]);
        var i: usize = elems.len - 1;
        while (i > 0) {
            i -= 1;
            const elem = try self.compileExpr(elems[i]);
            const lab = self.freshLabel();
            const loc = hvm.alloc_node(2);
            hvm.set(loc, elem);
            hvm.set(loc + 1, result);
            result = hvm.term_new(hvm.SUP, lab, @truncate(loc));
        }

        return result;
    }

    fn compileTuple(self: *Compiler, elems: []const *ast.Expr) CompileError!hvm.Term {
        // Compile tuple as constructor
        const tag: hvm.Tag = @truncate(hvm.C00 + @min(elems.len, 15));
        const loc = hvm.alloc_node(elems.len);
        for (elems, 0..) |elem, i| {
            const compiled = try self.compileExpr(elem);
            hvm.set(loc + i, compiled);
        }
        return hvm.term_new(tag, 0, @truncate(loc));
    }

    fn compileList(self: *Compiler, elems: []const *ast.Expr) CompileError!hvm.Term {
        // Build list from end: Cons(head, Cons(head, ... Nil))
        var result = self.compileConstructor("List/Nil", &[_]*ast.Expr{}) catch
            hvm.term_new(hvm.C00, 0, 0); // Nil

        var i: usize = elems.len;
        while (i > 0) {
            i -= 1;
            const head = try self.compileExpr(elems[i]);
            const loc = hvm.alloc_node(2);
            hvm.set(loc, head);
            hvm.set(loc + 1, result);
            result = hvm.term_new(hvm.C02, 1, @truncate(loc)); // Cons
        }

        return result;
    }

    fn compileString(self: *Compiler, s: []const u8) CompileError!hvm.Term {
        _ = self;
        // Compile string as list of characters
        var result = hvm.term_new(hvm.C00, 0, 0); // Nil

        var i: usize = s.len;
        while (i > 0) {
            i -= 1;
            const char_term = hvm.term_new(hvm.NUM, 0, s[i]);
            const loc = hvm.alloc_node(2);
            hvm.set(loc, char_term);
            hvm.set(loc + 1, result);
            result = hvm.term_new(hvm.C02, 1, @truncate(loc)); // Cons
        }

        return result;
    }

    fn compileFieldAccess(self: *Compiler, value: *const ast.Expr, field: []const u8) CompileError!hvm.Term {
        const compiled = try self.compileExpr(value);

        // Field access becomes a pattern match that extracts the field
        // For now, we need to know the constructor type
        // This is a simplified version - full implementation would need type info
        _ = field;

        return compiled; // Placeholder
    }

    // =========================================================================
    // Pattern Binding
    // =========================================================================

    fn bindPattern(self: *Compiler, pattern: *const ast.Pattern, value: hvm.Term) CompileError!void {
        switch (pattern.kind) {
            .var_ => |name| {
                try self.bindVar(name, hvm.term_val(value));
            },
            .wildcard => {},
            .constructor => |ctor| {
                // Bind each field
                const val_loc = hvm.term_val(value);
                for (ctor.fields, 0..) |field, i| {
                    const field_loc = val_loc + i;
                    try self.bindPattern(field.pattern, hvm.got(field_loc));
                }
            },
            .tuple => |pats| {
                const val_loc = hvm.term_val(value);
                for (pats, 0..) |p, i| {
                    const elem_loc = val_loc + i;
                    try self.bindPattern(p, hvm.got(elem_loc));
                }
            },
            else => {},
        }
    }
};

// =============================================================================
// Helpers
// =============================================================================

fn wrapInLambda(term: hvm.Term) hvm.Term {
    const loc = hvm.alloc_node(1);
    hvm.set(loc, term);
    return hvm.term_new(hvm.LAM, 0, @truncate(loc));
}

// =============================================================================
// Public API
// =============================================================================

pub fn compile(program: *const ast.Program, allocator: Allocator) !?hvm.Term {
    var book = parser.Book.init(allocator);
    var compiler = Compiler.init(allocator, &book);
    defer compiler.deinit();

    return compiler.compileProgram(program);
}
