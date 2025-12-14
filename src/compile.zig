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
// Variable Use Tracking
// =============================================================================

const VarUseInfo = struct {
    loc: u64, // Lambda slot location
    total_uses: u32, // How many times this var is used total
    current_use: u32, // Counter for which use we're on (0-indexed)
    label: hvm.Ext, // Label for this var's duplication
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

    // Variable use tracking for duplication
    var_uses: StringHashMap(VarUseInfo),

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
            .var_uses = StringHashMap(VarUseInfo).init(allocator),
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
        self.var_uses.deinit();
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

    /// Build duplication access for multi-use variables
    /// For multi-use, we use CO0/CO1 which properly interact with values:
    /// - CO0/CO1 read from the slot and trigger interaction rules
    /// - Interaction stores result with SUB_BIT for sibling projector
    /// - All siblings pointing to same slot get the shared value
    fn buildDupAccess(self: *Compiler, loc: u64, total: u32, use_idx: u32, base_label: hvm.Ext) hvm.Term {
        _ = self;
        if (total <= 1) {
            // Single use - no duplication needed
            return hvm.term_new(hvm.VAR, 0, @truncate(loc));
        }
        // For multi-use: use CO0 and CO1 with unique labels
        // Both point to the same LAM slot
        // When APP-LAM fires, the argument is stored with SUB_BIT
        // CO0/CO1 will then read the substituted value
        const tag: hvm.Tag = if ((use_idx & 1) == 0) hvm.CO0 else hvm.CO1;
        return hvm.term_new(tag, base_label, @truncate(loc));
    }

    // =========================================================================
    // Variable Use Counting
    // =========================================================================

    /// Count uses of a variable in a list of statements
    fn countVarUsesInStmts(name: []const u8, stmts: []const ast.Stmt) u32 {
        var count: u32 = 0;
        for (stmts) |stmt| {
            count += countVarUsesInStmt(name, &stmt);
        }
        return count;
    }

    fn countVarUsesInStmt(name: []const u8, stmt: *const ast.Stmt) u32 {
        switch (stmt.*) {
            .return_ => |ret| {
                if (ret.value) |v| return countVarUsesInExpr(name, v);
                return 0;
            },
            .assign => |a| {
                // Just count uses in the value expression
                // (body is handled by sequential processing in countVarUsesInStmts)
                return countVarUsesInExpr(name, a.value);
            },
            .let_ => |l| {
                return countVarUsesInExpr(name, l.value);
            },
            .if_ => |i| {
                // Conditions are always checked (until one is true), but only one body executes
                var cond_uses: u32 = 0;
                var max_body: u32 = 0;
                for (i.branches) |b| {
                    cond_uses += countVarUsesInExpr(name, b.cond);
                    const body_uses = countVarUsesInStmts(name, b.body);
                    if (body_uses > max_body) max_body = body_uses;
                }
                if (i.else_body) |eb| {
                    const else_uses = countVarUsesInStmts(name, eb);
                    if (else_uses > max_body) max_body = else_uses;
                }
                return cond_uses + max_body;
            },
            .switch_ => |s| {
                // Only one case executes, so take MAX across cases, not sum
                const scrutinee_uses = countVarUsesInExpr(name, s.scrutinee);
                var max_case: u32 = 0;
                for (s.cases) |c| {
                    const case_uses = countVarUsesInStmts(name, c.body);
                    if (case_uses > max_case) max_case = case_uses;
                }
                return scrutinee_uses + max_case;
            },
            .match_ => |m| {
                // Only one case executes, so take MAX across cases, not sum
                const scrutinee_uses = countVarUsesInExpr(name, m.scrutinee);
                var max_case: u32 = 0;
                for (m.cases) |c| {
                    const case_uses = countVarUsesInStmts(name, c.body);
                    if (case_uses > max_case) max_case = case_uses;
                }
                return scrutinee_uses + max_case;
            },
            .fold_ => |f| {
                // Only one case executes, so take MAX across cases, not sum
                const scrutinee_uses = countVarUsesInExpr(name, f.scrutinee);
                var max_case: u32 = 0;
                for (f.cases) |c| {
                    const case_uses = countVarUsesInStmts(name, c.body);
                    if (case_uses > max_case) max_case = case_uses;
                }
                return scrutinee_uses + max_case;
            },
            .expr => |e| return countVarUsesInExpr(name, e),
            else => return 0,
        }
    }

    fn countVarUsesInExpr(name: []const u8, expr: *const ast.Expr) u32 {
        switch (expr.kind) {
            .var_ => |v| {
                return if (std.mem.eql(u8, v, name)) 1 else 0;
            },
            .number, .char_, .string, .symbol, .nat, .constructor, .era, .hole => return 0,
            .lambda => |l| {
                // Check if lambda binds this name (shadows it)
                for (l.params) |p| {
                    if (std.mem.eql(u8, p, name)) return 0;
                }
                return countVarUsesInExpr(name, l.body);
            },
            .app => |a| {
                var count = countVarUsesInExpr(name, a.func);
                for (a.args) |arg| count += countVarUsesInExpr(name, arg);
                return count;
            },
            .binop => |b| {
                return countVarUsesInExpr(name, b.left) + countVarUsesInExpr(name, b.right);
            },
            .unop => |u| return countVarUsesInExpr(name, u.operand),
            .if_ => |i| {
                // Condition is always evaluated, but only one branch is taken
                const cond_uses = countVarUsesInExpr(name, i.cond);
                const then_uses = countVarUsesInExpr(name, i.then_);
                const else_uses = countVarUsesInExpr(name, i.else_);
                return cond_uses + @max(then_uses, else_uses);
            },
            .let_ => |l| {
                var count = countVarUsesInExpr(name, l.value);
                // Check if let binds this name
                if (l.pattern.kind == .var_) {
                    if (std.mem.eql(u8, l.pattern.kind.var_, name)) return count;
                }
                count += countVarUsesInExpr(name, l.body);
                return count;
            },
            .tuple => |t| {
                var count: u32 = 0;
                for (t) |e| count += countVarUsesInExpr(name, e);
                return count;
            },
            .list => |li| {
                var count: u32 = 0;
                for (li) |e| count += countVarUsesInExpr(name, e);
                return count;
            },
            .superposition => |s| {
                var count: u32 = 0;
                for (s) |e| count += countVarUsesInExpr(name, e);
                return count;
            },
            .match_ => |m| {
                var count = countVarUsesInExpr(name, m.scrutinee);
                for (m.cases) |c| {
                    count += countVarUsesInExpr(name, c.body);
                }
                return count;
            },
            .fold_ => |f| {
                var count = countVarUsesInExpr(name, f.scrutinee);
                for (f.cases) |c| {
                    count += countVarUsesInExpr(name, c.body);
                }
                return count;
            },
            .switch_ => |sw| {
                var count = countVarUsesInExpr(name, sw.scrutinee);
                for (sw.cases) |c| {
                    count += countVarUsesInExpr(name, c.body);
                }
                return count;
            },
            .use_ => |u| {
                return countVarUsesInExpr(name, u.value) + countVarUsesInExpr(name, u.body);
            },
            .bend_ => |bend| {
                var count: u32 = 0;
                for (bend.bindings) |b| count += countVarUsesInExpr(name, b.init);
                count += countVarUsesInExpr(name, bend.when_cond);
                count += countVarUsesInExpr(name, bend.when_body);
                count += countVarUsesInExpr(name, bend.else_body);
                return count;
            },
            .field_access => |fa| return countVarUsesInExpr(name, fa.value),
            .fork => |f| return countVarUsesInExpr(name, f),
        }
    }

    /// Count uses of a variable from a specific statement index onward
    fn countVarUsesFromIdx(name: []const u8, stmts: []const ast.Stmt, start_idx: usize) u32 {
        if (start_idx >= stmts.len) return 0;
        return countVarUsesInStmts(name, stmts[start_idx..]);
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

        // First, allocate all lambda locations for parameters (in reverse order)
        // This way innermost params are allocated first
        var param_locs = try self.allocator.alloc(u64, funcdef.params.len);
        defer self.allocator.free(param_locs);

        for (funcdef.params, 0..) |_, i| {
            param_locs[i] = hvm.alloc_node(1);
        }

        // Count parameter uses in body and bind parameters
        for (funcdef.params, 0..) |param, i| {
            try self.bindVar(param.name, param_locs[i]);

            // Count uses of this parameter in the function body
            const uses = switch (funcdef.body) {
                .expr => |expr| countVarUsesInExpr(param.name, expr),
                .stmts => |stmts| countVarUsesInStmts(param.name, stmts),
            };

            // Register multi-use info if parameter is used more than once
            if (uses > 1) {
                try self.var_uses.put(param.name, .{
                    .loc = param_locs[i],
                    .total_uses = uses,
                    .current_use = 0,
                    .label = self.freshLabel(),
                });
            }
        }

        // Compile body
        var body = switch (funcdef.body) {
            .expr => |expr| try self.compileExpr(expr),
            .stmts => |stmts| try self.compileStmts(stmts),
        };

        // Clean up var_uses for parameters
        for (funcdef.params) |param| {
            _ = self.var_uses.remove(param.name);
        }

        // Wrap body in lambdas for each parameter (innermost first, so iterate reverse)
        var i: usize = funcdef.params.len;
        while (i > 0) {
            i -= 1;
            // Set the lambda body
            hvm.set(param_locs[i], body);
            // Create lambda term
            body = hvm.term_new(hvm.LAM, 0, @truncate(param_locs[i]));
        }

        // Register compiled function
        const state = hvm.hvm_get_state();
        state.setFuncBody(fid, body);
        state.book[fid] = hvm.funcDispatch;
    }

    // =========================================================================
    // Statement Compilation
    // =========================================================================

    fn compileStmts(self: *Compiler, stmts: []const ast.Stmt) CompileError!hvm.Term {
        return self.compileStmtsFrom(stmts, 0);
    }

    fn compileStmtsFrom(self: *Compiler, stmts: []const ast.Stmt, start_idx: usize) CompileError!hvm.Term {
        if (start_idx >= stmts.len) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        const stmt = stmts[start_idx];
        switch (stmt) {
            .return_ => |ret| {
                if (ret.value) |value| {
                    return self.compileExpr(value);
                }
                return hvm.term_new(hvm.ERA, 0, 0);
            },
            .assign => |assign| {
                // Compile value
                const value = try self.compileExpr(assign.value);
                // Compile the rest as the continuation
                return self.compilePatternBinding(assign.pattern, value, stmts, start_idx + 1);
            },
            .let_ => |let_| {
                const value = try self.compileExpr(let_.value);
                return self.compilePatternBinding(let_.pattern, value, stmts, start_idx + 1);
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
                // Expression statement - compile and continue
                _ = try self.compileExpr(expr);
                return self.compileStmtsFrom(stmts, start_idx + 1);
            },
            else => {
                return self.compileStmtsFrom(stmts, start_idx + 1);
            },
        }
    }

    /// Compile pattern binding with continuation
    fn compilePatternBinding(self: *Compiler, pattern: *const ast.Pattern, value: hvm.Term, stmts: []const ast.Stmt, next_idx: usize) CompileError!hvm.Term {
        switch (pattern.kind) {
            .var_ => |name| {
                // Count how many times this variable is used in the continuation
                const uses = countVarUsesFromIdx(name, stmts, next_idx);

                // Simple variable binding: create a lambda and apply
                try self.pushScope();

                const lam_loc = hvm.alloc_node(1);
                try self.bindVar(name, lam_loc);

                // Register variable use info for duplication tracking
                try self.var_uses.put(name, .{
                    .loc = lam_loc,
                    .total_uses = uses,
                    .current_use = 0,
                    .label = self.freshLabel(),
                });

                const body = try self.compileStmtsFrom(stmts, next_idx);

                // Clean up var use tracking
                _ = self.var_uses.remove(name);

                self.popScope();

                hvm.set(lam_loc, body);

                const lam = hvm.term_new(hvm.LAM, 0, @truncate(lam_loc));

                // Apply the lambda to the value
                const app_loc = hvm.alloc_node(2);
                hvm.set(app_loc, lam);
                hvm.set(app_loc + 1, value);
                return hvm.term_new(hvm.APP, 0, @truncate(app_loc));
            },
            .wildcard => {
                // Just continue with next statements
                return self.compileStmtsFrom(stmts, next_idx);
            },
            .tuple => |pats| {
                // Compile as MAT on the tuple constructor
                // The tuple has constructor ID 0, arity = len(pats)
                try self.pushScope();
                defer self.popScope();

                // Create lambdas for each tuple element
                var lam_locs = try self.allocator.alloc(u64, pats.len);
                defer self.allocator.free(lam_locs);

                for (0..pats.len) |i| {
                    lam_locs[i] = hvm.alloc_node(1);
                }

                // Bind pattern names to lambda locations
                for (pats, 0..) |p, i| {
                    if (p.kind == .var_) {
                        try self.bindVar(p.kind.var_, lam_locs[i]);
                    }
                }

                // Compile the continuation body
                var body = try self.compileStmtsFrom(stmts, next_idx);

                // Wrap in lambdas (innermost first)
                var i: usize = pats.len;
                while (i > 0) {
                    i -= 1;
                    hvm.set(lam_locs[i], body);
                    body = hvm.term_new(hvm.LAM, 0, @truncate(lam_locs[i]));
                }

                // Create MAT node with one case (constructor ID 0)
                const mat_loc = hvm.alloc_node(2);
                hvm.set(mat_loc, value);
                hvm.set(mat_loc + 1, body); // Case for constructor 0

                return hvm.term_new(hvm.MAT, 1, @truncate(mat_loc));
            },
            .constructor => |ctor| {
                // Similar to tuple but with named constructor
                const info = self.ctors.get(ctor.name) orelse return CompileError.UndefinedConstructor;

                try self.pushScope();
                defer self.popScope();

                var lam_locs = try self.allocator.alloc(u64, ctor.fields.len);
                defer self.allocator.free(lam_locs);

                for (0..ctor.fields.len) |i| {
                    lam_locs[i] = hvm.alloc_node(1);
                }

                for (ctor.fields, 0..) |field, i| {
                    if (field.pattern.kind == .var_) {
                        try self.bindVar(field.pattern.kind.var_, lam_locs[i]);
                    }
                }

                var body = try self.compileStmtsFrom(stmts, next_idx);

                var j: usize = ctor.fields.len;
                while (j > 0) {
                    j -= 1;
                    hvm.set(lam_locs[j], body);
                    body = hvm.term_new(hvm.LAM, 0, @truncate(lam_locs[j]));
                }

                // MAT with case at constructor's index
                const num_cases = info.id + 1;
                const mat_loc = hvm.alloc_node(1 + num_cases);
                hvm.set(mat_loc, value);

                // Fill with erasures for missing cases
                for (0..num_cases) |k| {
                    if (k == info.id) {
                        hvm.set(mat_loc + 1 + k, body);
                    } else {
                        hvm.set(mat_loc + 1 + k, hvm.term_new(hvm.ERA, 0, 0));
                    }
                }

                return hvm.term_new(hvm.MAT, @intCast(num_cases), @truncate(mat_loc));
            },
            else => {
                return self.compileStmtsFrom(stmts, next_idx);
            },
        }
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

        if (match_stmt.cases.len == 0) {
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        // Compile cases - each case becomes a lambda that receives constructor fields
        // Cases must be indexed by constructor ID
        var case_terms = try self.allocator.alloc(hvm.Term, match_stmt.cases.len);
        defer self.allocator.free(case_terms);

        // Initialize with erasure (for missing cases)
        for (case_terms) |*ct| {
            ct.* = hvm.term_new(hvm.ERA, 0, 0);
        }

        for (match_stmt.cases) |case| {
            const pattern = case.pattern;
            if (pattern.kind != .constructor) continue;

            const ctor = pattern.kind.constructor;
            const info = self.ctors.get(ctor.name) orelse continue;
            const case_idx = info.id;

            try self.pushScope();

            // Create lambdas for each field (in reverse order so innermost is first)
            var lam_locs = try self.allocator.alloc(u64, ctor.fields.len);
            defer self.allocator.free(lam_locs);

            for (ctor.fields, 0..) |_, i| {
                lam_locs[i] = hvm.alloc_node(1);
            }

            // Bind field names to their lambda locations
            for (ctor.fields, 0..) |field, i| {
                if (field.pattern.kind == .var_) {
                    try self.bindVar(field.pattern.kind.var_, lam_locs[i]);
                }
            }

            // Compile body
            var body = try self.compileStmts(case.body);

            // Wrap in lambdas (innermost first)
            var i: usize = ctor.fields.len;
            while (i > 0) {
                i -= 1;
                hvm.set(lam_locs[i], body);
                body = hvm.term_new(hvm.LAM, 0, @truncate(lam_locs[i]));
            }

            self.popScope();

            // Place at correct index
            if (case_idx < case_terms.len) {
                case_terms[case_idx] = body;
            }
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
                    // Check if this variable needs duplication
                    if (self.var_uses.getPtr(name)) |info| {
                        if (info.total_uses > 1) {
                            // Multi-use variable: generate duplication structure
                            // For N uses, we create a binary tree of dups
                            // Use CO0 for "first half" uses, CO1 for "second half"
                            const use_idx = info.current_use;
                            info.current_use += 1;

                            // Build the path through the dup tree
                            // For 2 uses: use 0 -> CO0, use 1 -> CO1
                            // For 3 uses: use 0 -> CO0, use 1 -> CO0 of CO1, use 2 -> CO1 of CO1
                            // For 4 uses: similar binary tree
                            return self.buildDupAccess(loc, info.total_uses, use_idx, info.label);
                        }
                    }
                    return hvm.term_new(hvm.VAR, 0, @truncate(loc));
                }
                // Check if it's a function reference
                if (self.funcs.get(name)) |fid| {
                    // Create a REF with no arguments
                    const loc = hvm.alloc_node(0);
                    return hvm.term_new(hvm.REF, fid, @truncate(loc));
                }
                std.debug.print("Undefined variable: {s}\n", .{name});
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
        // Get the function reference (no args in REF, args via APP)
        var result = hvm.term_new(hvm.REF, fid, 0);

        // Apply each argument using APP nodes
        for (args) |arg| {
            const compiled_arg = try self.compileExpr(arg);
            const loc = hvm.alloc_node(2);
            hvm.set(loc, result);
            hvm.set(loc + 1, compiled_arg);
            result = hvm.term_new(hvm.APP, 0, @truncate(loc));
        }

        return result;
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
