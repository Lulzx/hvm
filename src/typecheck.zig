const std = @import("std");
const ast = @import("ast.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// =============================================================================
// Hindley-Milner Type Checker for Bend
// =============================================================================
// Implements Algorithm W with exhaustiveness checking

// =============================================================================
// Type Representation
// =============================================================================

pub const TypeVar = u32;

pub const Type = struct {
    kind: TypeKind,

    pub const TypeKind = union(enum) {
        /// Type variable (for inference)
        var_: TypeVar,

        /// Primitive types
        primitive: Primitive,

        /// Named type constructor: List, Tree, Option
        constructor: struct {
            name: []const u8,
            args: []const *Type,
        },

        /// Function type: a -> b
        function: struct {
            param: *Type,
            ret: *Type,
        },

        /// Tuple type: (a, b, c)
        tuple: []const *Type,

        /// Forall type (polymorphic): forall a. T
        forall: struct {
            vars: []TypeVar,
            body: *Type,
        },
    };

    pub fn isVar(self: *const Type) bool {
        return self.kind == .var_;
    }

    pub fn getVar(self: *const Type) ?TypeVar {
        return switch (self.kind) {
            .var_ => |v| v,
            else => null,
        };
    }
};

pub const Primitive = enum {
    u24,
    i24,
    f24,
    bool_,
    string,
    unit,
};

/// Type scheme: forall a1 a2 ... an. T
pub const TypeScheme = struct {
    vars: []TypeVar,
    body: *Type,
};

// =============================================================================
// Type Environment
// =============================================================================

pub const TypeEnv = struct {
    bindings: StringHashMap(TypeScheme),
    parent: ?*TypeEnv,

    pub fn init(allocator: Allocator) TypeEnv {
        return .{
            .bindings = StringHashMap(TypeScheme).init(allocator),
            .parent = null,
        };
    }

    pub fn child(self: *TypeEnv, allocator: Allocator) TypeEnv {
        return .{
            .bindings = StringHashMap(TypeScheme).init(allocator),
            .parent = self,
        };
    }

    pub fn deinit(self: *TypeEnv) void {
        self.bindings.deinit();
    }

    pub fn bind(self: *TypeEnv, name: []const u8, scheme: TypeScheme) !void {
        try self.bindings.put(name, scheme);
    }

    pub fn lookup(self: *const TypeEnv, name: []const u8) ?TypeScheme {
        if (self.bindings.get(name)) |scheme| {
            return scheme;
        }
        if (self.parent) |p| {
            return p.lookup(name);
        }
        return null;
    }
};

// =============================================================================
// Constructor Type Info
// =============================================================================

pub const ConstructorType = struct {
    name: []const u8,
    type_name: []const u8,
    field_types: []*Type,
    result_type: *Type,
};

// =============================================================================
// Type Checker
// =============================================================================

pub const TypeError = error{
    UnificationFail,
    OccursCheck,
    UnboundVariable,
    UnknownConstructor,
    NonExhaustivePattern,
    RedundantPattern,
    TypeMismatch,
    ArityMismatch,
    OutOfMemory,
};

pub const TypeChecker = struct {
    allocator: Allocator,

    // Type variable counter
    next_var: TypeVar,

    // Type environments
    env: TypeEnv,

    // Type definitions
    type_defs: StringHashMap(TypeDefInfo),

    // Constructor types
    ctor_types: StringHashMap(ConstructorType),

    // Substitution (type variable -> type)
    subst: std.AutoHashMap(TypeVar, *Type),

    // Errors
    errors: ArrayList(TypeCheckError),

    pub const TypeDefInfo = struct {
        name: []const u8,
        params: []const []const u8,
        constructors: []const []const u8,
    };

    pub const TypeCheckError = struct {
        message: []const u8,
        span: ast.Span,
        expected: ?*Type,
        actual: ?*Type,
    };

    pub fn init(allocator: Allocator) TypeChecker {
        var tc = TypeChecker{
            .allocator = allocator,
            .next_var = 0,
            .env = TypeEnv.init(allocator),
            .type_defs = StringHashMap(TypeDefInfo).init(allocator),
            .ctor_types = StringHashMap(ConstructorType).init(allocator),
            .subst = std.AutoHashMap(TypeVar, *Type).init(allocator),
            .errors = .empty,
        };

        // Add built-in types
        tc.addBuiltins() catch {};

        return tc;
    }

    pub fn deinit(self: *TypeChecker) void {
        self.env.deinit();
        self.type_defs.deinit();
        self.ctor_types.deinit();
        self.subst.deinit();
        self.errors.deinit(self.allocator);
    }

    fn addBuiltins(self: *TypeChecker) !void {
        // Add built-in constructors for Bool
        try self.type_defs.put("Bool", .{
            .name = "Bool",
            .params = &[_][]const u8{},
            .constructors = &[_][]const u8{ "True", "False" },
        });
    }

    // =========================================================================
    // Type Variable Management
    // =========================================================================

    fn freshVar(self: *TypeChecker) !*Type {
        const v = self.next_var;
        self.next_var += 1;

        const ty = try self.allocator.create(Type);
        ty.* = .{ .kind = .{ .var_ = v } };
        return ty;
    }

    // =========================================================================
    // Type Construction
    // =========================================================================

    fn makePrimitive(self: *TypeChecker, prim: Primitive) !*Type {
        const ty = try self.allocator.create(Type);
        ty.* = .{ .kind = .{ .primitive = prim } };
        return ty;
    }

    fn makeFunction(self: *TypeChecker, param: *Type, ret: *Type) !*Type {
        const ty = try self.allocator.create(Type);
        ty.* = .{ .kind = .{ .function = .{ .param = param, .ret = ret } } };
        return ty;
    }

    fn makeConstructor(self: *TypeChecker, name: []const u8, args: []const *Type) !*Type {
        const ty = try self.allocator.create(Type);
        ty.* = .{ .kind = .{ .constructor = .{ .name = name, .args = args } } };
        return ty;
    }

    fn makeTuple(self: *TypeChecker, elems: []const *Type) !*Type {
        const ty = try self.allocator.create(Type);
        ty.* = .{ .kind = .{ .tuple = elems } };
        return ty;
    }

    // =========================================================================
    // Substitution and Occurs Check
    // =========================================================================

    fn apply(self: *TypeChecker, ty: *Type) *Type {
        switch (ty.kind) {
            .var_ => |v| {
                if (self.subst.get(v)) |replacement| {
                    return self.apply(replacement);
                }
                return ty;
            },
            .function => |f| {
                const param = self.apply(f.param);
                const ret = self.apply(f.ret);
                if (param == f.param and ret == f.ret) return ty;
                const new_ty = self.allocator.create(Type) catch return ty;
                new_ty.* = .{ .kind = .{ .function = .{ .param = param, .ret = ret } } };
                return new_ty;
            },
            .constructor => |c| {
                var changed = false;
                var new_args = self.allocator.alloc(*Type, c.args.len) catch return ty;
                for (c.args, 0..) |arg, i| {
                    new_args[i] = self.apply(arg);
                    if (new_args[i] != arg) changed = true;
                }
                if (!changed) return ty;
                const new_ty = self.allocator.create(Type) catch return ty;
                new_ty.* = .{ .kind = .{ .constructor = .{ .name = c.name, .args = new_args } } };
                return new_ty;
            },
            .tuple => |elems| {
                var changed = false;
                var new_elems = self.allocator.alloc(*Type, elems.len) catch return ty;
                for (elems, 0..) |elem, i| {
                    new_elems[i] = self.apply(elem);
                    if (new_elems[i] != elem) changed = true;
                }
                if (!changed) return ty;
                const new_ty = self.allocator.create(Type) catch return ty;
                new_ty.* = .{ .kind = .{ .tuple = new_elems } };
                return new_ty;
            },
            else => return ty,
        }
    }

    fn occurs(self: *TypeChecker, v: TypeVar, ty: *Type) bool {
        switch (ty.kind) {
            .var_ => |v2| return v == v2,
            .function => |f| return self.occurs(v, f.param) or self.occurs(v, f.ret),
            .constructor => |c| {
                for (c.args) |arg| {
                    if (self.occurs(v, arg)) return true;
                }
                return false;
            },
            .tuple => |elems| {
                for (elems) |elem| {
                    if (self.occurs(v, elem)) return true;
                }
                return false;
            },
            else => return false,
        }
    }

    // =========================================================================
    // Unification (Robinson's Algorithm)
    // =========================================================================

    fn unify(self: *TypeChecker, t1: *Type, t2: *Type) TypeError!void {
        const a = self.apply(t1);
        const b = self.apply(t2);

        // Same type
        if (a == b) return;

        // Variable cases
        if (a.kind == .var_) {
            const v = a.kind.var_;
            if (self.occurs(v, b)) return TypeError.OccursCheck;
            try self.subst.put(v, b);
            return;
        }

        if (b.kind == .var_) {
            const v = b.kind.var_;
            if (self.occurs(v, a)) return TypeError.OccursCheck;
            try self.subst.put(v, a);
            return;
        }

        // Primitive types
        if (a.kind == .primitive and b.kind == .primitive) {
            if (a.kind.primitive == b.kind.primitive) return;
            return TypeError.UnificationFail;
        }

        // Function types
        if (a.kind == .function and b.kind == .function) {
            try self.unify(a.kind.function.param, b.kind.function.param);
            try self.unify(a.kind.function.ret, b.kind.function.ret);
            return;
        }

        // Constructor types
        if (a.kind == .constructor and b.kind == .constructor) {
            if (!std.mem.eql(u8, a.kind.constructor.name, b.kind.constructor.name)) {
                return TypeError.UnificationFail;
            }
            if (a.kind.constructor.args.len != b.kind.constructor.args.len) {
                return TypeError.ArityMismatch;
            }
            for (a.kind.constructor.args, b.kind.constructor.args) |arg_a, arg_b| {
                try self.unify(arg_a, arg_b);
            }
            return;
        }

        // Tuple types
        if (a.kind == .tuple and b.kind == .tuple) {
            if (a.kind.tuple.len != b.kind.tuple.len) {
                return TypeError.ArityMismatch;
            }
            for (a.kind.tuple, b.kind.tuple) |elem_a, elem_b| {
                try self.unify(elem_a, elem_b);
            }
            return;
        }

        return TypeError.UnificationFail;
    }

    // =========================================================================
    // Generalization and Instantiation
    // =========================================================================

    fn generalize(self: *TypeChecker, ty: *Type) !TypeScheme {
        const applied = self.apply(ty);
        var free_vars: ArrayList(TypeVar) = .empty;

        try self.collectFreeVars(applied, &free_vars);

        return .{
            .vars = try free_vars.toOwnedSlice(self.allocator),
            .body = applied,
        };
    }

    fn collectFreeVars(self: *TypeChecker, ty: *Type, vars: *ArrayList(TypeVar)) !void {
        switch (ty.kind) {
            .var_ => |v| {
                // Check if already in list
                for (vars.items) |existing| {
                    if (existing == v) return;
                }
                // Check if bound in substitution
                if (self.subst.contains(v)) return;
                try vars.append(self.allocator, v);
            },
            .function => |f| {
                try self.collectFreeVars(f.param, vars);
                try self.collectFreeVars(f.ret, vars);
            },
            .constructor => |c| {
                for (c.args) |arg| {
                    try self.collectFreeVars(arg, vars);
                }
            },
            .tuple => |elems| {
                for (elems) |elem| {
                    try self.collectFreeVars(elem, vars);
                }
            },
            else => {},
        }
    }

    fn instantiate(self: *TypeChecker, scheme: TypeScheme) !*Type {
        // Create fresh variables for each quantified variable
        var var_map = std.AutoHashMap(TypeVar, *Type).init(self.allocator);
        defer var_map.deinit();

        for (scheme.vars) |v| {
            const fresh = try self.freshVar();
            try var_map.put(v, fresh);
        }

        return self.substituteVars(scheme.body, &var_map);
    }

    fn substituteVars(self: *TypeChecker, ty: *Type, var_map: *std.AutoHashMap(TypeVar, *Type)) !*Type {
        switch (ty.kind) {
            .var_ => |v| {
                if (var_map.get(v)) |replacement| {
                    return replacement;
                }
                return ty;
            },
            .function => |f| {
                const param = try self.substituteVars(f.param, var_map);
                const ret = try self.substituteVars(f.ret, var_map);
                return self.makeFunction(param, ret);
            },
            .constructor => |c| {
                var args = try self.allocator.alloc(*Type, c.args.len);
                for (c.args, 0..) |arg, i| {
                    args[i] = try self.substituteVars(arg, var_map);
                }
                return self.makeConstructor(c.name, args);
            },
            .tuple => |elems| {
                var new_elems = try self.allocator.alloc(*Type, elems.len);
                for (elems, 0..) |elem, i| {
                    new_elems[i] = try self.substituteVars(elem, var_map);
                }
                return self.makeTuple(new_elems);
            },
            else => return ty,
        }
    }

    // =========================================================================
    // Type Inference (Algorithm W)
    // =========================================================================

    pub fn inferProgram(self: *TypeChecker, program: *const ast.Program) !void {
        // Register type definitions
        for (program.types.items) |typedef| {
            try self.registerTypeDef(&typedef);
        }

        // Register object definitions
        for (program.objects.items) |objdef| {
            try self.registerObjectDef(&objdef);
        }

        // Infer function types
        for (program.funcs.items) |funcdef| {
            const ty = try self.inferFuncDef(&funcdef);
            const scheme = try self.generalize(ty);
            try self.env.bind(funcdef.name, scheme);
        }

        // Infer main expression type
        if (program.main) |main_expr| {
            _ = try self.inferExpr(main_expr);
        }
    }

    fn registerTypeDef(self: *TypeChecker, typedef: *const ast.TypeDef) !void {
        var ctor_names = try self.allocator.alloc([]const u8, typedef.constructors.len);
        for (typedef.constructors, 0..) |ctor, i| {
            ctor_names[i] = ctor.name;

            // Register constructor type
            const full_name = try std.fmt.allocPrint(self.allocator, "{s}/{s}", .{ typedef.name, ctor.name });

            var field_types = try self.allocator.alloc(*Type, ctor.fields.len);
            for (ctor.fields, 0..) |_, j| {
                field_types[j] = try self.freshVar();
            }

            const result_type = try self.makeConstructor(typedef.name, &[_]*Type{});

            try self.ctor_types.put(full_name, .{
                .name = full_name,
                .type_name = typedef.name,
                .field_types = field_types,
                .result_type = result_type,
            });

            // Also register short name
            try self.ctor_types.put(ctor.name, .{
                .name = ctor.name,
                .type_name = typedef.name,
                .field_types = field_types,
                .result_type = result_type,
            });
        }

        try self.type_defs.put(typedef.name, .{
            .name = typedef.name,
            .params = typedef.params,
            .constructors = ctor_names,
        });
    }

    fn registerObjectDef(self: *TypeChecker, objdef: *const ast.ObjectDef) !void {
        var field_types = try self.allocator.alloc(*Type, objdef.fields.len);
        for (objdef.fields, 0..) |_, i| {
            field_types[i] = try self.freshVar();
        }

        const result_type = try self.makeConstructor(objdef.name, &[_]*Type{});

        try self.ctor_types.put(objdef.name, .{
            .name = objdef.name,
            .type_name = objdef.name,
            .field_types = field_types,
            .result_type = result_type,
        });

        try self.type_defs.put(objdef.name, .{
            .name = objdef.name,
            .params = objdef.params,
            .constructors = &[_][]const u8{objdef.name},
        });
    }

    fn inferFuncDef(self: *TypeChecker, funcdef: *const ast.FuncDef) !*Type {
        // Create fresh type variables for parameters
        var param_types = try self.allocator.alloc(*Type, funcdef.params.len);
        for (funcdef.params, 0..) |param, i| {
            param_types[i] = if (param.type_) |ty_expr|
                try self.typeExprToType(ty_expr)
            else
                try self.freshVar();

            // Bind parameter in environment
            try self.env.bind(param.name, .{
                .vars = &[_]TypeVar{},
                .body = param_types[i],
            });
        }

        // Infer body type
        const body_type = switch (funcdef.body) {
            .expr => |expr| try self.inferExpr(expr),
            .stmts => |stmts| try self.inferStmts(stmts),
        };

        // Check return type annotation if present
        if (funcdef.return_type) |ret_ty_expr| {
            const ret_ty = try self.typeExprToType(ret_ty_expr);
            try self.unify(body_type, ret_ty);
        }

        // Build function type
        var result_type = body_type;
        var i: usize = funcdef.params.len;
        while (i > 0) {
            i -= 1;
            result_type = try self.makeFunction(param_types[i], result_type);
        }

        return result_type;
    }

    fn inferStmts(self: *TypeChecker, stmts: []const ast.Stmt) (TypeError || Allocator.Error)!*Type {
        var result_type = try self.makePrimitive(.unit);

        for (stmts) |stmt| {
            switch (stmt) {
                .return_ => |ret| {
                    if (ret.value) |value| {
                        result_type = try self.inferExpr(value);
                    }
                },
                .assign => |assign| {
                    const value_type = try self.inferExpr(assign.value);
                    try self.bindPatternType(assign.pattern, value_type);
                },
                .let_ => |let_stmt| {
                    const value_type = try self.inferExpr(let_stmt.value);
                    try self.bindPatternType(let_stmt.pattern, value_type);
                },
                .if_ => |if_stmt| {
                    result_type = try self.inferIfStmt(&if_stmt);
                },
                .match_ => |match_stmt| {
                    result_type = try self.inferMatchStmt(&match_stmt);
                },
                .expr => |expr| {
                    _ = try self.inferExpr(expr);
                },
                else => {},
            }
        }

        return result_type;
    }

    fn inferIfStmt(self: *TypeChecker, if_stmt: *const ast.IfStmtPayload) (TypeError || Allocator.Error)!*Type {
        // Check condition is numeric (for truthiness)
        for (if_stmt.branches) |branch| {
            const cond_type = try self.inferExpr(branch.cond);
            try self.unify(cond_type, try self.makePrimitive(.u24));
        }

        // Infer branch types and unify
        const result_type = try self.freshVar();
        for (if_stmt.branches) |branch| {
            const branch_type = try self.inferStmts(branch.body);
            try self.unify(result_type, branch_type);
        }

        if (if_stmt.else_body) |else_body| {
            const else_type = try self.inferStmts(else_body);
            try self.unify(result_type, else_type);
        }

        return result_type;
    }

    fn inferMatchStmt(self: *TypeChecker, match_stmt: *const ast.MatchStmtPayload) (TypeError || Allocator.Error)!*Type {
        const scrutinee_type = try self.inferExpr(match_stmt.scrutinee);

        if (match_stmt.name) |name| {
            try self.env.bind(name, .{
                .vars = &[_]TypeVar{},
                .body = scrutinee_type,
            });
        }

        const result_type = try self.freshVar();

        for (match_stmt.cases) |case| {
            // Check pattern against scrutinee type
            try self.checkPattern(case.pattern, scrutinee_type);

            // Bind pattern variables
            try self.bindPatternType(case.pattern, scrutinee_type);

            // Infer body type
            const body_type = try self.inferStmts(case.body);
            try self.unify(result_type, body_type);
        }

        // Check exhaustiveness
        try self.checkExhaustive(scrutinee_type, match_stmt.cases);

        return result_type;
    }

    fn inferExpr(self: *TypeChecker, expr: *const ast.Expr) !*Type {
        switch (expr.kind) {
            .var_ => |name| {
                if (self.env.lookup(name)) |scheme| {
                    return self.instantiate(scheme);
                }
                try self.errors.append(self.allocator, .{
                    .message = "unbound variable",
                    .span = expr.span,
                    .expected = null,
                    .actual = null,
                });
                return TypeError.UnboundVariable;
            },

            .constructor => |name| {
                if (self.ctor_types.get(name)) |ctor_info| {
                    // Build constructor function type
                    var ty = ctor_info.result_type;
                    var i: usize = ctor_info.field_types.len;
                    while (i > 0) {
                        i -= 1;
                        ty = try self.makeFunction(ctor_info.field_types[i], ty);
                    }
                    return ty;
                }
                return TypeError.UnknownConstructor;
            },

            .number => |num| {
                return switch (num) {
                    .unsigned => try self.makePrimitive(.u24),
                    .signed => try self.makePrimitive(.i24),
                    .float => try self.makePrimitive(.f24),
                };
            },

            .char_ => {
                return try self.makePrimitive(.u24);
            },

            .string => {
                return try self.makePrimitive(.string);
            },

            .lambda => |lam| {
                var param_types = try self.allocator.alloc(*Type, lam.params.len);
                for (lam.params, 0..) |param, i| {
                    param_types[i] = try self.freshVar();
                    try self.env.bind(param, .{
                        .vars = &[_]TypeVar{},
                        .body = param_types[i],
                    });
                }

                const body_type = try self.inferExpr(lam.body);

                // Build curried function type
                var result_type = body_type;
                var i: usize = lam.params.len;
                while (i > 0) {
                    i -= 1;
                    result_type = try self.makeFunction(param_types[i], result_type);
                }

                return result_type;
            },

            .app => |app| {
                const func_type = try self.inferExpr(app.func);
                var result_type = func_type;

                for (app.args) |arg| {
                    const arg_type = try self.inferExpr(arg);
                    const ret_type = try self.freshVar();

                    try self.unify(result_type, try self.makeFunction(arg_type, ret_type));
                    result_type = ret_type;
                }

                return self.apply(result_type);
            },

            .binop => |binop| {
                const left_type = try self.inferExpr(binop.left);
                const right_type = try self.inferExpr(binop.right);

                // Most operators work on numbers
                const num_type = try self.makePrimitive(.u24);
                try self.unify(left_type, num_type);
                try self.unify(right_type, num_type);

                // Comparison operators return bool-like (0 or 1)
                return switch (binop.op) {
                    .eq, .ne, .lt, .le, .gt, .ge, .land, .lor => num_type,
                    else => num_type,
                };
            },

            .if_ => |if_expr| {
                const cond_type = try self.inferExpr(if_expr.cond);
                try self.unify(cond_type, try self.makePrimitive(.u24));

                const then_type = try self.inferExpr(if_expr.then_);
                const else_type = try self.inferExpr(if_expr.else_);

                try self.unify(then_type, else_type);
                return self.apply(then_type);
            },

            .let_ => |let_expr| {
                const value_type = try self.inferExpr(let_expr.value);

                // Generalize value type (let-polymorphism)
                const scheme = try self.generalize(value_type);
                try self.bindPatternScheme(let_expr.pattern, scheme);

                return self.inferExpr(let_expr.body);
            },

            .tuple => |elems| {
                var elem_types = try self.allocator.alloc(*Type, elems.len);
                for (elems, 0..) |elem, i| {
                    elem_types[i] = try self.inferExpr(elem);
                }
                return self.makeTuple(elem_types);
            },

            .list => |elems| {
                const elem_type = try self.freshVar();
                for (elems) |elem| {
                    const t = try self.inferExpr(elem);
                    try self.unify(elem_type, t);
                }
                return self.makeConstructor("List", &[_]*Type{self.apply(elem_type)});
            },

            .superposition => |elems| {
                // All elements must have the same type
                const elem_type = try self.freshVar();
                for (elems) |elem| {
                    const t = try self.inferExpr(elem);
                    try self.unify(elem_type, t);
                }
                return self.apply(elem_type);
            },

            .era => {
                return try self.freshVar();
            },

            .hole => {
                return try self.freshVar();
            },

            else => {
                return try self.freshVar();
            },
        }
    }

    // =========================================================================
    // Pattern Type Binding
    // =========================================================================

    fn bindPatternType(self: *TypeChecker, pattern: *const ast.Pattern, ty: *Type) !void {
        switch (pattern.kind) {
            .var_ => |name| {
                try self.env.bind(name, .{
                    .vars = &[_]TypeVar{},
                    .body = ty,
                });
            },
            .wildcard => {},
            .constructor => |ctor| {
                // Get constructor info and bind field types
                if (self.ctor_types.get(ctor.name)) |ctor_info| {
                    for (ctor.fields, 0..) |field, i| {
                        if (i < ctor_info.field_types.len) {
                            try self.bindPatternType(field.pattern, ctor_info.field_types[i]);
                        }
                    }
                }
            },
            .tuple => |pats| {
                if (ty.kind == .tuple) {
                    for (pats, 0..) |p, i| {
                        if (i < ty.kind.tuple.len) {
                            try self.bindPatternType(p, ty.kind.tuple[i]);
                        }
                    }
                }
            },
            else => {},
        }
    }

    fn bindPatternScheme(self: *TypeChecker, pattern: *const ast.Pattern, scheme: TypeScheme) !void {
        switch (pattern.kind) {
            .var_ => |name| {
                try self.env.bind(name, scheme);
            },
            else => {
                // For non-variable patterns, just bind with monomorphic type
                try self.bindPatternType(pattern, scheme.body);
            },
        }
    }

    fn checkPattern(self: *TypeChecker, pattern: *const ast.Pattern, ty: *Type) !void {
        switch (pattern.kind) {
            .constructor => |ctor| {
                if (self.ctor_types.get(ctor.name)) |ctor_info| {
                    try self.unify(ty, ctor_info.result_type);
                }
            },
            .number => {
                try self.unify(ty, try self.makePrimitive(.u24));
            },
            .tuple => |pats| {
                var elem_types = try self.allocator.alloc(*Type, pats.len);
                for (pats, 0..) |_, i| {
                    elem_types[i] = try self.freshVar();
                }
                try self.unify(ty, try self.makeTuple(elem_types));

                for (pats, 0..) |p, i| {
                    try self.checkPattern(p, elem_types[i]);
                }
            },
            else => {},
        }
    }

    // =========================================================================
    // Exhaustiveness Checking
    // =========================================================================

    fn checkExhaustive(self: *TypeChecker, scrutinee_type: *const Type, cases: []const ast.MatchCase) !void {
        // Get type name from scrutinee type
        const type_name = switch (scrutinee_type.kind) {
            .constructor => |c| c.name,
            else => return, // Can't check exhaustiveness for non-constructor types
        };

        // Get all constructors for this type
        const type_info = self.type_defs.get(type_name) orelse return;

        // Track which constructors are covered
        var covered = StringHashMap(void).init(self.allocator);
        defer covered.deinit();

        for (cases) |case| {
            if (case.pattern.kind == .constructor) {
                try covered.put(case.pattern.kind.constructor.name, {});
            } else if (case.pattern.kind == .wildcard or case.pattern.kind == .var_) {
                // Wildcard covers everything
                return;
            }
        }

        // Check for missing constructors
        for (type_info.constructors) |ctor_name| {
            if (!covered.contains(ctor_name)) {
                try self.errors.append(self.allocator, .{
                    .message = "non-exhaustive pattern match",
                    .span = cases[0].span,
                    .expected = null,
                    .actual = null,
                });
                return TypeError.NonExhaustivePattern;
            }
        }
    }

    // =========================================================================
    // Type Expression Conversion
    // =========================================================================

    fn typeExprToType(self: *TypeChecker, ty_expr: *const ast.TypeExpr) !*Type {
        switch (ty_expr.kind) {
            .var_ => |_| {
                // Type variable - create fresh
                return try self.freshVar();
            },
            .named => |name| {
                return try self.makeConstructor(name, &[_]*Type{});
            },
            .primitive => |prim| {
                return switch (prim) {
                    .u24 => try self.makePrimitive(.u24),
                    .i24 => try self.makePrimitive(.i24),
                    .f24 => try self.makePrimitive(.f24),
                };
            },
            .function => |f| {
                const param = try self.typeExprToType(f.param);
                const ret = try self.typeExprToType(f.ret);
                return try self.makeFunction(param, ret);
            },
            .app => |a| {
                var args = try self.allocator.alloc(*Type, a.args.len);
                for (a.args, 0..) |arg, i| {
                    args[i] = try self.typeExprToType(arg);
                }
                return try self.makeConstructor(a.func.kind.named, args);
            },
            .tuple => |elems| {
                var elem_types = try self.allocator.alloc(*Type, elems.len);
                for (elems, 0..) |elem, i| {
                    elem_types[i] = try self.typeExprToType(elem);
                }
                return try self.makeTuple(elem_types);
            },
            .hole, .any => {
                return try self.freshVar();
            },
            .none => {
                return try self.makePrimitive(.unit);
            },
        }
    }

    // =========================================================================
    // Pretty Printing
    // =========================================================================

    pub fn prettyType(self: *TypeChecker, ty: *const Type, writer: anytype) !void {
        const applied = self.apply(@constCast(ty));
        try self.prettyTypeInner(applied, writer);
    }

    fn prettyTypeInner(self: *TypeChecker, ty: *const Type, writer: anytype) !void {
        switch (ty.kind) {
            .var_ => |v| {
                try writer.print("t{d}", .{v});
            },
            .primitive => |p| {
                try writer.writeAll(switch (p) {
                    .u24 => "u24",
                    .i24 => "i24",
                    .f24 => "f24",
                    .bool_ => "Bool",
                    .string => "String",
                    .unit => "()",
                });
            },
            .constructor => |c| {
                try writer.writeAll(c.name);
                if (c.args.len > 0) {
                    try writer.writeByte('(');
                    for (c.args, 0..) |arg, i| {
                        if (i > 0) try writer.writeAll(", ");
                        try self.prettyTypeInner(arg, writer);
                    }
                    try writer.writeByte(')');
                }
            },
            .function => |f| {
                try self.prettyTypeInner(f.param, writer);
                try writer.writeAll(" -> ");
                try self.prettyTypeInner(f.ret, writer);
            },
            .tuple => |elems| {
                try writer.writeByte('(');
                for (elems, 0..) |elem, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try self.prettyTypeInner(elem, writer);
                }
                try writer.writeByte(')');
            },
            .forall => |f| {
                try writer.writeAll("forall ");
                for (f.vars, 0..) |v, i| {
                    if (i > 0) try writer.writeByte(' ');
                    try writer.print("t{d}", .{v});
                }
                try writer.writeAll(". ");
                try self.prettyTypeInner(f.body, writer);
            },
        }
    }
};

// =============================================================================
// Public API
// =============================================================================

pub fn typecheck(program: *const ast.Program, allocator: Allocator) !TypeChecker {
    var tc = TypeChecker.init(allocator);
    try tc.inferProgram(program);
    return tc;
}
