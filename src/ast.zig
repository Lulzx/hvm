const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// Bend Abstract Syntax Tree
// =============================================================================
// Supports both Imp (Python-like) and Fun (Haskell-like) syntaxes

// =============================================================================
// Source Location
// =============================================================================

pub const Span = struct {
    start: u32,
    end: u32,
    line: u32,
    col: u32,

    pub fn merge(a: Span, b: Span) Span {
        return .{
            .start = @min(a.start, b.start),
            .end = @max(a.end, b.end),
            .line = a.line,
            .col = a.col,
        };
    }
};

// =============================================================================
// Program (Top-Level)
// =============================================================================

pub const Program = struct {
    types: ArrayList(TypeDef),
    objects: ArrayList(ObjectDef),
    funcs: ArrayList(FuncDef),
    main: ?*Expr,
    allocator: Allocator,

    pub fn init(allocator: Allocator) Program {
        return .{
            .types = .empty,
            .objects = .empty,
            .funcs = .empty,
            .main = null,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Program) void {
        self.types.deinit(self.allocator);
        self.objects.deinit(self.allocator);
        self.funcs.deinit(self.allocator);
    }
};

// =============================================================================
// Type Definitions
// =============================================================================

/// Algebraic data type definition
/// type List(T):
///   Nil
///   Cons { head: T, ~tail: List(T) }
pub const TypeDef = struct {
    name: []const u8,
    params: []const []const u8, // Type parameters (T, U, etc.)
    constructors: []Constructor,
    span: Span,
};

/// Constructor definition
/// Nil
/// Cons { head: T, ~tail: List(T) }
pub const Constructor = struct {
    name: []const u8,
    fields: []Field,
    span: Span,
};

/// Field in a constructor
/// head: T
/// ~tail: List(T)  (recursive field marked with ~)
pub const Field = struct {
    name: []const u8,
    type_: ?*TypeExpr,
    is_recursive: bool, // marked with ~
    span: Span,
};

/// Single-constructor type (record/struct)
/// object Point { x, y }
/// object Pair(A, B) { fst: A, snd: B }
pub const ObjectDef = struct {
    name: []const u8,
    params: []const []const u8,
    fields: []Field,
    span: Span,
};

// =============================================================================
// Function Definitions
// =============================================================================

/// Function definition (supports both styles)
/// Imp: def add(x: u24, y: u24) -> u24: return x + y
/// Fun: add(x, y) = x + y
pub const FuncDef = struct {
    name: []const u8,
    params: []Param,
    return_type: ?*TypeExpr,
    body: FuncBody,
    is_checked: bool, // checked/unchecked modifier
    span: Span,
};

/// Function parameter
/// x: u24
/// f: a -> b
pub const Param = struct {
    name: []const u8,
    type_: ?*TypeExpr,
    span: Span,
};

/// Function body - either expression (Fun) or statements (Imp)
pub const FuncBody = union(enum) {
    expr: *Expr,
    stmts: []Stmt,
};

// =============================================================================
// Statements (Imp Style)
// =============================================================================

/// Named struct type for if statement payload
pub const IfStmtPayload = struct {
    branches: []IfBranch,
    else_body: ?[]Stmt,
    span: Span,
};

/// Named struct type for match statement payload
pub const MatchStmtPayload = struct {
    name: ?[]const u8,
    scrutinee: *Expr,
    cases: []MatchCase,
    span: Span,
};

/// Named struct type for switch statement payload
pub const SwitchStmtPayload = struct {
    name: ?[]const u8,
    scrutinee: *Expr,
    cases: []SwitchCase,
    span: Span,
};

/// Named struct type for fold statement payload
pub const FoldStmtPayload = struct {
    name: ?[]const u8,
    scrutinee: *Expr,
    cases: []MatchCase,
    span: Span,
};

pub const Stmt = union(enum) {
    /// x = expr
    assign: struct {
        pattern: *Pattern,
        value: *Expr,
        span: Span,
    },

    /// x += expr, x -= expr, etc.
    assign_op: struct {
        target: []const u8,
        op: BinOp,
        value: *Expr,
        span: Span,
    },

    /// return expr
    return_: struct {
        value: ?*Expr,
        span: Span,
    },

    /// if cond: ... elif cond: ... else: ...
    if_: IfStmtPayload,

    /// switch x = scrutinee: case 0: ... case 1: ... case _: ...
    switch_: SwitchStmtPayload,

    /// match x = scrutinee: case Ctor: ...
    match_: MatchStmtPayload,

    /// fold x = scrutinee: case Ctor: ...
    fold_: FoldStmtPayload,

    /// bend x = init: when cond: ... else: ...
    bend_: struct {
        bindings: []BendBinding,
        when_cond: *Expr,
        when_body: []Stmt,
        else_body: []Stmt,
        span: Span,
    },

    /// with Monad: x <- expr; ...
    with_: struct {
        monad: []const u8,
        stmts: []WithStmt,
        span: Span,
    },

    /// open Type: x
    open_: struct {
        type_name: []const u8,
        value: *Expr,
        span: Span,
    },

    /// use x = expr
    use_: struct {
        name: []const u8,
        value: *Expr,
        span: Span,
    },

    /// let x = expr
    let_: struct {
        pattern: *Pattern,
        value: *Expr,
        span: Span,
    },

    /// Expression statement
    expr: *Expr,

    pub fn getSpan(self: Stmt) Span {
        return switch (self) {
            .assign => |a| a.span,
            .assign_op => |a| a.span,
            .return_ => |r| r.span,
            .if_ => |i| i.span,
            .switch_ => |s| s.span,
            .match_ => |m| m.span,
            .fold_ => |f| f.span,
            .bend_ => |b| b.span,
            .with_ => |w| w.span,
            .open_ => |o| o.span,
            .use_ => |u| u.span,
            .let_ => |l| l.span,
            .expr => |e| e.span,
        };
    }
};

pub const IfBranch = struct {
    cond: *Expr,
    body: []Stmt,
    span: Span,
};

pub const SwitchCase = struct {
    value: ?u32, // null = wildcard (_)
    body: []Stmt,
    span: Span,
};

pub const MatchCase = struct {
    pattern: *Pattern,
    body: []Stmt,
    span: Span,
};

pub const BendBinding = struct {
    name: []const u8,
    init: *Expr,
    span: Span,
};

pub const WithStmt = union(enum) {
    bind: struct {
        name: []const u8,
        value: *Expr,
        span: Span,
    },
    return_: struct {
        value: *Expr,
        span: Span,
    },
    stmt: Stmt,
};

// =============================================================================
// Expressions
// =============================================================================

/// Named struct type for match expression payload
pub const MatchExprPayload = struct {
    name: ?[]const u8,
    scrutinee: *Expr,
    cases: []ExprMatchCase,
};

/// Named struct type for fold expression payload
pub const FoldExprPayload = struct {
    name: ?[]const u8,
    scrutinee: *Expr,
    cases: []ExprMatchCase,
};

pub const Expr = struct {
    kind: ExprKind,
    span: Span,
};

pub const ExprKind = union(enum) {
    /// Variable reference: x, my_var
    var_: []const u8,

    /// Constructor reference: List/Nil, Tree/Node
    constructor: []const u8,

    /// Number literal: 42, -7, 3.14
    number: Number,

    /// Character literal: 'x'
    char_: u8,

    /// String literal: "hello"
    string: []const u8,

    /// Symbol literal: :name
    symbol: []const u8,

    /// List literal: [1, 2, 3]
    list: []*Expr,

    /// Tuple: (a, b, c)
    tuple: []*Expr,

    /// Superposition: {a b c}
    superposition: []*Expr,

    /// Lambda: λx body or \x body
    lambda: struct {
        params: [][]const u8,
        body: *Expr,
    },

    /// Application: f(x, y)
    app: struct {
        func: *Expr,
        args: []*Expr,
    },

    /// Binary operator: x + y
    binop: struct {
        op: BinOp,
        left: *Expr,
        right: *Expr,
    },

    /// Unary operator: -x, !x
    unop: struct {
        op: UnOp,
        operand: *Expr,
    },

    /// Let binding: let x = e1; e2
    let_: struct {
        pattern: *Pattern,
        value: *Expr,
        body: *Expr,
    },

    /// Use binding: use x = e1; e2
    use_: struct {
        name: []const u8,
        value: *Expr,
        body: *Expr,
    },

    /// If expression: if cond then e1 else e2
    if_: struct {
        cond: *Expr,
        then_: *Expr,
        else_: *Expr,
    },

    /// Match expression
    match_: MatchExprPayload,

    /// Switch expression
    switch_: struct {
        name: ?[]const u8,
        scrutinee: *Expr,
        cases: []ExprSwitchCase,
    },

    /// Fold expression
    fold_: FoldExprPayload,

    /// Bend expression
    bend_: struct {
        bindings: []BendBinding,
        when_cond: *Expr,
        when_body: *Expr,
        else_body: *Expr,
    },

    /// Fork (inside bend)
    fork: *Expr,

    /// Field access: x.field
    field_access: struct {
        value: *Expr,
        field: []const u8,
    },

    /// Nat literal: @n (binary tree encoding)
    nat: u32,

    /// Era/Eraser: *
    era,

    /// Hole: ?
    hole,
};

pub const ExprMatchCase = struct {
    pattern: *Pattern,
    body: *Expr,
    span: Span,
};

pub const ExprSwitchCase = struct {
    value: ?u32, // null = wildcard
    body: *Expr,
    span: Span,
};

// =============================================================================
// Patterns
// =============================================================================

pub const Pattern = struct {
    kind: PatternKind,
    span: Span,
};

pub const PatternKind = union(enum) {
    /// Variable binding: x
    var_: []const u8,

    /// Wildcard: _
    wildcard,

    /// Constructor pattern: List/Cons { head, tail }
    constructor: struct {
        name: []const u8,
        fields: []PatternField,
    },

    /// Number literal pattern: 0, 1, 42
    number: Number,

    /// Tuple pattern: (a, b, c)
    tuple: []*Pattern,

    /// Superposition pattern: {a b}
    superposition: []*Pattern,

    /// String pattern: "hello"
    string: []const u8,

    /// Char pattern: 'x'
    char_: u8,
};

pub const PatternField = struct {
    name: ?[]const u8, // Named field binding
    pattern: *Pattern,
    span: Span,
};

// =============================================================================
// Types
// =============================================================================

pub const TypeExpr = struct {
    kind: TypeExprKind,
    span: Span,
};

pub const TypeExprKind = union(enum) {
    /// Type variable: a, T
    var_: []const u8,

    /// Named type: List, Tree, Option
    named: []const u8,

    /// Type application: List(u24), Tree(T)
    app: struct {
        func: *TypeExpr,
        args: []*TypeExpr,
    },

    /// Function type: a -> b
    function: struct {
        param: *TypeExpr,
        ret: *TypeExpr,
    },

    /// Tuple type: (a, b, c)
    tuple: []*TypeExpr,

    /// Primitive types
    primitive: Primitive,

    /// Hole (type inference): ?
    hole,

    /// Any type
    any,

    /// None type
    none,
};

pub const Primitive = enum {
    u24,
    i24,
    f24,
};

// =============================================================================
// Operators
// =============================================================================

pub const BinOp = enum {
    // Arithmetic
    add, // +
    sub, // -
    mul, // *
    div, // /
    mod, // %

    // Comparison
    eq, // ==
    ne, // !=
    lt, // <
    le, // <=
    gt, // >
    ge, // >=

    // Bitwise
    band, // &
    bor, // |
    bxor, // ^
    shl, // <<
    shr, // >>

    // Logical
    land, // &&
    lor, // ||

    // Power (floats)
    pow, // **

    pub fn symbol(self: BinOp) []const u8 {
        return switch (self) {
            .add => "+",
            .sub => "-",
            .mul => "*",
            .div => "/",
            .mod => "%",
            .eq => "==",
            .ne => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",
            .band => "&",
            .bor => "|",
            .bxor => "^",
            .shl => "<<",
            .shr => ">>",
            .land => "&&",
            .lor => "||",
            .pow => "**",
        };
    }
};

pub const UnOp = enum {
    neg, // -
    not, // !
    bnot, // ~

    pub fn symbol(self: UnOp) []const u8 {
        return switch (self) {
            .neg => "-",
            .not => "!",
            .bnot => "~",
        };
    }
};

// =============================================================================
// Literals
// =============================================================================

pub const Number = union(enum) {
    unsigned: u32,
    signed: i32,
    float: f32,

    pub fn isZero(self: Number) bool {
        return switch (self) {
            .unsigned => |u| u == 0,
            .signed => |s| s == 0,
            .float => |f| f == 0.0,
        };
    }
};

// =============================================================================
// AST Builder Helpers
// =============================================================================

pub const AstBuilder = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) AstBuilder {
        return .{ .allocator = allocator };
    }

    // Expression builders
    pub fn var_(self: *AstBuilder, name: []const u8, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .{ .var_ = name }, .span = span };
        return expr;
    }

    pub fn number(self: *AstBuilder, n: Number, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .{ .number = n }, .span = span };
        return expr;
    }

    pub fn app(self: *AstBuilder, func: *Expr, args: []*Expr, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .{ .app = .{ .func = func, .args = args } }, .span = span };
        return expr;
    }

    pub fn lambda(self: *AstBuilder, params: [][]const u8, body: *Expr, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .{ .lambda = .{ .params = params, .body = body } }, .span = span };
        return expr;
    }

    pub fn binop(self: *AstBuilder, op: BinOp, left: *Expr, right: *Expr, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .{ .binop = .{ .op = op, .left = left, .right = right } }, .span = span };
        return expr;
    }

    pub fn if_(self: *AstBuilder, cond: *Expr, then_: *Expr, else_: *Expr, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .{ .if_ = .{ .cond = cond, .then_ = then_, .else_ = else_ } }, .span = span };
        return expr;
    }

    pub fn era(self: *AstBuilder, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .era, .span = span };
        return expr;
    }

    pub fn constructor(self: *AstBuilder, name: []const u8, span: Span) !*Expr {
        const expr = try self.allocator.create(Expr);
        expr.* = .{ .kind = .{ .constructor = name }, .span = span };
        return expr;
    }

    // Pattern builders
    pub fn patVar(self: *AstBuilder, name: []const u8, span: Span) !*Pattern {
        const pat = try self.allocator.create(Pattern);
        pat.* = .{ .kind = .{ .var_ = name }, .span = span };
        return pat;
    }

    pub fn patWildcard(self: *AstBuilder, span: Span) !*Pattern {
        const pat = try self.allocator.create(Pattern);
        pat.* = .{ .kind = .wildcard, .span = span };
        return pat;
    }

    pub fn patConstructor(self: *AstBuilder, name: []const u8, fields: []PatternField, span: Span) !*Pattern {
        const pat = try self.allocator.create(Pattern);
        pat.* = .{ .kind = .{ .constructor = .{ .name = name, .fields = fields } }, .span = span };
        return pat;
    }

    // Type expression builders
    pub fn typeVar(self: *AstBuilder, name: []const u8, span: Span) !*TypeExpr {
        const ty = try self.allocator.create(TypeExpr);
        ty.* = .{ .kind = .{ .var_ = name }, .span = span };
        return ty;
    }

    pub fn typeNamed(self: *AstBuilder, name: []const u8, span: Span) !*TypeExpr {
        const ty = try self.allocator.create(TypeExpr);
        ty.* = .{ .kind = .{ .named = name }, .span = span };
        return ty;
    }

    pub fn typeFunction(self: *AstBuilder, param: *TypeExpr, ret: *TypeExpr, span: Span) !*TypeExpr {
        const ty = try self.allocator.create(TypeExpr);
        ty.* = .{ .kind = .{ .function = .{ .param = param, .ret = ret } }, .span = span };
        return ty;
    }

    pub fn typePrimitive(self: *AstBuilder, prim: Primitive, span: Span) !*TypeExpr {
        const ty = try self.allocator.create(TypeExpr);
        ty.* = .{ .kind = .{ .primitive = prim }, .span = span };
        return ty;
    }
};

// =============================================================================
// Pretty Printing (for debugging)
// =============================================================================

pub fn prettyExpr(writer: anytype, expr: *const Expr) !void {
    switch (expr.kind) {
        .var_ => |name| try writer.writeAll(name),
        .constructor => |name| try writer.writeAll(name),
        .number => |n| switch (n) {
            .unsigned => |u| try std.fmt.format(writer, "{d}", .{u}),
            .signed => |s| try std.fmt.format(writer, "{d}", .{s}),
            .float => |f| try std.fmt.format(writer, "{d}", .{f}),
        },
        .char_ => |c| try std.fmt.format(writer, "'{c}'", .{c}),
        .string => |s| try std.fmt.format(writer, "\"{s}\"", .{s}),
        .lambda => |lam| {
            try writer.writeAll("λ");
            for (lam.params, 0..) |p, i| {
                if (i > 0) try writer.writeByte(' ');
                try writer.writeAll(p);
            }
            try writer.writeAll(" ");
            try prettyExpr(writer, lam.body);
        },
        .app => |a| {
            try prettyExpr(writer, a.func);
            try writer.writeByte('(');
            for (a.args, 0..) |arg, i| {
                if (i > 0) try writer.writeAll(", ");
                try prettyExpr(writer, arg);
            }
            try writer.writeByte(')');
        },
        .binop => |b| {
            try writer.writeByte('(');
            try prettyExpr(writer, b.left);
            try writer.writeByte(' ');
            try writer.writeAll(b.op.symbol());
            try writer.writeByte(' ');
            try prettyExpr(writer, b.right);
            try writer.writeByte(')');
        },
        .if_ => |i| {
            try writer.writeAll("if ");
            try prettyExpr(writer, i.cond);
            try writer.writeAll(" then ");
            try prettyExpr(writer, i.then_);
            try writer.writeAll(" else ");
            try prettyExpr(writer, i.else_);
        },
        .era => try writer.writeByte('*'),
        .hole => try writer.writeByte('?'),
        else => try writer.writeAll("<expr>"),
    }
}
