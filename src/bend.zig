const std = @import("std");
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const Token = lexer.Token;
const TokenKind = lexer.TokenKind;

// =============================================================================
// Bend Parser
// =============================================================================
// Parses Bend source code into AST, supporting both Imp and Fun syntaxes

pub const ParseError = error{
    UnexpectedToken,
    ExpectedToken,
    ExpectedExpression,
    ExpectedPattern,
    ExpectedType,
    ExpectedIdentifier,
    ExpectedConstructor,
    InvalidIndentation,
    UnterminatedString,
    OutOfMemory,
};

pub const Parser = struct {
    tokens: []const Token,
    pos: usize,
    allocator: Allocator,
    builder: ast.AstBuilder,
    errors: ArrayList(Error),

    pub const Error = struct {
        message: []const u8,
        span: ast.Span,
    };

    pub fn init(tokens: []const Token, allocator: Allocator) Parser {
        return .{
            .tokens = tokens,
            .pos = 0,
            .allocator = allocator,
            .builder = ast.AstBuilder.init(allocator),
            .errors = .empty,
        };
    }

    pub fn deinit(self: *Parser) void {
        self.errors.deinit(self.allocator);
    }

    // =========================================================================
    // Token Utilities
    // =========================================================================

    fn peek(self: *Parser) Token {
        if (self.pos >= self.tokens.len) {
            return .{ .kind = .eof, .text = "", .span = .{ .start = 0, .end = 0, .line = 0, .col = 0 } };
        }
        return self.tokens[self.pos];
    }

    fn peekKind(self: *Parser) TokenKind {
        return self.peek().kind;
    }

    fn advance(self: *Parser) Token {
        const token = self.peek();
        if (self.pos < self.tokens.len) {
            self.pos += 1;
        }
        return token;
    }

    fn check(self: *Parser, kind: TokenKind) bool {
        return self.peekKind() == kind;
    }

    fn match(self: *Parser, kind: TokenKind) bool {
        if (self.check(kind)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn expect(self: *Parser, kind: TokenKind) ParseError!Token {
        if (self.check(kind)) {
            return self.advance();
        }
        try self.errors.append(self.allocator, .{
            .message = "unexpected token",
            .span = self.peek().span,
        });
        return ParseError.ExpectedToken;
    }

    fn skipNewlines(self: *Parser) void {
        while (self.check(.newline)) {
            _ = self.advance();
        }
    }

    fn currentSpan(self: *Parser) ast.Span {
        return self.peek().span;
    }

    // =========================================================================
    // Program Parsing
    // =========================================================================

    pub fn parseProgram(self: *Parser) ParseError!ast.Program {
        var program = ast.Program.init(self.allocator);

        self.skipNewlines();

        while (!self.check(.eof)) {
            if (self.check(.kw_type)) {
                const typedef = try self.parseTypeDef();
                try program.types.append(self.allocator, typedef);
            } else if (self.check(.kw_object)) {
                const objdef = try self.parseObjectDef();
                try program.objects.append(self.allocator, objdef);
            } else if (self.check(.kw_def)) {
                const funcdef = try self.parseFuncDefImp();
                try program.funcs.append(self.allocator, funcdef);
            } else if (self.check(.ident) or self.check(.upper_ident)) {
                // Could be Fun-style function or main expression
                const saved_pos = self.pos;
                _ = self.advance();

                if (self.check(.lparen)) {
                    // Function definition (Fun style)
                    self.pos = saved_pos;
                    const funcdef = try self.parseFuncDefFun();
                    try program.funcs.append(self.allocator, funcdef);
                } else if (self.check(.assign)) {
                    // Simple assignment: main = expr
                    self.pos = saved_pos;
                    const funcdef = try self.parseFuncDefFun();
                    if (std.mem.eql(u8, funcdef.name, "main")) {
                        if (funcdef.body == .expr) {
                            program.main = funcdef.body.expr;
                        }
                    } else {
                        try program.funcs.append(self.allocator, funcdef);
                    }
                } else {
                    // Expression
                    self.pos = saved_pos;
                    program.main = try self.parseExpr();
                }
            } else if (self.check(.newline) or self.check(.dedent)) {
                _ = self.advance();
            } else {
                // Try to parse as main expression
                program.main = try self.parseExpr();
            }

            self.skipNewlines();
        }

        return program;
    }

    // =========================================================================
    // Type Definition Parsing
    // =========================================================================

    /// type List(T):
    ///   Nil
    ///   Cons { head: T, ~tail: List(T) }
    fn parseTypeDef(self: *Parser) ParseError!ast.TypeDef {
        const start = self.currentSpan();
        _ = try self.expect(.kw_type);

        const name_tok = try self.expect(.upper_ident);
        const name = name_tok.text;

        // Type parameters
        var params: ArrayList([]const u8) = .empty;
        if (self.match(.lparen)) {
            while (!self.check(.rparen) and !self.check(.eof)) {
                const param = try self.expect(.upper_ident);
                try params.append(self.allocator, param.text);
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.rparen);
        }

        _ = try self.expect(.colon);
        self.skipNewlines();

        // Expect indent for constructors
        if (self.check(.indent)) {
            _ = self.advance();
        }

        // Parse constructors
        var constructors: ArrayList(ast.Constructor) = .empty;
        while (!self.check(.dedent) and !self.check(.eof) and !self.check(.kw_type) and !self.check(.kw_def) and !self.check(.kw_object)) {
            self.skipNewlines();
            if (self.check(.dedent) or self.check(.eof)) break;

            const ctor = try self.parseConstructor();
            try constructors.append(self.allocator, ctor);

            self.skipNewlines();
        }

        if (self.check(.dedent)) {
            _ = self.advance();
        }

        return .{
            .name = name,
            .params = try params.toOwnedSlice(self.allocator),
            .constructors = try constructors.toOwnedSlice(self.allocator),
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    fn parseConstructor(self: *Parser) ParseError!ast.Constructor {
        const start = self.currentSpan();
        const name_tok = try self.expect(.upper_ident);

        var fields: ArrayList(ast.Field) = .empty;

        // Optional fields in braces
        if (self.match(.lbrace)) {
            while (!self.check(.rbrace) and !self.check(.eof)) {
                const field = try self.parseField();
                try fields.append(self.allocator, field);
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.rbrace);
        }

        return .{
            .name = name_tok.text,
            .fields = try fields.toOwnedSlice(self.allocator),
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    fn parseField(self: *Parser) ParseError!ast.Field {
        const start = self.currentSpan();

        // Check for recursive marker
        const is_recursive = self.match(.tilde);

        const name_tok = try self.expect(.ident);

        // Optional type annotation
        var type_: ?*ast.TypeExpr = null;
        if (self.match(.colon)) {
            type_ = try self.parseTypeExpr();
        }

        return .{
            .name = name_tok.text,
            .type_ = type_,
            .is_recursive = is_recursive,
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    // =========================================================================
    // Object Definition Parsing
    // =========================================================================

    /// object Point { x, y }
    fn parseObjectDef(self: *Parser) ParseError!ast.ObjectDef {
        const start = self.currentSpan();
        _ = try self.expect(.kw_object);

        const name_tok = try self.expect(.upper_ident);

        // Type parameters
        var params: ArrayList([]const u8) = .empty;
        if (self.match(.lparen)) {
            while (!self.check(.rparen) and !self.check(.eof)) {
                const param = try self.expect(.upper_ident);
                try params.append(self.allocator, param.text);
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.rparen);
        }

        // Fields
        var fields: ArrayList(ast.Field) = .empty;
        _ = try self.expect(.lbrace);
        while (!self.check(.rbrace) and !self.check(.eof)) {
            const field = try self.parseField();
            try fields.append(self.allocator, field);
            if (!self.match(.comma)) break;
        }
        _ = try self.expect(.rbrace);

        return .{
            .name = name_tok.text,
            .params = try params.toOwnedSlice(self.allocator),
            .fields = try fields.toOwnedSlice(self.allocator),
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    // =========================================================================
    // Function Definition Parsing
    // =========================================================================

    /// Imp style: def add(x: u24, y: u24) -> u24: return x + y
    fn parseFuncDefImp(self: *Parser) ParseError!ast.FuncDef {
        const start = self.currentSpan();
        _ = try self.expect(.kw_def);

        // Optional checked/unchecked
        const is_checked = !self.match(.kw_unchecked);
        if (!is_checked) {} else if (self.match(.kw_checked)) {}

        const name_tok = if (self.check(.ident))
            try self.expect(.ident)
        else
            try self.expect(.upper_ident);

        // Parameters
        var params: ArrayList(ast.Param) = .empty;
        _ = try self.expect(.lparen);
        while (!self.check(.rparen) and !self.check(.eof)) {
            const param = try self.parseParam();
            try params.append(self.allocator, param);
            if (!self.match(.comma)) break;
        }
        _ = try self.expect(.rparen);

        // Return type
        var return_type: ?*ast.TypeExpr = null;
        if (self.match(.arrow)) {
            return_type = try self.parseTypeExpr();
        }

        _ = try self.expect(.colon);
        self.skipNewlines();

        // Parse body (statements)
        var stmts: ArrayList(ast.Stmt) = .empty;
        if (self.match(.indent)) {
            while (!self.check(.dedent) and !self.check(.eof)) {
                self.skipNewlines();
                if (self.check(.dedent) or self.check(.eof)) break;
                const stmt = try self.parseStmt();
                try stmts.append(self.allocator, stmt);
                self.skipNewlines();
            }
            if (self.check(.dedent)) {
                _ = self.advance();
            }
        } else {
            // Single line body
            const stmt = try self.parseStmt();
            try stmts.append(self.allocator, stmt);
        }

        return .{
            .name = name_tok.text,
            .params = try params.toOwnedSlice(self.allocator),
            .return_type = return_type,
            .body = .{ .stmts = try stmts.toOwnedSlice(self.allocator) },
            .is_checked = is_checked,
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    /// Fun style: add(x, y) = x + y
    fn parseFuncDefFun(self: *Parser) ParseError!ast.FuncDef {
        const start = self.currentSpan();

        const name_tok = if (self.check(.ident))
            self.advance()
        else
            try self.expect(.upper_ident);

        // Parameters
        var params: ArrayList(ast.Param) = .empty;
        if (self.match(.lparen)) {
            while (!self.check(.rparen) and !self.check(.eof)) {
                const param = try self.parseParam();
                try params.append(self.allocator, param);
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.rparen);
        }

        // Optional return type
        var return_type: ?*ast.TypeExpr = null;
        if (self.match(.arrow)) {
            return_type = try self.parseTypeExpr();
        }

        _ = try self.expect(.assign);
        self.skipNewlines();

        // Parse body (expression)
        const body = try self.parseExpr();

        return .{
            .name = name_tok.text,
            .params = try params.toOwnedSlice(self.allocator),
            .return_type = return_type,
            .body = .{ .expr = body },
            .is_checked = true,
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    fn parseParam(self: *Parser) ParseError!ast.Param {
        const start = self.currentSpan();
        const name_tok = try self.expect(.ident);

        var type_: ?*ast.TypeExpr = null;
        if (self.match(.colon)) {
            type_ = try self.parseTypeExpr();
        }

        return .{
            .name = name_tok.text,
            .type_ = type_,
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    // =========================================================================
    // Statement Parsing
    // =========================================================================

    fn parseStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();

        if (self.check(.kw_return)) {
            return self.parseReturn();
        }
        if (self.check(.kw_if)) {
            return self.parseIfStmt();
        }
        if (self.check(.kw_match)) {
            return self.parseMatchStmt();
        }
        if (self.check(.kw_switch)) {
            return self.parseSwitchStmt();
        }
        if (self.check(.kw_fold)) {
            return self.parseFoldStmt();
        }
        if (self.check(.kw_bend)) {
            return self.parseBendStmt();
        }
        if (self.check(.kw_with)) {
            return self.parseWithStmt();
        }
        if (self.check(.kw_open)) {
            return self.parseOpenStmt();
        }
        if (self.check(.kw_use)) {
            return self.parseUseStmt();
        }
        if (self.check(.kw_let)) {
            return self.parseLetStmt();
        }

        // Assignment or expression
        const expr = try self.parseExpr();

        // Check for assignment
        if (self.check(.assign) or self.check(.plus_eq) or self.check(.minus_eq) or
            self.check(.star_eq) or self.check(.slash_eq))
        {
            const op_tok = self.advance();

            // Convert expression to pattern for LHS
            const pattern = try self.exprToPattern(expr);
            const value = try self.parseExpr();

            if (op_tok.kind == .assign) {
                return .{ .assign = .{
                    .pattern = pattern,
                    .value = value,
                    .span = ast.Span.merge(start, self.currentSpan()),
                } };
            } else {
                // Compound assignment
                const op: ast.BinOp = switch (op_tok.kind) {
                    .plus_eq => .add,
                    .minus_eq => .sub,
                    .star_eq => .mul,
                    .slash_eq => .div,
                    else => .add,
                };
                return .{ .assign_op = .{
                    .target = if (expr.kind == .var_) expr.kind.var_ else "",
                    .op = op,
                    .value = value,
                    .span = ast.Span.merge(start, self.currentSpan()),
                } };
            }
        }

        return .{ .expr = expr };
    }

    fn parseReturn(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_return);

        var value: ?*ast.Expr = null;
        if (!self.check(.newline) and !self.check(.dedent) and !self.check(.eof)) {
            value = try self.parseExpr();
        }

        return .{ .return_ = .{
            .value = value,
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseIfStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_if);

        var branches: ArrayList(ast.IfBranch) = .empty;

        // First branch
        const cond = try self.parseExpr();
        _ = try self.expect(.colon);
        const body = try self.parseBlock();
        try branches.append(self.allocator, .{
            .cond = cond,
            .body = body,
            .span = ast.Span.merge(start, self.currentSpan()),
        });

        // Elif branches
        while (self.match(.kw_elif)) {
            const elif_cond = try self.parseExpr();
            _ = try self.expect(.colon);
            const elif_body = try self.parseBlock();
            try branches.append(self.allocator, .{
                .cond = elif_cond,
                .body = elif_body,
                .span = ast.Span.merge(start, self.currentSpan()),
            });
        }

        // Else branch
        var else_body: ?[]ast.Stmt = null;
        if (self.match(.kw_else)) {
            _ = try self.expect(.colon);
            else_body = try self.parseBlock();
        }

        return .{ .if_ = .{
            .branches = try branches.toOwnedSlice(self.allocator),
            .else_body = else_body,
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseMatchStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_match);

        // Optional binding name
        var name: ?[]const u8 = null;
        if (self.check(.ident)) {
            const saved = self.pos;
            const n = self.advance();
            if (self.match(.assign)) {
                name = n.text;
            } else {
                self.pos = saved;
            }
        }

        const scrutinee = try self.parseExpr();
        _ = try self.expect(.colon);
        self.skipNewlines();

        var cases: ArrayList(ast.MatchCase) = .empty;
        if (self.match(.indent)) {
            while (!self.check(.dedent) and !self.check(.eof)) {
                self.skipNewlines();
                if (self.check(.dedent)) break;
                const case = try self.parseMatchCase();
                try cases.append(self.allocator, case);
            }
            if (self.check(.dedent)) _ = self.advance();
        }

        return .{ .match_ = .{
            .name = name,
            .scrutinee = scrutinee,
            .cases = try cases.toOwnedSlice(self.allocator),
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseMatchCase(self: *Parser) ParseError!ast.MatchCase {
        const start = self.currentSpan();
        _ = try self.expect(.kw_case);

        const pattern = try self.parsePattern();
        _ = try self.expect(.colon);
        const body = try self.parseBlock();

        return .{
            .pattern = pattern,
            .body = body,
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    fn parseSwitchStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_switch);

        var name: ?[]const u8 = null;
        if (self.check(.ident)) {
            const saved = self.pos;
            const n = self.advance();
            if (self.match(.assign)) {
                name = n.text;
            } else {
                self.pos = saved;
            }
        }

        const scrutinee = try self.parseExpr();
        _ = try self.expect(.colon);
        self.skipNewlines();

        var cases: ArrayList(ast.SwitchCase) = .empty;
        if (self.match(.indent)) {
            while (!self.check(.dedent) and !self.check(.eof)) {
                self.skipNewlines();
                if (self.check(.dedent)) break;
                const case = try self.parseSwitchCase();
                try cases.append(self.allocator, case);
            }
            if (self.check(.dedent)) _ = self.advance();
        }

        return .{ .switch_ = .{
            .name = name,
            .scrutinee = scrutinee,
            .cases = try cases.toOwnedSlice(self.allocator),
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseSwitchCase(self: *Parser) ParseError!ast.SwitchCase {
        const start = self.currentSpan();
        _ = try self.expect(.kw_case);

        var value: ?u32 = null;
        if (self.check(.number)) {
            const num_tok = self.advance();
            if (num_tok.number_value) |nv| {
                value = switch (nv) {
                    .unsigned => |u| u,
                    .signed => |s| @bitCast(s),
                    .float => |f| @intFromFloat(f),
                };
            }
        } else if (self.match(.underscore)) {
            value = null; // Wildcard
        }

        _ = try self.expect(.colon);
        const body = try self.parseBlock();

        return .{
            .value = value,
            .body = body,
            .span = ast.Span.merge(start, self.currentSpan()),
        };
    }

    fn parseFoldStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_fold);

        var name: ?[]const u8 = null;
        if (self.check(.ident)) {
            const saved = self.pos;
            const n = self.advance();
            if (self.match(.assign)) {
                name = n.text;
            } else {
                self.pos = saved;
            }
        }

        const scrutinee = try self.parseExpr();
        _ = try self.expect(.colon);
        self.skipNewlines();

        var cases: ArrayList(ast.MatchCase) = .empty;
        if (self.match(.indent)) {
            while (!self.check(.dedent) and !self.check(.eof)) {
                self.skipNewlines();
                if (self.check(.dedent)) break;
                const case = try self.parseMatchCase();
                try cases.append(self.allocator, case);
            }
            if (self.check(.dedent)) _ = self.advance();
        }

        return .{ .fold_ = .{
            .name = name,
            .scrutinee = scrutinee,
            .cases = try cases.toOwnedSlice(self.allocator),
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseBendStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_bend);

        var bindings: ArrayList(ast.BendBinding) = .empty;
        while (self.check(.ident)) {
            const name_tok = self.advance();
            _ = try self.expect(.assign);
            const init_expr = try self.parseExpr();
            try bindings.append(self.allocator, .{
                .name = name_tok.text,
                .init = init_expr,
                .span = ast.Span.merge(start, self.currentSpan()),
            });
            if (!self.match(.comma)) break;
        }

        _ = try self.expect(.colon);
        self.skipNewlines();

        var when_cond: *ast.Expr = undefined;
        var when_body: []ast.Stmt = &[_]ast.Stmt{};
        var else_body: []ast.Stmt = &[_]ast.Stmt{};

        if (self.match(.indent)) {
            if (self.match(.kw_when)) {
                when_cond = try self.parseExpr();
                _ = try self.expect(.colon);
                when_body = try self.parseBlock();
            }
            self.skipNewlines();
            if (self.match(.kw_else)) {
                _ = try self.expect(.colon);
                else_body = try self.parseBlock();
            }
            if (self.check(.dedent)) _ = self.advance();
        }

        return .{ .bend_ = .{
            .bindings = try bindings.toOwnedSlice(self.allocator),
            .when_cond = when_cond,
            .when_body = when_body,
            .else_body = else_body,
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseWithStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_with);

        const monad_tok = try self.expect(.upper_ident);
        _ = try self.expect(.colon);
        self.skipNewlines();

        var stmts: ArrayList(ast.WithStmt) = .empty;
        if (self.match(.indent)) {
            while (!self.check(.dedent) and !self.check(.eof)) {
                self.skipNewlines();
                if (self.check(.dedent)) break;
                const ws = try self.parseWithStmtInner();
                try stmts.append(self.allocator, ws);
            }
            if (self.check(.dedent)) _ = self.advance();
        }

        return .{ .with_ = .{
            .monad = monad_tok.text,
            .stmts = try stmts.toOwnedSlice(self.allocator),
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseWithStmtInner(self: *Parser) ParseError!ast.WithStmt {
        const start = self.currentSpan();

        if (self.check(.kw_return)) {
            _ = self.advance();
            const value = try self.parseExpr();
            return .{ .return_ = .{
                .value = value,
                .span = ast.Span.merge(start, self.currentSpan()),
            } };
        }

        // Check for bind: x <- expr
        if (self.check(.ident)) {
            const saved = self.pos;
            const name_tok = self.advance();
            if (self.match(.bind_arrow)) {
                const value = try self.parseExpr();
                return .{ .bind = .{
                    .name = name_tok.text,
                    .value = value,
                    .span = ast.Span.merge(start, self.currentSpan()),
                } };
            }
            self.pos = saved;
        }

        // Regular statement
        const stmt = try self.parseStmt();
        return .{ .stmt = stmt };
    }

    fn parseOpenStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_open);

        const type_tok = try self.expect(.upper_ident);
        _ = try self.expect(.colon);
        const value = try self.parseExpr();

        return .{ .open_ = .{
            .type_name = type_tok.text,
            .value = value,
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseUseStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_use);

        const name_tok = try self.expect(.ident);
        _ = try self.expect(.assign);
        const value = try self.parseExpr();

        return .{ .use_ = .{
            .name = name_tok.text,
            .value = value,
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseLetStmt(self: *Parser) ParseError!ast.Stmt {
        const start = self.currentSpan();
        _ = try self.expect(.kw_let);

        const pattern = try self.parsePattern();
        _ = try self.expect(.assign);
        const value = try self.parseExpr();

        return .{ .let_ = .{
            .pattern = pattern,
            .value = value,
            .span = ast.Span.merge(start, self.currentSpan()),
        } };
    }

    fn parseBlock(self: *Parser) ParseError![]ast.Stmt {
        self.skipNewlines();
        var stmts: ArrayList(ast.Stmt) = .empty;

        if (self.match(.indent)) {
            while (!self.check(.dedent) and !self.check(.eof)) {
                self.skipNewlines();
                if (self.check(.dedent) or self.check(.eof)) break;
                const stmt = try self.parseStmt();
                try stmts.append(self.allocator, stmt);
                self.skipNewlines();
            }
            if (self.check(.dedent)) _ = self.advance();
        } else {
            // Single statement on same line
            const stmt = try self.parseStmt();
            try stmts.append(self.allocator, stmt);
        }

        return try stmts.toOwnedSlice(self.allocator);
    }

    // =========================================================================
    // Expression Parsing
    // =========================================================================

    fn parseExpr(self: *Parser) ParseError!*ast.Expr {
        return self.parseOr();
    }

    fn parseOr(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseAnd();

        while (self.match(.lor)) {
            const right = try self.parseAnd();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(.lor, left, right, span);
        }

        return left;
    }

    fn parseAnd(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseEquality();

        while (self.match(.land)) {
            const right = try self.parseEquality();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(.land, left, right, span);
        }

        return left;
    }

    fn parseEquality(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseComparison();

        while (true) {
            const op: ast.BinOp = if (self.match(.eq))
                .eq
            else if (self.match(.ne))
                .ne
            else
                break;

            const right = try self.parseComparison();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(op, left, right, span);
        }

        return left;
    }

    fn parseComparison(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseBitOr();

        while (true) {
            const op: ast.BinOp = if (self.match(.lt))
                .lt
            else if (self.match(.le))
                .le
            else if (self.match(.gt))
                .gt
            else if (self.match(.ge))
                .ge
            else
                break;

            const right = try self.parseBitOr();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(op, left, right, span);
        }

        return left;
    }

    fn parseBitOr(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseBitXor();

        while (self.match(.pipe)) {
            const right = try self.parseBitXor();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(.bor, left, right, span);
        }

        return left;
    }

    fn parseBitXor(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseBitAnd();

        while (self.match(.caret)) {
            const right = try self.parseBitAnd();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(.bxor, left, right, span);
        }

        return left;
    }

    fn parseBitAnd(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseShift();

        while (self.match(.ampersand)) {
            const right = try self.parseShift();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(.band, left, right, span);
        }

        return left;
    }

    fn parseShift(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseAddSub();

        while (true) {
            const op: ast.BinOp = if (self.match(.shl))
                .shl
            else if (self.match(.shr))
                .shr
            else
                break;

            const right = try self.parseAddSub();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(op, left, right, span);
        }

        return left;
    }

    fn parseAddSub(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseMulDiv();

        while (true) {
            const op: ast.BinOp = if (self.match(.plus))
                .add
            else if (self.match(.minus))
                .sub
            else
                break;

            const right = try self.parseMulDiv();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(op, left, right, span);
        }

        return left;
    }

    fn parseMulDiv(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parsePower();

        while (true) {
            const op: ast.BinOp = if (self.match(.star))
                .mul
            else if (self.match(.slash))
                .div
            else if (self.match(.percent))
                .mod
            else
                break;

            const right = try self.parsePower();
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(op, left, right, span);
        }

        return left;
    }

    fn parsePower(self: *Parser) ParseError!*ast.Expr {
        var left = try self.parseUnary();

        if (self.match(.pow)) {
            const right = try self.parsePower(); // Right associative
            const span = ast.Span.merge(left.span, right.span);
            left = try self.builder.binop(.pow, left, right, span);
        }

        return left;
    }

    fn parseUnary(self: *Parser) ParseError!*ast.Expr {
        const start = self.currentSpan();

        if (self.match(.minus)) {
            const operand = try self.parseUnary();
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{
                .kind = .{ .unop = .{ .op = .neg, .operand = operand } },
                .span = ast.Span.merge(start, operand.span),
            };
            return expr;
        }

        if (self.match(.tilde)) {
            const operand = try self.parseUnary();
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{
                .kind = .{ .unop = .{ .op = .bnot, .operand = operand } },
                .span = ast.Span.merge(start, operand.span),
            };
            return expr;
        }

        return self.parsePostfix();
    }

    fn parsePostfix(self: *Parser) ParseError!*ast.Expr {
        var expr = try self.parsePrimary();

        while (true) {
            if (self.match(.lparen)) {
                // Function call
                var args: ArrayList(*ast.Expr) = .empty;
                while (!self.check(.rparen) and !self.check(.eof)) {
                    try args.append(self.allocator, try self.parseExpr());
                    if (!self.match(.comma)) break;
                }
                _ = try self.expect(.rparen);

                const new_expr = try self.allocator.create(ast.Expr);
                new_expr.* = .{
                    .kind = .{ .app = .{ .func = expr, .args = try args.toOwnedSlice(self.allocator) } },
                    .span = ast.Span.merge(expr.span, self.currentSpan()),
                };
                expr = new_expr;
            } else if (self.match(.dot)) {
                // Field access
                const field_tok = try self.expect(.ident);
                const new_expr = try self.allocator.create(ast.Expr);
                new_expr.* = .{
                    .kind = .{ .field_access = .{ .value = expr, .field = field_tok.text } },
                    .span = ast.Span.merge(expr.span, self.currentSpan()),
                };
                expr = new_expr;
            } else {
                break;
            }
        }

        return expr;
    }

    fn parsePrimary(self: *Parser) ParseError!*ast.Expr {
        const start = self.currentSpan();

        // Number
        if (self.check(.number)) {
            const tok = self.advance();
            return self.builder.number(tok.number_value orelse .{ .unsigned = 0 }, tok.span);
        }

        // String
        if (self.check(.string)) {
            const tok = self.advance();
            const expr = try self.allocator.create(ast.Expr);
            // Remove quotes
            const text = if (tok.text.len >= 2) tok.text[1 .. tok.text.len - 1] else tok.text;
            expr.* = .{ .kind = .{ .string = text }, .span = tok.span };
            return expr;
        }

        // Character
        if (self.check(.char_)) {
            const tok = self.advance();
            const expr = try self.allocator.create(ast.Expr);
            const c = if (tok.text.len >= 2) tok.text[1] else 0;
            expr.* = .{ .kind = .{ .char_ = c }, .span = tok.span };
            return expr;
        }

        // Variable or constructor
        if (self.check(.ident)) {
            const tok = self.advance();
            return self.builder.var_(tok.text, tok.span);
        }

        if (self.check(.upper_ident)) {
            const tok = self.advance();
            return self.builder.constructor(tok.text, tok.span);
        }

        // Lambda
        if (self.check(.lambda) or self.check(.backslash)) {
            _ = self.advance();
            var params: ArrayList([]const u8) = .empty;
            while (self.check(.ident)) {
                try params.append(self.allocator, self.advance().text);
            }
            const body = try self.parseExpr();
            return self.builder.lambda(try params.toOwnedSlice(self.allocator), body, ast.Span.merge(start, body.span));
        }

        // Eraser
        if (self.match(.star)) {
            return self.builder.era(start);
        }

        // Hole
        if (self.match(.question)) {
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{ .kind = .hole, .span = start };
            return expr;
        }

        // Parenthesized expression or tuple
        if (self.match(.lparen)) {
            if (self.match(.rparen)) {
                // Unit/empty tuple
                const expr = try self.allocator.create(ast.Expr);
                expr.* = .{ .kind = .{ .tuple = &[_]*ast.Expr{} }, .span = ast.Span.merge(start, self.currentSpan()) };
                return expr;
            }

            const first = try self.parseExpr();
            if (self.match(.comma)) {
                // Tuple
                var elements: ArrayList(*ast.Expr) = .empty;
                try elements.append(self.allocator, first);
                while (!self.check(.rparen) and !self.check(.eof)) {
                    try elements.append(self.allocator, try self.parseExpr());
                    if (!self.match(.comma)) break;
                }
                _ = try self.expect(.rparen);
                const expr = try self.allocator.create(ast.Expr);
                expr.* = .{ .kind = .{ .tuple = try elements.toOwnedSlice(self.allocator) }, .span = ast.Span.merge(start, self.currentSpan()) };
                return expr;
            }
            _ = try self.expect(.rparen);
            return first;
        }

        // List literal
        if (self.match(.lbracket)) {
            var elements: ArrayList(*ast.Expr) = .empty;
            while (!self.check(.rbracket) and !self.check(.eof)) {
                try elements.append(self.allocator, try self.parseExpr());
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.rbracket);
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{ .kind = .{ .list = try elements.toOwnedSlice(self.allocator) }, .span = ast.Span.merge(start, self.currentSpan()) };
            return expr;
        }

        // Superposition
        if (self.match(.lbrace)) {
            var elements: ArrayList(*ast.Expr) = .empty;
            while (!self.check(.rbrace) and !self.check(.eof)) {
                try elements.append(self.allocator, try self.parseExpr());
            }
            _ = try self.expect(.rbrace);
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{ .kind = .{ .superposition = try elements.toOwnedSlice(self.allocator) }, .span = ast.Span.merge(start, self.currentSpan()) };
            return expr;
        }

        // If expression
        if (self.match(.kw_if)) {
            const cond = try self.parseExpr();
            _ = try self.expect(.kw_then);
            const then_ = try self.parseExpr();
            _ = try self.expect(.kw_else);
            const else_ = try self.parseExpr();
            return self.builder.if_(cond, then_, else_, ast.Span.merge(start, else_.span));
        }

        // Let expression
        if (self.match(.kw_let)) {
            const pattern = try self.parsePattern();
            _ = try self.expect(.assign);
            const value = try self.parseExpr();
            // Expect semicolon or newline, then body
            if (self.check(.newline)) _ = self.advance();
            const body = try self.parseExpr();
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{
                .kind = .{ .let_ = .{ .pattern = pattern, .value = value, .body = body } },
                .span = ast.Span.merge(start, body.span),
            };
            return expr;
        }

        // Match expression
        if (self.match(.kw_match)) {
            return self.parseMatchExpr(start);
        }

        // Fork
        if (self.match(.kw_fork)) {
            _ = try self.expect(.lparen);
            const arg = try self.parseExpr();
            _ = try self.expect(.rparen);
            const expr = try self.allocator.create(ast.Expr);
            expr.* = .{
                .kind = .{ .fork = arg },
                .span = ast.Span.merge(start, self.currentSpan()),
            };
            return expr;
        }

        try self.errors.append(self.allocator, .{
            .message = "expected expression",
            .span = start,
        });
        return ParseError.ExpectedExpression;
    }

    fn parseMatchExpr(self: *Parser, start: ast.Span) ParseError!*ast.Expr {
        var name: ?[]const u8 = null;
        if (self.check(.ident)) {
            const saved = self.pos;
            const n = self.advance();
            if (self.match(.assign)) {
                name = n.text;
            } else {
                self.pos = saved;
            }
        }

        const scrutinee = try self.parseExpr();
        _ = try self.expect(.colon);
        self.skipNewlines();

        var cases: ArrayList(ast.ExprMatchCase) = .empty;
        if (self.match(.indent)) {
            while (!self.check(.dedent) and !self.check(.eof)) {
                self.skipNewlines();
                if (self.check(.dedent)) break;
                _ = try self.expect(.kw_case);
                const pattern = try self.parsePattern();
                _ = try self.expect(.colon);
                const body = try self.parseExpr();
                try cases.append(self.allocator, .{
                    .pattern = pattern,
                    .body = body,
                    .span = ast.Span.merge(pattern.span, body.span),
                });
                self.skipNewlines();
            }
            if (self.check(.dedent)) _ = self.advance();
        }

        const expr = try self.allocator.create(ast.Expr);
        expr.* = .{
            .kind = .{ .match_ = .{
                .name = name,
                .scrutinee = scrutinee,
                .cases = try cases.toOwnedSlice(self.allocator),
            } },
            .span = ast.Span.merge(start, self.currentSpan()),
        };
        return expr;
    }

    // =========================================================================
    // Pattern Parsing
    // =========================================================================

    fn parsePattern(self: *Parser) ParseError!*ast.Pattern {
        const start = self.currentSpan();

        // Wildcard
        if (self.match(.underscore)) {
            return self.builder.patWildcard(start);
        }

        // Variable
        if (self.check(.ident)) {
            const tok = self.advance();
            return self.builder.patVar(tok.text, tok.span);
        }

        // Constructor pattern
        if (self.check(.upper_ident)) {
            const tok = self.advance();
            var fields: ArrayList(ast.PatternField) = .empty;

            if (self.match(.lbrace)) {
                while (!self.check(.rbrace) and !self.check(.eof)) {
                    const field_pat = try self.parsePatternField();
                    try fields.append(self.allocator, field_pat);
                    if (!self.match(.comma)) break;
                }
                _ = try self.expect(.rbrace);
            }

            return self.builder.patConstructor(tok.text, try fields.toOwnedSlice(self.allocator), ast.Span.merge(start, self.currentSpan()));
        }

        // Number pattern
        if (self.check(.number)) {
            const tok = self.advance();
            const pat = try self.allocator.create(ast.Pattern);
            pat.* = .{
                .kind = .{ .number = tok.number_value orelse .{ .unsigned = 0 } },
                .span = tok.span,
            };
            return pat;
        }

        // Tuple pattern
        if (self.match(.lparen)) {
            var elements: ArrayList(*ast.Pattern) = .empty;
            while (!self.check(.rparen) and !self.check(.eof)) {
                try elements.append(self.allocator, try self.parsePattern());
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.rparen);
            const pat = try self.allocator.create(ast.Pattern);
            pat.* = .{
                .kind = .{ .tuple = try elements.toOwnedSlice(self.allocator) },
                .span = ast.Span.merge(start, self.currentSpan()),
            };
            return pat;
        }

        try self.errors.append(self.allocator, .{
            .message = "expected pattern",
            .span = start,
        });
        return ParseError.ExpectedPattern;
    }

    fn parsePatternField(self: *Parser) ParseError!ast.PatternField {
        const start = self.currentSpan();

        // Check for named field: name: pattern or just pattern
        if (self.check(.ident)) {
            const saved = self.pos;
            const name_tok = self.advance();
            if (self.match(.colon)) {
                // Named field
                const pattern = try self.parsePattern();
                return .{
                    .name = name_tok.text,
                    .pattern = pattern,
                    .span = ast.Span.merge(start, pattern.span),
                };
            } else {
                // Just a variable pattern with same name
                self.pos = saved;
            }
        }

        const pattern = try self.parsePattern();
        return .{
            .name = null,
            .pattern = pattern,
            .span = pattern.span,
        };
    }

    // =========================================================================
    // Type Expression Parsing
    // =========================================================================

    fn parseTypeExpr(self: *Parser) ParseError!*ast.TypeExpr {
        return self.parseTypeFunction();
    }

    fn parseTypeFunction(self: *Parser) ParseError!*ast.TypeExpr {
        var left = try self.parseTypePrimary();

        while (self.match(.arrow)) {
            const right = try self.parseTypeFunction();
            left = try self.builder.typeFunction(left, right, ast.Span.merge(left.span, right.span));
        }

        return left;
    }

    fn parseTypePrimary(self: *Parser) ParseError!*ast.TypeExpr {
        const start = self.currentSpan();

        // Primitives
        if (self.check(.ident)) {
            const tok = self.peek();
            if (std.mem.eql(u8, tok.text, "u24")) {
                _ = self.advance();
                return self.builder.typePrimitive(.u24, tok.span);
            }
            if (std.mem.eql(u8, tok.text, "i24")) {
                _ = self.advance();
                return self.builder.typePrimitive(.i24, tok.span);
            }
            if (std.mem.eql(u8, tok.text, "f24")) {
                _ = self.advance();
                return self.builder.typePrimitive(.f24, tok.span);
            }
            // Type variable
            _ = self.advance();
            return self.builder.typeVar(tok.text, tok.span);
        }

        // Named type
        if (self.check(.upper_ident)) {
            const tok = self.advance();
            var type_expr = try self.builder.typeNamed(tok.text, tok.span);

            // Type application
            if (self.match(.lparen)) {
                var args: ArrayList(*ast.TypeExpr) = .empty;
                while (!self.check(.rparen) and !self.check(.eof)) {
                    try args.append(self.allocator, try self.parseTypeExpr());
                    if (!self.match(.comma)) break;
                }
                _ = try self.expect(.rparen);

                const app = try self.allocator.create(ast.TypeExpr);
                app.* = .{
                    .kind = .{ .app = .{ .func = type_expr, .args = try args.toOwnedSlice(self.allocator) } },
                    .span = ast.Span.merge(start, self.currentSpan()),
                };
                type_expr = app;
            }

            return type_expr;
        }

        // Tuple type
        if (self.match(.lparen)) {
            var elements: ArrayList(*ast.TypeExpr) = .empty;
            while (!self.check(.rparen) and !self.check(.eof)) {
                try elements.append(self.allocator, try self.parseTypeExpr());
                if (!self.match(.comma)) break;
            }
            _ = try self.expect(.rparen);

            const ty = try self.allocator.create(ast.TypeExpr);
            ty.* = .{
                .kind = .{ .tuple = try elements.toOwnedSlice(self.allocator) },
                .span = ast.Span.merge(start, self.currentSpan()),
            };
            return ty;
        }

        // Hole
        if (self.match(.question)) {
            const ty = try self.allocator.create(ast.TypeExpr);
            ty.* = .{ .kind = .hole, .span = start };
            return ty;
        }

        try self.errors.append(self.allocator, .{
            .message = "expected type",
            .span = start,
        });
        return ParseError.ExpectedType;
    }

    // =========================================================================
    // Utilities
    // =========================================================================

    fn exprToPattern(self: *Parser, expr: *ast.Expr) ParseError!*ast.Pattern {
        const pat = try self.allocator.create(ast.Pattern);
        pat.span = expr.span;

        switch (expr.kind) {
            .var_ => |name| {
                pat.kind = .{ .var_ = name };
            },
            .constructor => |name| {
                pat.kind = .{ .constructor = .{ .name = name, .fields = &[_]ast.PatternField{} } };
            },
            .tuple => |elements| {
                var pats = try self.allocator.alloc(*ast.Pattern, elements.len);
                for (elements, 0..) |e, i| {
                    pats[i] = try self.exprToPattern(e);
                }
                pat.kind = .{ .tuple = pats };
            },
            else => {
                pat.kind = .wildcard;
            },
        }

        return pat;
    }
};

// =============================================================================
// Public API
// =============================================================================

pub fn parse(source: []const u8, allocator: Allocator) !ast.Program {
    var lex = lexer.Lexer.init(source, allocator);
    defer lex.deinit();

    var tokens = try lex.tokenize();
    defer tokens.deinit(allocator);

    // Create an arena for AST allocations - destroyed on Program.deinit()
    const arena = try allocator.create(std.heap.ArenaAllocator);
    arena.* = std.heap.ArenaAllocator.init(allocator);
    errdefer {
        arena.deinit();
        allocator.destroy(arena);
    }
    const arena_alloc = arena.allocator();

    var parser_inst = Parser.init(tokens.items, arena_alloc);
    // Don't defer parser_inst.deinit() - errors list is in arena

    var program = parser_inst.parseProgram() catch |err| {
        // Arena cleanup happens via errdefer
        return err;
    };
    program.arena = arena;
    return program;
}

// =============================================================================
// Tests
// =============================================================================

test "parse simple function" {
    const allocator = std.testing.allocator;
    const source = "def main(): return 42";
    var program = try parse(source, allocator);
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 1), program.funcs.items.len);
    try std.testing.expectEqualStrings("main", program.funcs.items[0].name);
}

test "parse type definition" {
    const allocator = std.testing.allocator;
    const source =
        \\type Option:
        \\  Some { value }
        \\  None
    ;
    var program = try parse(source, allocator);
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 1), program.types.items.len);
    try std.testing.expectEqualStrings("Option", program.types.items[0].name);
    try std.testing.expectEqual(@as(usize, 2), program.types.items[0].constructors.len);
}
