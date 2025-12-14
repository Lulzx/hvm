const std = @import("std");
const ast = @import("ast.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// Bend Lexer
// =============================================================================
// Tokenizes Bend source code with indentation tracking for layout-sensitive parsing

// =============================================================================
// Token Types
// =============================================================================

pub const TokenKind = enum {
    // Keywords
    kw_type,
    kw_object,
    kw_def,
    kw_let,
    kw_use,
    kw_match,
    kw_switch,
    kw_fold,
    kw_bend,
    kw_when,
    kw_if,
    kw_elif,
    kw_else,
    kw_then,
    kw_case,
    kw_return,
    kw_with,
    kw_open,
    kw_fork,
    kw_checked,
    kw_unchecked,
    kw_import,
    kw_from,
    kw_as,

    // Literals
    number,
    string,
    char_,
    symbol, // :name

    // Identifiers
    ident, // lowercase start: variable, function
    upper_ident, // uppercase start: type, constructor

    // Operators
    plus, // +
    minus, // -
    star, // *
    slash, // /
    percent, // %
    ampersand, // &
    pipe, // |
    caret, // ^
    tilde, // ~
    eq, // ==
    ne, // !=
    lt, // <
    le, // <=
    gt, // >
    ge, // >=
    shl, // <<
    shr, // >>
    land, // &&
    lor, // ||
    pow, // **

    // Assignment operators
    assign, // =
    plus_eq, // +=
    minus_eq, // -=
    star_eq, // *=
    slash_eq, // /=
    amp_eq, // &=
    pipe_eq, // |=
    caret_eq, // ^=
    at_eq, // @=

    // Arrows
    arrow, // ->
    fat_arrow, // =>
    bind_arrow, // <-

    // Delimiters
    lparen, // (
    rparen, // )
    lbrace, // {
    rbrace, // }
    lbracket, // [
    rbracket, // ]

    // Punctuation
    colon, // :
    comma, // ,
    dot, // .
    underscore, // _
    at, // @
    hash, // #
    question, // ?
    lambda, // λ or \
    backslash, // \

    // Layout
    newline,
    indent, // Increased indentation
    dedent, // Decreased indentation

    // Special
    eof,
    invalid,
};

pub const Token = struct {
    kind: TokenKind,
    text: []const u8,
    span: ast.Span,

    // For number tokens
    number_value: ?ast.Number = null,

    pub fn isKeyword(self: Token) bool {
        return @intFromEnum(self.kind) >= @intFromEnum(TokenKind.kw_type) and
            @intFromEnum(self.kind) <= @intFromEnum(TokenKind.kw_as);
    }

    pub fn isOperator(self: Token) bool {
        return @intFromEnum(self.kind) >= @intFromEnum(TokenKind.plus) and
            @intFromEnum(self.kind) <= @intFromEnum(TokenKind.pow);
    }
};

// =============================================================================
// Lexer State
// =============================================================================

pub const Lexer = struct {
    source: []const u8,
    pos: usize,
    line: u32,
    col: u32,
    line_start: usize,
    allocator: Allocator,

    // Indentation tracking
    indent_stack: ArrayList(u32),
    pending_dedents: u32,
    at_line_start: bool,
    current_indent: u32,

    // Error handling
    errors: ArrayList(LexError),

    pub const LexError = struct {
        message: []const u8,
        span: ast.Span,
    };

    pub fn init(source: []const u8, allocator: Allocator) Lexer {
        var indent_stack: ArrayList(u32) = .empty;
        indent_stack.append(allocator, 0) catch unreachable;

        return .{
            .source = source,
            .pos = 0,
            .line = 1,
            .col = 1,
            .line_start = 0,
            .allocator = allocator,
            .indent_stack = indent_stack,
            .pending_dedents = 0,
            .at_line_start = true,
            .current_indent = 0,
            .errors = .empty,
        };
    }

    pub fn deinit(self: *Lexer) void {
        self.indent_stack.deinit(self.allocator);
        self.errors.deinit(self.allocator);
    }

    // =========================================================================
    // Character Utilities
    // =========================================================================

    fn peek(self: *Lexer) ?u8 {
        if (self.pos >= self.source.len) return null;
        return self.source[self.pos];
    }

    fn peekN(self: *Lexer, n: usize) ?u8 {
        if (self.pos + n >= self.source.len) return null;
        return self.source[self.pos + n];
    }

    fn advance(self: *Lexer) ?u8 {
        if (self.pos >= self.source.len) return null;
        const c = self.source[self.pos];
        self.pos += 1;
        if (c == '\n') {
            self.line += 1;
            self.col = 1;
            self.line_start = self.pos;
        } else {
            self.col += 1;
        }
        return c;
    }

    fn currentSpan(self: *Lexer, start: usize, start_line: u32, start_col: u32) ast.Span {
        return .{
            .start = @intCast(start),
            .end = @intCast(self.pos),
            .line = start_line,
            .col = start_col,
        };
    }

    // =========================================================================
    // Tokenization
    // =========================================================================

    pub fn nextToken(self: *Lexer) Token {
        // Handle pending dedents
        if (self.pending_dedents > 0) {
            self.pending_dedents -= 1;
            return self.makeToken(.dedent, "");
        }

        // Handle indentation at line start
        if (self.at_line_start) {
            self.at_line_start = false;
            const indent = self.measureIndent();
            self.current_indent = indent;

            const top = self.indent_stack.items[self.indent_stack.items.len - 1];

            if (indent > top) {
                self.indent_stack.append(self.allocator, indent) catch {};
                return self.makeToken(.indent, "");
            } else if (indent < top) {
                // Pop indent levels and count dedents
                while (self.indent_stack.items.len > 1) {
                    const level = self.indent_stack.items[self.indent_stack.items.len - 1];
                    if (level <= indent) break;
                    _ = self.indent_stack.pop();
                    self.pending_dedents += 1;
                }
                if (self.pending_dedents > 0) {
                    self.pending_dedents -= 1;
                    return self.makeToken(.dedent, "");
                }
            }
        }

        // Skip whitespace (not newlines)
        self.skipWhitespace();

        // Check for EOF
        if (self.peek() == null) {
            // Emit remaining dedents
            if (self.indent_stack.items.len > 1) {
                _ = self.indent_stack.pop();
                return self.makeToken(.dedent, "");
            }
            return self.makeToken(.eof, "");
        }

        const start = self.pos;
        const start_line = self.line;
        const start_col = self.col;
        const c = self.advance().?;

        // Newline
        if (c == '\n') {
            self.at_line_start = true;
            return .{
                .kind = .newline,
                .text = "\n",
                .span = self.currentSpan(start, start_line, start_col),
            };
        }

        // Comments
        if (c == '#') {
            if (self.peek() == '{') {
                self.skipBlockComment();
                return self.nextToken();
            } else {
                self.skipLineComment();
                return self.nextToken();
            }
        }

        // String literal
        if (c == '"') {
            return self.scanString(start, start_line, start_col);
        }

        // Character literal
        if (c == '\'') {
            return self.scanChar(start, start_line, start_col);
        }

        // Symbol literal :name
        if (c == ':' and self.peek() != null and isIdentStart(self.peek().?)) {
            return self.scanSymbol(start, start_line, start_col);
        }

        // Number (only scan '-' as part of number at start of expression, not after identifiers)
        if (isDigit(c)) {
            return self.scanNumber(start, start_line, start_col);
        }

        // Lambda (λ or \)
        if (c == '\\') {
            return .{
                .kind = .lambda,
                .text = "\\",
                .span = self.currentSpan(start, start_line, start_col),
            };
        }

        // UTF-8 lambda (λ)
        if (c == 0xCE and self.peek() == 0xBB) {
            _ = self.advance();
            return .{
                .kind = .lambda,
                .text = "λ",
                .span = self.currentSpan(start, start_line, start_col),
            };
        }

        // Nat literal @n
        if (c == '@' and self.peek() != null and isDigit(self.peek().?)) {
            return self.scanNat(start, start_line, start_col);
        }

        // Check for standalone underscore (wildcard pattern)
        if (c == '_') {
            // If followed by identifier char, it's part of an identifier
            if (self.peek()) |next| {
                if (isIdentChar(next)) {
                    return self.scanIdentifier(start, start_line, start_col, c);
                }
            }
            // Otherwise it's a standalone underscore
            return .{
                .kind = .underscore,
                .text = "_",
                .span = self.currentSpan(start, start_line, start_col),
            };
        }

        // Identifier or keyword
        if (isIdentStart(c)) {
            return self.scanIdentifier(start, start_line, start_col, c);
        }

        // Operators and punctuation
        return self.scanOperator(start, start_line, start_col, c);
    }

    fn makeToken(self: *Lexer, kind: TokenKind, text: []const u8) Token {
        return .{
            .kind = kind,
            .text = text,
            .span = .{
                .start = @intCast(self.pos),
                .end = @intCast(self.pos),
                .line = self.line,
                .col = self.col,
            },
        };
    }

    fn measureIndent(self: *Lexer) u32 {
        var indent: u32 = 0;
        while (self.peek()) |c| {
            if (c == ' ') {
                indent += 1;
                _ = self.advance();
            } else if (c == '\t') {
                indent += 4; // Tabs are 4 spaces
                _ = self.advance();
            } else {
                break;
            }
        }
        return indent;
    }

    fn skipWhitespace(self: *Lexer) void {
        while (self.peek()) |c| {
            if (c == ' ' or c == '\t' or c == '\r') {
                _ = self.advance();
            } else {
                break;
            }
        }
    }

    fn skipLineComment(self: *Lexer) void {
        while (self.peek()) |c| {
            if (c == '\n') break;
            _ = self.advance();
        }
    }

    fn skipBlockComment(self: *Lexer) void {
        _ = self.advance(); // Skip {
        var depth: u32 = 1;
        while (depth > 0) {
            const c = self.advance() orelse break;
            if (c == '#' and self.peek() == '{') {
                _ = self.advance();
                depth += 1;
            } else if (c == '#' and self.peek() == '}') {
                _ = self.advance();
                depth -= 1;
            }
        }
    }

    fn scanString(self: *Lexer, start: usize, start_line: u32, start_col: u32) Token {
        while (self.peek()) |c| {
            if (c == '"') {
                _ = self.advance();
                break;
            }
            if (c == '\\') {
                _ = self.advance();
                _ = self.advance(); // Skip escaped char
            } else {
                _ = self.advance();
            }
        }
        return .{
            .kind = .string,
            .text = self.source[start..self.pos],
            .span = self.currentSpan(start, start_line, start_col),
        };
    }

    fn scanChar(self: *Lexer, start: usize, start_line: u32, start_col: u32) Token {
        if (self.peek() == '\\') {
            _ = self.advance();
            _ = self.advance(); // Escaped char
        } else {
            _ = self.advance();
        }
        if (self.peek() == '\'') {
            _ = self.advance();
        }
        return .{
            .kind = .char_,
            .text = self.source[start..self.pos],
            .span = self.currentSpan(start, start_line, start_col),
        };
    }

    fn scanSymbol(self: *Lexer, start: usize, start_line: u32, start_col: u32) Token {
        while (self.peek()) |c| {
            if (isIdentChar(c)) {
                _ = self.advance();
            } else {
                break;
            }
        }
        return .{
            .kind = .symbol,
            .text = self.source[start..self.pos],
            .span = self.currentSpan(start, start_line, start_col),
        };
    }

    fn scanNumber(self: *Lexer, start: usize, start_line: u32, start_col: u32) Token {
        var is_float = false;
        const is_negative = self.source[start] == '-';
        _ = is_negative;

        // Integer part
        while (self.peek()) |c| {
            if (isDigit(c)) {
                _ = self.advance();
            } else {
                break;
            }
        }

        // Decimal part
        if (self.peek() == '.' and self.peekN(1) != null and isDigit(self.peekN(1).?)) {
            is_float = true;
            _ = self.advance(); // .
            while (self.peek()) |c| {
                if (isDigit(c)) {
                    _ = self.advance();
                } else {
                    break;
                }
            }
        }

        const text = self.source[start..self.pos];
        var token = Token{
            .kind = .number,
            .text = text,
            .span = self.currentSpan(start, start_line, start_col),
        };

        // Parse number value
        if (is_float) {
            if (std.fmt.parseFloat(f32, text)) |f| {
                token.number_value = .{ .float = f };
            } else |_| {}
        } else if (text[0] == '-') {
            if (std.fmt.parseInt(i32, text, 10)) |i| {
                token.number_value = .{ .signed = i };
            } else |_| {}
        } else {
            if (std.fmt.parseInt(u32, text, 10)) |u| {
                token.number_value = .{ .unsigned = u };
            } else |_| {}
        }

        return token;
    }

    fn scanNat(self: *Lexer, start: usize, start_line: u32, start_col: u32) Token {
        while (self.peek()) |c| {
            if (isDigit(c)) {
                _ = self.advance();
            } else {
                break;
            }
        }
        const text = self.source[start..self.pos];
        var token = Token{
            .kind = .number,
            .text = text,
            .span = self.currentSpan(start, start_line, start_col),
        };
        // Parse the nat value (skip @)
        if (std.fmt.parseInt(u32, text[1..], 10)) |u| {
            token.number_value = .{ .unsigned = u };
        } else |_| {}
        return token;
    }

    fn scanIdentifier(self: *Lexer, start: usize, start_line: u32, start_col: u32, first: u8) Token {
        const is_upper = first >= 'A' and first <= 'Z';

        while (self.peek()) |c| {
            // Don't include '-' if followed by a digit (it's subtraction)
            if (c == '-') {
                if (self.peekN(1)) |next| {
                    if (isDigit(next)) break;
                }
            }
            if (isIdentChar(c) or c == '/') {
                _ = self.advance();
            } else {
                break;
            }
        }

        const text = self.source[start..self.pos];
        const kind = self.keywordOrIdent(text, is_upper);

        return .{
            .kind = kind,
            .text = text,
            .span = self.currentSpan(start, start_line, start_col),
        };
    }

    fn keywordOrIdent(self: *Lexer, text: []const u8, is_upper: bool) TokenKind {
        _ = self;
        const keywords = std.StaticStringMap(TokenKind).initComptime(.{
            .{ "type", .kw_type },
            .{ "object", .kw_object },
            .{ "def", .kw_def },
            .{ "let", .kw_let },
            .{ "use", .kw_use },
            .{ "match", .kw_match },
            .{ "switch", .kw_switch },
            .{ "fold", .kw_fold },
            .{ "bend", .kw_bend },
            .{ "when", .kw_when },
            .{ "if", .kw_if },
            .{ "elif", .kw_elif },
            .{ "else", .kw_else },
            .{ "then", .kw_then },
            .{ "case", .kw_case },
            .{ "return", .kw_return },
            .{ "with", .kw_with },
            .{ "open", .kw_open },
            .{ "fork", .kw_fork },
            .{ "checked", .kw_checked },
            .{ "unchecked", .kw_unchecked },
            .{ "import", .kw_import },
            .{ "from", .kw_from },
            .{ "as", .kw_as },
        });

        if (keywords.get(text)) |kw| {
            return kw;
        }

        return if (is_upper) .upper_ident else .ident;
    }

    fn scanOperator(self: *Lexer, start: usize, start_line: u32, start_col: u32, c: u8) Token {
        const next = self.peek();

        const kind: TokenKind = switch (c) {
            '+' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .plus_eq;
            } else .plus,
            '-' => if (next == '>') blk: {
                _ = self.advance();
                break :blk .arrow;
            } else if (next == '=') blk: {
                _ = self.advance();
                break :blk .minus_eq;
            } else .minus,
            '*' => if (next == '*') blk: {
                _ = self.advance();
                break :blk .pow;
            } else if (next == '=') blk: {
                _ = self.advance();
                break :blk .star_eq;
            } else .star,
            '/' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .slash_eq;
            } else .slash,
            '%' => .percent,
            '&' => if (next == '&') blk: {
                _ = self.advance();
                break :blk .land;
            } else if (next == '=') blk: {
                _ = self.advance();
                break :blk .amp_eq;
            } else .ampersand,
            '|' => if (next == '|') blk: {
                _ = self.advance();
                break :blk .lor;
            } else if (next == '=') blk: {
                _ = self.advance();
                break :blk .pipe_eq;
            } else .pipe,
            '^' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .caret_eq;
            } else .caret,
            '~' => .tilde,
            '=' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .eq;
            } else if (next == '>') blk: {
                _ = self.advance();
                break :blk .fat_arrow;
            } else .assign,
            '!' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .ne;
            } else .invalid, // ! alone is not valid
            '<' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .le;
            } else if (next == '<') blk: {
                _ = self.advance();
                break :blk .shl;
            } else if (next == '-') blk: {
                _ = self.advance();
                break :blk .bind_arrow;
            } else .lt,
            '>' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .ge;
            } else if (next == '>') blk: {
                _ = self.advance();
                break :blk .shr;
            } else .gt,
            '(' => .lparen,
            ')' => .rparen,
            '{' => .lbrace,
            '}' => .rbrace,
            '[' => .lbracket,
            ']' => .rbracket,
            ':' => .colon,
            ',' => .comma,
            '.' => .dot,
            '_' => .underscore,
            '@' => if (next == '=') blk: {
                _ = self.advance();
                break :blk .at_eq;
            } else .at,
            '#' => .hash,
            '?' => .question,
            else => .invalid,
        };

        return .{
            .kind = kind,
            .text = self.source[start..self.pos],
            .span = self.currentSpan(start, start_line, start_col),
        };
    }

    // =========================================================================
    // Tokenize All
    // =========================================================================

    pub fn tokenize(self: *Lexer) !ArrayList(Token) {
        var tokens: ArrayList(Token) = .empty;
        while (true) {
            const token = self.nextToken();
            try tokens.append(self.allocator, token);
            if (token.kind == .eof) break;
        }
        return tokens;
    }
};

// =============================================================================
// Character Classification
// =============================================================================

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isIdentStart(c: u8) bool {
    return (c >= 'a' and c <= 'z') or
        (c >= 'A' and c <= 'Z') or
        c == '_';
}

fn isIdentChar(c: u8) bool {
    return isIdentStart(c) or isDigit(c) or c == '-' or c == '.';
}

// =============================================================================
// Tests
// =============================================================================

test "lexer basic tokens" {
    const allocator = std.testing.allocator;
    var lexer = Lexer.init("def main():", allocator);
    defer lexer.deinit();

    const t1 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.kw_def, t1.kind);

    const t2 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.ident, t2.kind);
    try std.testing.expectEqualStrings("main", t2.text);

    const t3 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.lparen, t3.kind);

    const t4 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.rparen, t4.kind);

    const t5 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.colon, t5.kind);
}

test "lexer indentation" {
    const allocator = std.testing.allocator;
    const source =
        \\def main():
        \\  x = 1
        \\  y = 2
        \\
    ;
    var lexer = Lexer.init(source, allocator);
    defer lexer.deinit();

    var tokens = try lexer.tokenize();
    defer tokens.deinit();

    // Check that we get indent after the first newline
    var found_indent = false;
    for (tokens.items) |tok| {
        if (tok.kind == .indent) found_indent = true;
    }
    try std.testing.expect(found_indent);
}

test "lexer numbers" {
    const allocator = std.testing.allocator;
    var lexer = Lexer.init("42 -7 3.14", allocator);
    defer lexer.deinit();

    const t1 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.number, t1.kind);
    try std.testing.expectEqual(@as(u32, 42), t1.number_value.?.unsigned);

    const t2 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.number, t2.kind);
    try std.testing.expectEqual(@as(i32, -7), t2.number_value.?.signed);

    const t3 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.number, t3.kind);
}

test "lexer constructors" {
    const allocator = std.testing.allocator;
    var lexer = Lexer.init("List/Cons Option/Some", allocator);
    defer lexer.deinit();

    const t1 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.upper_ident, t1.kind);
    try std.testing.expectEqualStrings("List/Cons", t1.text);

    const t2 = lexer.nextToken();
    try std.testing.expectEqual(TokenKind.upper_ident, t2.kind);
    try std.testing.expectEqualStrings("Option/Some", t2.text);
}
