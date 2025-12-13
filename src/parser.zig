const std = @import("std");
const hvm = @import("hvm.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// =============================================================================
// Parser Error
// =============================================================================

pub const ParseError = error{
    UnexpectedChar,
    UnexpectedEof,
    ExpectedChar,
    InvalidNumber,
    InvalidOperator,
    UndefinedVariable,
    UndefinedFunction,
    OutOfMemory,
};

// =============================================================================
// Function Definition
// =============================================================================

pub const Func = struct {
    name: []const u8,
    params: [][]const u8,
    body_text: []const u8, // Store text for lazy compilation
    fid: u16,
};

// =============================================================================
// Book (Function Registry)
// =============================================================================

pub const Book = struct {
    funcs: StringHashMap(Func),
    next_fid: u16,
    allocator: Allocator,

    pub fn init(allocator: Allocator) Book {
        return .{
            .funcs = StringHashMap(Func).init(allocator),
            .next_fid = 0x10, // Reserve 0x00-0x0F for builtins
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Book) void {
        self.funcs.deinit();
    }

    pub fn define(self: *Book, name: []const u8, params: [][]const u8, body_text: []const u8) !u16 {
        const fid = self.next_fid;
        self.next_fid += 1;
        try self.funcs.put(name, .{
            .name = name,
            .params = params,
            .body_text = body_text,
            .fid = fid,
        });
        return fid;
    }

    pub fn get(self: *Book, name: []const u8) ?Func {
        return self.funcs.get(name);
    }
};

// =============================================================================
// Parser
// =============================================================================

/// Variable binding info with linearity tracking
pub const VarBinding = struct {
    loc: u64,
    linearity: hvm.Linearity,
    label: ?hvm.Ext,
    usage_count: u8,
};

pub const Parser = struct {
    text: []const u8,
    pos: usize,
    allocator: Allocator,
    book: *Book,
    vars: StringHashMap(u64), // Variable name -> heap location (legacy)
    var_bindings: StringHashMap(VarBinding), // Variable name -> full binding info
    var_counter: u64,
    linearity_errors: ArrayList(hvm.LinearityError),

    pub fn init(text: []const u8, allocator: Allocator, book: *Book) Parser {
        return .{
            .text = text,
            .pos = 0,
            .allocator = allocator,
            .book = book,
            .vars = StringHashMap(u64).init(allocator),
            .var_bindings = StringHashMap(VarBinding).init(allocator),
            .var_counter = 0,
            .linearity_errors = ArrayList(hvm.LinearityError).empty,
        };
    }

    pub fn deinit(self: *Parser) void {
        self.vars.deinit();
        self.var_bindings.deinit();
        self.linearity_errors.deinit(self.allocator);
    }

    /// Check if any linearity errors occurred during parsing
    pub fn hasLinearityErrors(self: *Parser) bool {
        return self.linearity_errors.items.len > 0;
    }

    /// Get linearity errors for reporting
    pub fn getLinearityErrors(self: *Parser) []hvm.LinearityError {
        return self.linearity_errors.items;
    }

    // Skip whitespace and comments
    fn skip(self: *Parser) void {
        while (self.pos < self.text.len) {
            const c = self.text[self.pos];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') {
                self.pos += 1;
            } else if (c == '/' and self.pos + 1 < self.text.len and self.text[self.pos + 1] == '/') {
                // Line comment
                while (self.pos < self.text.len and self.text[self.pos] != '\n') {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn peek(self: *Parser) ?u8 {
        self.skip();
        return if (self.pos < self.text.len) self.text[self.pos] else null;
    }

    fn advance(self: *Parser) ?u8 {
        if (self.pos < self.text.len) {
            const c = self.text[self.pos];
            self.pos += 1;
            return c;
        }
        return null;
    }

    fn consume(self: *Parser, expected: u8) ParseError!void {
        self.skip();
        if (self.pos >= self.text.len) return ParseError.UnexpectedEof;
        if (self.text[self.pos] != expected) return ParseError.ExpectedChar;
        self.pos += 1;
    }

    fn consumeStr(self: *Parser, expected: []const u8) ParseError!void {
        self.skip();
        if (self.pos + expected.len > self.text.len) return ParseError.UnexpectedEof;
        if (!std.mem.eql(u8, self.text[self.pos .. self.pos + expected.len], expected)) {
            return ParseError.ExpectedChar;
        }
        self.pos += expected.len;
    }

    fn isNameChar(c: u8) bool {
        return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or c == '_' or c == '$';
    }

    fn isNameStart(c: u8) bool {
        return (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c == '_' or c == '$';
    }

    fn name(self: *Parser) ParseError![]const u8 {
        self.skip();
        const start = self.pos;
        if (self.pos >= self.text.len or !isNameStart(self.text[self.pos])) {
            return ParseError.UnexpectedChar;
        }
        while (self.pos < self.text.len and isNameChar(self.text[self.pos])) {
            self.pos += 1;
        }
        return self.text[start..self.pos];
    }

    fn integer(self: *Parser) ParseError!u32 {
        self.skip();
        var negative = false;
        if (self.pos < self.text.len and self.text[self.pos] == '-') {
            negative = true;
            self.pos += 1;
        }
        if (self.pos >= self.text.len or self.text[self.pos] < '0' or self.text[self.pos] > '9') {
            return ParseError.InvalidNumber;
        }
        var val: u32 = 0;
        while (self.pos < self.text.len and self.text[self.pos] >= '0' and self.text[self.pos] <= '9') {
            val = val * 10 + (self.text[self.pos] - '0');
            self.pos += 1;
        }
        if (negative) {
            return @bitCast(-@as(i32, @intCast(val)));
        }
        return val;
    }

    // Special operator for structural equality
    const STRUCT_EQ: hvm.Ext = 0xFF;

    fn parseOp(self: *Parser) ?hvm.Ext {
        self.skip();
        if (self.pos >= self.text.len) return null;

        // Check for three-char operators first (===)
        if (self.pos + 2 < self.text.len) {
            const three = self.text[self.pos .. self.pos + 3];
            if (std.mem.eql(u8, three, "===")) {
                self.pos += 3;
                return STRUCT_EQ; // Structural equality marker
            }
        }

        // Check for two-char operators using central table
        if (self.pos + 1 < self.text.len) {
            const two = self.text[self.pos .. self.pos + 2];
            if (hvm.op_from_symbol(two)) |op| {
                self.pos += 2;
                return op;
            }
        }

        // Check for single-char operators using central table
        const c = self.text[self.pos];
        const single = &[_]u8{c};
        if (hvm.op_from_symbol(single)) |op| {
            self.pos += 1;
            return op;
        }

        return null;
    }

    // Bind a variable name to a heap location
    fn bindVar(self: *Parser, var_name: []const u8, loc: u64) !void {
        try self.vars.put(var_name, loc);
        // Also add to bindings with unrestricted linearity for compatibility
        try self.var_bindings.put(var_name, .{
            .loc = loc,
            .linearity = .unrestricted,
            .label = null,
            .usage_count = 0,
        });
    }

    // Bind a variable with linearity tracking
    fn bindVarWithLinearity(self: *Parser, var_name: []const u8, loc: u64, linearity: hvm.Linearity, label: ?hvm.Ext) !void {
        try self.vars.put(var_name, loc);
        try self.var_bindings.put(var_name, .{
            .loc = loc,
            .linearity = linearity,
            .label = label,
            .usage_count = 0,
        });
    }

    // Look up a variable and track usage for linearity checking
    fn lookupVar(self: *Parser, var_name: []const u8) ?u64 {
        return self.vars.get(var_name);
    }

    // Look up a variable and increment usage count (for linearity checking)
    fn useVar(self: *Parser, var_name: []const u8) !?u64 {
        if (self.var_bindings.getPtr(var_name)) |binding| {
            binding.usage_count += 1;

            // Check for linearity violations
            if (binding.linearity == .linear and binding.usage_count > 1) {
                try self.linearity_errors.append(self.allocator, .{
                    .kind = .linear_used_twice,
                    .location = @truncate(binding.loc),
                    .message = "Linear variable used more than once",
                });
            } else if (binding.linearity == .affine and binding.usage_count > 1) {
                try self.linearity_errors.append(self.allocator, .{
                    .kind = .affine_used_twice,
                    .location = @truncate(binding.loc),
                    .message = "Affine variable used more than once",
                });
            }

            return binding.loc;
        }
        return self.vars.get(var_name);
    }

    // Parse a term and allocate it on the heap
    pub fn term(self: *Parser) ParseError!hvm.Term {
        const c = self.peek() orelse return ParseError.UnexpectedEof;

        // Erasure: *
        if (c == '*') {
            self.pos += 1;
            return hvm.term_new(hvm.ERA, 0, 0);
        }

        // Type annotation: {term : Type}
        if (c == '{') {
            self.pos += 1;
            const inner_term = try self.term();
            try self.consume(':');
            const typ = try self.term();
            try self.consume('}');

            const loc = hvm.alloc_node(2);
            hvm.set(loc, inner_term);
            hvm.set(loc + 1, typ);
            return hvm.term_new(hvm.ANN, 0, @truncate(loc));
        }

        // Type universe: Type
        if (c == 'T' and self.pos + 4 <= self.text.len) {
            if (std.mem.eql(u8, self.text[self.pos .. self.pos + 4], "Type")) {
                // Make sure it's not part of a longer name
                if (self.pos + 4 >= self.text.len or !isNameChar(self.text[self.pos + 4])) {
                    self.pos += 4;
                    return hvm.term_new(hvm.TYP, 0, 0);
                }
            }
        }

        // Number or Constructor: #123 or #Name{fields}
        if (c == '#') {
            self.pos += 1;
            const next = self.peek() orelse return ParseError.UnexpectedEof;

            // Constructor: #Name{fields}
            if (isNameStart(next)) {
                const ctr_name = try self.name();
                _ = ctr_name; // TODO: look up constructor ID

                var fields: ArrayList(hvm.Term) = .empty;
                defer fields.deinit(self.allocator);

                if (self.peek() == @as(u8, '{')) {
                    try self.consume('{');
                    while (self.peek()) |fc| {
                        if (fc == '}') break;
                        try fields.append(self.allocator, try self.term());
                    }
                    try self.consume('}');
                }

                // Allocate constructor on heap - use C00-C15 based on arity
                const arity = fields.items.len;
                const ctr_tag: hvm.Tag = @truncate(hvm.C00 + @min(arity, 15));
                const loc = hvm.alloc_node(arity);
                for (fields.items, 0..) |field, i| {
                    hvm.set(loc + i, field);
                }
                return hvm.term_new(ctr_tag, 0, @truncate(loc));
            }

            // Number: #123
            const val = try self.integer();
            return hvm.term_new(hvm.NUM, 0, val);
        }

        // Character: 'c' (characters are numbers in HVM4)
        if (c == '\'') {
            self.pos += 1;
            const chr = self.advance() orelse return ParseError.UnexpectedEof;
            try self.consume('\'');
            return hvm.term_new(hvm.NUM, 0, chr);
        }

        // Lambda: λx.body or \x.body
        // Linear lambda: \!x.body (x must be used exactly once)
        // Affine lambda: \?x.body (x may be used at most once)
        // Label-scoped linear: \!L:x.body (safe with SUP of label L)
        if (c == '\\' or c == 0xCE) { // 0xCE is first byte of λ in UTF-8
            if (c == 0xCE) {
                self.pos += 2; // Skip λ (2 bytes in UTF-8)
            } else {
                self.pos += 1;
            }

            // Check for linearity modifier
            var linearity: hvm.Linearity = .unrestricted;
            var label_bound: ?hvm.Ext = null;

            if (self.peek()) |modifier| {
                if (modifier == '!') {
                    self.pos += 1;
                    linearity = .linear;
                    // Check for label: \!L:x.body
                    if (self.peek()) |pc| {
                        if (pc >= '0' and pc <= '9') {
                            label_bound = @truncate(try self.integer());
                            try self.consume(':');
                        }
                    }
                } else if (modifier == '?') {
                    self.pos += 1;
                    linearity = .affine;
                }
            }

            const var_name = try self.name();
            try self.consume('.');

            // Allocate lambda node
            const lam_loc = hvm.alloc_node(1);

            // Bind variable to lambda location (for VAR to point back)
            // Store linearity info in the binding for later checking
            try self.bindVarWithLinearity(var_name, lam_loc, linearity, label_bound);

            // Parse body
            const body = try self.term();

            // Set lambda body
            hvm.set(lam_loc, body);

            // For linear/affine lambdas, wrap the result
            const lam_term = hvm.term_new(hvm.LAM, 0, @truncate(lam_loc));

            if (linearity == .linear) {
                return if (label_bound) |lab|
                    hvm.linear_labeled(lam_term, lab)
                else
                    hvm.linear(lam_term);
            } else if (linearity == .affine) {
                return hvm.affine(lam_term);
            }

            return lam_term;
        }

        // Function reference: @name(args...)
        if (c == '@') {
            self.pos += 1;
            const ref_name = try self.name();

            // Check for builtin functions
            if (std.mem.eql(u8, ref_name, "SUP")) {
                return self.parseBuiltinSUP();
            }
            if (std.mem.eql(u8, ref_name, "DUP")) {
                return self.parseBuiltinDUP();
            }
            if (std.mem.eql(u8, ref_name, "LOG")) {
                return self.parseBuiltinLOG();
            }

            var args: ArrayList(hvm.Term) = .empty;
            defer args.deinit(self.allocator);

            if (self.peek() == @as(u8, '(')) {
                try self.consume('(');
                while (self.peek()) |ac| {
                    if (ac == ')') break;
                    try args.append(self.allocator, try self.term());
                }
                try self.consume(')');
            }

            // Look up function
            if (self.book.get(ref_name)) |func| {
                // Allocate arguments on heap
                const loc = hvm.alloc_node(args.items.len);
                for (args.items, 0..) |arg, i| {
                    hvm.set(loc + i, arg);
                }
                hvm.hvm_get_state().fun_arity[func.fid] = @intCast(args.items.len);
                return hvm.term_new(hvm.REF, func.fid, @truncate(loc));
            } else {
                return ParseError.UndefinedFunction;
            }
        }

        // Pattern match: ~x { cases... }
        if (c == '~') {
            self.pos += 1;
            const val = try self.term();
            try self.consume('{');

            var cases: ArrayList(hvm.Term) = .empty;
            defer cases.deinit(self.allocator);

            while (self.peek()) |cc| {
                if (cc == '}') break;
                try self.consume('#');
                const case_name = try self.name();
                _ = case_name;

                // Parse case variables
                var case_vars: ArrayList([]const u8) = .empty;
                defer case_vars.deinit(self.allocator);

                if (self.peek() == @as(u8, '{')) {
                    try self.consume('{');
                    while (self.peek()) |vc| {
                        if (vc == '}') break;
                        try case_vars.append(self.allocator, try self.name());
                    }
                    try self.consume('}');
                }

                try self.consume(':');

                // Bind case variables
                for (case_vars.items) |cv| {
                    const cv_loc = hvm.alloc_node(1);
                    try self.bindVar(cv, cv_loc);
                }

                const case_body = try self.term();
                try cases.append(self.allocator, case_body);
            }
            try self.consume('}');

            // Allocate MAT node: [scrutinee, case0, case1, ...]
            const loc = hvm.alloc_node(1 + cases.items.len);
            hvm.set(loc, val);
            for (cases.items, 0..) |case_term, i| {
                hvm.set(loc + 1 + i, case_term);
            }

            return hvm.term_new(hvm.MAT, @intCast(cases.items.len), @truncate(loc));
        }

        // Parentheses: operator, switch, or application
        if (c == '(') {
            try self.consume('(');
            const c2 = self.peek() orelse return ParseError.UnexpectedEof;

            // Switch: (?n zero succ)
            if (c2 == '?') {
                self.pos += 1;
                const n = try self.term();
                const zero = try self.term();
                const succ = try self.term();
                try self.consume(')');

                // Allocate SWI node: [scrutinee, zero_case, succ_case]
                const loc = hvm.alloc_node(3);
                hvm.set(loc, n);
                hvm.set(loc + 1, zero);
                hvm.set(loc + 2, succ);

                return hvm.term_new(hvm.SWI, 2, @truncate(loc));
            }

            // Check for binary operator
            if (self.parseOp()) |op| {
                const a = try self.term();
                const b = try self.term();
                try self.consume(')');

                // Allocate binary node
                const loc = hvm.alloc_node(2);
                hvm.set(loc, a);
                hvm.set(loc + 1, b);

                // Structural equality uses EQL tag
                if (op == STRUCT_EQ) {
                    return hvm.term_new(hvm.EQL, 0, @truncate(loc));
                }

                // Other operators use P02 (arity 2 primitive)
                return hvm.term_new(hvm.P02, op, @truncate(loc));
            }

            // Application: (f x y z)
            var fun = try self.term();
            while (self.peek()) |pc| {
                if (pc == ')') break;
                const arg = try self.term();

                // Allocate APP node
                const loc = hvm.alloc_node(2);
                hvm.set(loc, fun);
                hvm.set(loc + 1, arg);
                fun = hvm.term_new(hvm.APP, 0, @truncate(loc));
            }
            try self.consume(')');
            return fun;
        }

        // Superposition: &L{a, b}
        if (c == '&') {
            self.pos += 1;
            const lab: hvm.Ext = if (self.peek()) |pc|
                (if (pc >= '0' and pc <= '9') @truncate(try self.integer()) else 0)
            else
                0;
            try self.consume('{');
            const a = try self.term();
            try self.consume(',');
            const b = try self.term();
            try self.consume('}');

            // Allocate SUP node
            const loc = hvm.alloc_node(2);
            hvm.set(loc, a);
            hvm.set(loc + 1, b);

            return hvm.term_new(hvm.SUP, lab, @truncate(loc));
        }

        // Duplication: !&L{x,y}=val;body
        if (c == '!') {
            self.pos += 1;
            try self.consume('&');
            const lab: hvm.Ext = if (self.peek()) |pc|
                (if (pc >= '0' and pc <= '9') @truncate(try self.integer()) else 0)
            else
                0;
            try self.consume('{');
            const x = try self.name();
            try self.consume(',');
            const y = try self.name();
            try self.consume('}');
            try self.consume('=');
            const val = try self.term();
            try self.consume(';');

            // Allocate DUP node - stores the value to be duplicated
            const dup_loc = hvm.alloc_node(1);
            hvm.set(dup_loc, val);

            // Bind x to CO0, y to CO1
            try self.bindVar(x, dup_loc);
            try self.bindVar(y, dup_loc);

            // Store which projection each variable uses
            // We'll handle this by creating the CO0/CO1 terms when variables are used

            // Parse body with bindings
            const body = try self.termWithDupBindings(x, y, lab, dup_loc);

            return body;
        }

        // Variable
        if (isNameStart(c)) {
            const var_name = try self.name();
            // Use useVar to track linearity
            if (try self.useVar(var_name)) |loc| {
                return hvm.term_new(hvm.VAR, 0, @truncate(loc));
            }
            return ParseError.UndefinedVariable;
        }

        return ParseError.UnexpectedChar;
    }

    // Parse term with dup bindings - replaces x with CO0 and y with CO1
    fn termWithDupBindings(self: *Parser, x: []const u8, y: []const u8, lab: hvm.Ext, dup_loc: u64) ParseError!hvm.Term {
        const c = self.peek() orelse return ParseError.UnexpectedEof;

        // Check if it's a variable that matches x or y
        if (isNameStart(c)) {
            const save_pos = self.pos;
            const var_name = try self.name();

            if (std.mem.eql(u8, var_name, x)) {
                return hvm.term_new(hvm.CO0, lab, @truncate(dup_loc));
            }
            if (std.mem.eql(u8, var_name, y)) {
                return hvm.term_new(hvm.CO1, lab, @truncate(dup_loc));
            }

            // Not x or y, restore and parse normally
            self.pos = save_pos;
        }

        // Parse normally
        return self.term();
    }

    fn parseBuiltinSUP(self: *Parser) ParseError!hvm.Term {
        try self.consume('(');
        const lab = try self.term();
        const tm0 = try self.term();
        const tm1 = try self.term();
        try self.consume(')');

        const loc = hvm.alloc_node(3);
        hvm.set(loc, lab);
        hvm.set(loc + 1, tm0);
        hvm.set(loc + 2, tm1);

        return hvm.term_new(hvm.REF, hvm.SUP_F, @truncate(loc));
    }

    fn parseBuiltinDUP(self: *Parser) ParseError!hvm.Term {
        try self.consume('(');
        const lab = try self.term();
        const val = try self.term();
        const bod = try self.term();
        try self.consume(')');

        const loc = hvm.alloc_node(3);
        hvm.set(loc, lab);
        hvm.set(loc + 1, val);
        hvm.set(loc + 2, bod);

        return hvm.term_new(hvm.REF, hvm.DUP_F, @truncate(loc));
    }

    fn parseBuiltinLOG(self: *Parser) ParseError!hvm.Term {
        try self.consume('(');
        const val = try self.term();
        const cont = try self.term();
        try self.consume(')');

        const loc = hvm.alloc_node(2);
        hvm.set(loc, val);
        hvm.set(loc + 1, cont);

        return hvm.term_new(hvm.REF, hvm.LOG_F, @truncate(loc));
    }

    // Parse a function definition: @name(params) = body
    pub fn definition(self: *Parser) ParseError!?struct { name: []const u8, params: [][]const u8, body_start: usize, body_end: usize } {
        if (self.peek() != @as(u8, '@')) return null;

        // Look ahead to check if this is a definition
        const save = self.pos;
        self.pos += 1; // skip @

        // Get the name
        const func_name = self.name() catch {
            self.pos = save;
            return null;
        };

        // Skip params if present
        var params: ArrayList([]const u8) = .empty;
        if (self.peek() == @as(u8, '(')) {
            try self.consume('(');
            while (self.peek()) |c| {
                if (c == ')') break;
                try params.append(self.allocator, self.name() catch {
                    self.pos = save;
                    params.deinit(self.allocator);
                    return null;
                });
            }
            self.consume(')') catch {
                self.pos = save;
                params.deinit(self.allocator);
                return null;
            };
        }

        self.skip();

        // Check for '='
        if (self.peek() != @as(u8, '=')) {
            self.pos = save;
            params.deinit(self.allocator);
            return null;
        }
        self.pos += 1; // skip '='

        const body_start = self.pos;

        // Find end of body (next definition or EOF)
        var depth: u32 = 0;
        while (self.pos < self.text.len) {
            const ch = self.text[self.pos];
            if (ch == '(' or ch == '{') depth += 1;
            if (ch == ')' or ch == '}') depth -= 1;
            if (depth == 0 and ch == '@' and self.pos > body_start) {
                // Check if this is a new definition (has = after name)
                const check_pos = self.pos + 1;
                var is_def = false;
                var check = check_pos;
                // Skip name
                while (check < self.text.len and isNameChar(self.text[check])) check += 1;
                // Skip whitespace
                while (check < self.text.len and (self.text[check] == ' ' or self.text[check] == '\t' or self.text[check] == '\n')) check += 1;
                // Skip params
                if (check < self.text.len and self.text[check] == '(') {
                    var d: u32 = 1;
                    check += 1;
                    while (check < self.text.len and d > 0) {
                        if (self.text[check] == '(') d += 1;
                        if (self.text[check] == ')') d -= 1;
                        check += 1;
                    }
                }
                // Skip whitespace
                while (check < self.text.len and (self.text[check] == ' ' or self.text[check] == '\t' or self.text[check] == '\n')) check += 1;
                if (check < self.text.len and self.text[check] == '=') is_def = true;

                if (is_def) break;
            }
            self.pos += 1;
        }

        const body_end = self.pos;

        return .{
            .name = func_name,
            .params = params.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory,
            .body_start = body_start,
            .body_end = body_end,
        };
    }

    // Parse entire program
    pub fn program(self: *Parser) ParseError!?hvm.Term {
        // First pass: collect all definitions
        while (true) {
            if (try self.definition()) |def| {
                const body_text = self.text[def.body_start..def.body_end];
                _ = try self.book.define(def.name, def.params, body_text);
            } else {
                break;
            }
        }

        // Compile all functions
        var it = self.book.funcs.iterator();
        while (it.next()) |entry| {
            const func = entry.value_ptr.*;
            try self.compileFunction(func);
        }

        // Parse main expression if any remains
        if (self.peek() != null) {
            return try self.term();
        }
        return null;
    }

    fn compileFunction(self: *Parser, func: Func) !void {
        // Create a sub-parser for the function body
        var body_parser = Parser.init(func.body_text, self.allocator, self.book);
        defer body_parser.deinit();

        // Bind parameters
        for (func.params, 0..) |param, i| {
            const loc = hvm.alloc_node(1);
            try body_parser.bindVar(param, loc);
            _ = i;
        }

        // Parse body
        const body = try body_parser.term();

        // Register function in HVM
        // We need to create a function that, when called, expands to the body
        // For now, we'll store the compiled body location
        const state = hvm.hvm_get_state();
        state.book[func.fid] = createFuncWrapper(func.fid, body);
    }

    fn createFuncWrapper(fid: u16, body: hvm.Term) hvm.BookFn {
        // Store the body in HVM state (not global - thread-safe)
        hvm.hvm_get_state().setFuncBody(fid, body);
        return hvm.funcDispatch;
    }
};

// =============================================================================
// Public API
// =============================================================================

pub fn parse(allocator: Allocator, text: []const u8) !hvm.Term {
    var book = Book.init(allocator);
    defer book.deinit();

    var parser = Parser.init(text, allocator, &book);
    defer parser.deinit();

    return parser.term();
}

pub fn parseProgram(allocator: Allocator, text: []const u8) !?hvm.Term {
    var book = Book.init(allocator);
    // Don't deinit book - it's needed for function lookups

    var parser = Parser.init(text, allocator, &book);
    defer parser.deinit();

    return parser.program();
}

pub fn parseAndRun(allocator: Allocator, text: []const u8) !hvm.Term {
    var book = Book.init(allocator);

    var parser = Parser.init(text, allocator, &book);
    defer parser.deinit();

    if (try parser.program()) |main_term| {
        return hvm.reduce(main_term);
    }
    return hvm.term_new(hvm.ERA, 0, 0);
}

// =============================================================================
// Pretty Printer
// =============================================================================

pub fn pretty(allocator: Allocator, term: hvm.Term) ![]const u8 {
    var buf: ArrayList(u8) = .empty;
    try prettyInto(allocator, &buf, term);
    return buf.toOwnedSlice(allocator);
}

fn prettyInto(allocator: Allocator, buf: *ArrayList(u8), term: hvm.Term) !void {
    const tag = hvm.term_tag(term);
    const ext = hvm.term_ext(term);
    const val = hvm.term_val(term);

    switch (tag) {
        hvm.ERA => try buf.appendSlice(allocator, "*"),
        hvm.NUM => try std.fmt.format(buf.writer(allocator), "#{d}", .{val}),
        hvm.LAM => {
            try buf.appendSlice(allocator, "\\_."); // Use ASCII backslash instead of λ
            try prettyInto(allocator, buf, hvm.got(val));
        },
        hvm.APP => {
            try buf.append(allocator, '(');
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ' ');
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try buf.append(allocator, ')');
        },
        hvm.SUP => {
            try std.fmt.format(buf.writer(allocator), "&{d}{{", .{ext});
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ',');
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try buf.append(allocator, '}');
        },
        hvm.CO0 => try std.fmt.format(buf.writer(allocator), "CO0({d},{d})", .{ ext, val }),
        hvm.CO1 => try std.fmt.format(buf.writer(allocator), "CO1({d},{d})", .{ ext, val }),
        hvm.VAR => try std.fmt.format(buf.writer(allocator), "VAR({d})", .{val}),
        hvm.REF => try std.fmt.format(buf.writer(allocator), "@{d}", .{ext}),
        hvm.P02, hvm.F_OP2 => {
            try buf.append(allocator, '(');
            try buf.appendSlice(allocator, hvm.op_symbol(ext));
            try buf.append(allocator, ' ');
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ' ');
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try buf.append(allocator, ')');
        },
        hvm.SWI => {
            try buf.appendSlice(allocator, "(?");
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ' ');
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try buf.append(allocator, ' ');
            try prettyInto(allocator, buf, hvm.got(val + 2));
            try buf.append(allocator, ')');
        },
        hvm.MAT => {
            try buf.appendSlice(allocator, "~...");
        },
        hvm.EQL => {
            try buf.appendSlice(allocator, "(=== ");
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ' ');
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try buf.append(allocator, ')');
        },
        hvm.RED => {
            try buf.appendSlice(allocator, "(~> ");
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ' ');
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try buf.append(allocator, ')');
        },
        hvm.ANN => {
            try buf.append(allocator, '{');
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.appendSlice(allocator, " : ");
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try buf.append(allocator, '}');
        },
        hvm.BRI => {
            try buf.appendSlice(allocator, "θ_.");
            try prettyInto(allocator, buf, hvm.got(val));
        },
        hvm.TYP => {
            try buf.appendSlice(allocator, "Type");
        },
        hvm.ALL => {
            try buf.appendSlice(allocator, "Π(");
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.appendSlice(allocator, ").");
            try prettyInto(allocator, buf, hvm.got(val + 1));
        },
        hvm.SIG => {
            try buf.appendSlice(allocator, "Σ(");
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.appendSlice(allocator, ").");
            try prettyInto(allocator, buf, hvm.got(val + 1));
        },
        hvm.SLF => {
            try buf.appendSlice(allocator, "Self.");
            try prettyInto(allocator, buf, hvm.got(val));
        },
        hvm.LIN => {
            // Linear type: Lin(term) or Lin_L(term) if labeled
            if (ext != 0) {
                try std.fmt.format(buf.writer(allocator), "Lin_{d}(", .{ext});
            } else {
                try buf.appendSlice(allocator, "Lin(");
            }
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ')');
        },
        hvm.AFF => {
            try buf.appendSlice(allocator, "Aff(");
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.append(allocator, ')');
        },
        hvm.ERR => {
            // Type error: ERR(expected, actual, code)
            try buf.appendSlice(allocator, "ERR{expected=");
            try prettyInto(allocator, buf, hvm.got(val));
            try buf.appendSlice(allocator, ", actual=");
            try prettyInto(allocator, buf, hvm.got(val + 1));
            try std.fmt.format(buf.writer(allocator), ", code={d}}}", .{ext});
        },
        else => {
            // Handle constructors C00-C15
            if (tag >= hvm.C00 and tag <= hvm.C15) {
                const arity = hvm.ctr_arity(tag);
                try std.fmt.format(buf.writer(allocator), "#C{d}{{", .{ext});
                for (0..arity) |i| {
                    if (i > 0) try buf.append(allocator, ',');
                    try prettyInto(allocator, buf, hvm.got(val + i));
                }
                try buf.append(allocator, '}');
            } else {
                try std.fmt.format(buf.writer(allocator), "?{d}({d},{d})", .{ tag, ext, val });
            }
        },
    }
}
