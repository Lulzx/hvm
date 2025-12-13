const std = @import("std");
const hvm = @import("hvm.zig");
const parser = @import("parser.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// HVM4 Debugger - Step-by-step reduction and inspection
// =============================================================================

/// Types of debug events that can occur during reduction
pub const EventKind = enum {
    /// Beta reduction: APP + LAM
    beta_reduction,
    /// Superposition distribution: APP + SUP
    app_sup_distribution,
    /// Duplication annihilation: CO0/CO1 + SUP (same label)
    dup_annihilation,
    /// Duplication commutation: CO0/CO1 + SUP (different label) - oracle risk!
    dup_commutation,
    /// Lambda duplication: CO0/CO1 + LAM
    dup_lam,
    /// Constructor duplication: CO0/CO1 + CTR
    dup_ctr,
    /// Pattern match: MAT + CTR
    pattern_match,
    /// Numeric switch: SWI + NUM
    numeric_switch,
    /// Binary operation: OP2 + NUM
    binary_op,
    /// Erasure: ERA interaction
    erasure,
    /// Reference resolution: REF
    ref_resolution,
    /// Stack push
    stack_push,
    /// Stack pop
    stack_pop,
    /// Heap allocation
    heap_alloc,
    /// Linearity violation detected
    linearity_violation,
    /// Type error
    type_error,
};

/// A debug event capturing a single step of reduction
pub const DebugEvent = struct {
    kind: EventKind,
    /// The term being reduced
    term: hvm.Term,
    /// Second term involved (for interactions)
    term2: ?hvm.Term,
    /// Heap location affected
    location: hvm.Val,
    /// Current stack depth
    stack_depth: u32,
    /// Interaction count at this point
    interaction_num: u64,
    /// Labels involved (for SUP/DUP)
    label1: ?hvm.Ext,
    label2: ?hvm.Ext,
    /// Additional message
    message: []const u8,
};

/// Debugger mode
pub const DebugMode = enum {
    /// Run without stopping
    run,
    /// Step one interaction at a time
    step,
    /// Run until breakpoint
    run_to_breakpoint,
    /// Step over (skip into function calls)
    step_over,
};

/// Breakpoint condition
pub const Breakpoint = struct {
    /// Break on specific interaction kind
    on_event: ?EventKind,
    /// Break at specific heap location
    at_location: ?hvm.Val,
    /// Break after N interactions
    after_interactions: ?u64,
    /// Break on specific tag pair
    on_tags: ?struct { tag1: hvm.Tag, tag2: hvm.Tag },
    /// Break on oracle risk (dup commutation)
    on_oracle_risk: bool,
    /// Break on linearity violation
    on_linearity_violation: bool,
};

/// The debugger state
pub const Debugger = struct {
    allocator: Allocator,
    /// Recorded events (ring buffer)
    events: ArrayList(DebugEvent),
    /// Maximum events to keep
    max_events: usize,
    /// Current mode
    mode: DebugMode,
    /// Breakpoints
    breakpoints: ArrayList(Breakpoint),
    /// Is debugger active?
    active: bool,
    /// Should we pause on next interaction?
    pause_requested: bool,
    /// Last event that triggered a pause
    pause_event: ?DebugEvent,
    /// Statistics
    stats: Stats,

    pub const Stats = struct {
        total_interactions: u64,
        beta_reductions: u64,
        dup_annihilations: u64,
        dup_commutations: u64,
        pattern_matches: u64,
        heap_allocs: u64,
        max_stack_depth: u32,
    };

    pub fn init(allocator: Allocator) Debugger {
        return .{
            .allocator = allocator,
            .events = ArrayList(DebugEvent).empty,
            .max_events = 10000,
            .mode = .run,
            .breakpoints = ArrayList(Breakpoint).empty,
            .active = false,
            .pause_requested = false,
            .pause_event = null,
            .stats = std.mem.zeroes(Stats),
        };
    }

    pub fn deinit(self: *Debugger) void {
        self.events.deinit(self.allocator);
        self.breakpoints.deinit(self.allocator);
    }

    /// Start debugging session
    pub fn start(self: *Debugger) void {
        self.active = true;
        self.pause_requested = false;
        self.stats = std.mem.zeroes(Stats);
    }

    /// Stop debugging session
    pub fn stop(self: *Debugger) void {
        self.active = false;
    }

    /// Set debug mode
    pub fn setMode(self: *Debugger, mode: DebugMode) void {
        self.mode = mode;
    }

    /// Add a breakpoint
    pub fn addBreakpoint(self: *Debugger, bp: Breakpoint) !void {
        try self.breakpoints.append(self.allocator, bp);
    }

    /// Clear all breakpoints
    pub fn clearBreakpoints(self: *Debugger) void {
        self.breakpoints.clearRetainingCapacity();
    }

    /// Request pause on next interaction
    pub fn requestPause(self: *Debugger) void {
        self.pause_requested = true;
    }

    /// Record an event
    pub fn recordEvent(self: *Debugger, event: DebugEvent) !void {
        if (!self.active) return;

        // Update stats
        self.stats.total_interactions += 1;
        switch (event.kind) {
            .beta_reduction => self.stats.beta_reductions += 1,
            .dup_annihilation => self.stats.dup_annihilations += 1,
            .dup_commutation => self.stats.dup_commutations += 1,
            .pattern_match => self.stats.pattern_matches += 1,
            .heap_alloc => self.stats.heap_allocs += 1,
            else => {},
        }
        if (event.stack_depth > self.stats.max_stack_depth) {
            self.stats.max_stack_depth = event.stack_depth;
        }

        // Add to event list (ring buffer behavior)
        if (self.events.items.len >= self.max_events) {
            // Remove oldest event
            _ = self.events.orderedRemove(0);
        }
        try self.events.append(self.allocator, event);

        // Check breakpoints
        if (self.shouldBreak(event)) {
            self.pause_requested = true;
            self.pause_event = event;
        }
    }

    /// Check if we should break on this event
    fn shouldBreak(self: *Debugger, event: DebugEvent) bool {
        if (self.mode == .step) return true;

        for (self.breakpoints.items) |bp| {
            if (bp.on_event) |ev| {
                if (ev == event.kind) return true;
            }
            if (bp.at_location) |loc| {
                if (loc == event.location) return true;
            }
            if (bp.after_interactions) |n| {
                if (self.stats.total_interactions >= n) return true;
            }
            if (bp.on_oracle_risk and event.kind == .dup_commutation) {
                return true;
            }
            if (bp.on_linearity_violation and event.kind == .linearity_violation) {
                return true;
            }
        }

        return false;
    }

    /// Get the last N events
    pub fn getRecentEvents(self: *Debugger, n: usize) []const DebugEvent {
        const start = if (self.events.items.len > n) self.events.items.len - n else 0;
        return self.events.items[start..];
    }

    /// Format an event for display
    pub fn formatEvent(self: *Debugger, event: DebugEvent) ![]const u8 {
        var buf = ArrayList(u8).empty;

        const kind_str = switch (event.kind) {
            .beta_reduction => "BETA",
            .app_sup_distribution => "APP+SUP",
            .dup_annihilation => "ANNIHILATE",
            .dup_commutation => "COMMUTE(!)",
            .dup_lam => "DUP+LAM",
            .dup_ctr => "DUP+CTR",
            .pattern_match => "MATCH",
            .numeric_switch => "SWITCH",
            .binary_op => "OP",
            .erasure => "ERASE",
            .ref_resolution => "REF",
            .stack_push => "PUSH",
            .stack_pop => "POP",
            .heap_alloc => "ALLOC",
            .linearity_violation => "LINEAR(!)",
            .type_error => "TYPE_ERR",
        };

        try std.fmt.format(buf.writer(self.allocator), "[{d}] {s} @ loc={d} depth={d}", .{
            event.interaction_num,
            kind_str,
            event.location,
            event.stack_depth,
        });

        if (event.label1) |l1| {
            try std.fmt.format(buf.writer(self.allocator), " label1={d}", .{l1});
        }
        if (event.label2) |l2| {
            try std.fmt.format(buf.writer(self.allocator), " label2={d}", .{l2});
        }

        if (event.message.len > 0) {
            try std.fmt.format(buf.writer(self.allocator), " -- {s}", .{event.message});
        }

        return buf.toOwnedSlice(self.allocator);
    }

    /// Inspect a term at a heap location
    pub fn inspect(self: *Debugger, loc: hvm.Val) TermInfo {
        _ = self;
        const term = hvm.got(loc);
        return TermInfo.fromTerm(term, loc);
    }

    /// Get statistics
    pub fn getStats(self: *Debugger) Stats {
        return self.stats;
    }

    /// Print statistics summary
    pub fn printStats(self: *Debugger) void {
        const s = self.stats;
        std.debug.print(
            \\=== Debug Statistics ===
            \\Total interactions: {d}
            \\  Beta reductions:   {d}
            \\  Annihilations:     {d}
            \\  Commutations:      {d} (oracle risk!)
            \\  Pattern matches:   {d}
            \\  Heap allocations:  {d}
            \\Max stack depth:     {d}
            \\========================
            \\
        , .{
            s.total_interactions,
            s.beta_reductions,
            s.dup_annihilations,
            s.dup_commutations,
            s.pattern_matches,
            s.heap_allocs,
            s.max_stack_depth,
        });
    }
};

/// Information about a term for inspection
pub const TermInfo = struct {
    tag: hvm.Tag,
    tag_name: []const u8,
    ext: hvm.Ext,
    val: hvm.Val,
    loc: hvm.Val,
    children: []hvm.Val,
    is_value: bool,
    is_eliminator: bool,

    pub fn fromTerm(term: hvm.Term, loc: hvm.Val) TermInfo {
        const tag = hvm.term_tag(term);
        const ext = hvm.term_ext(term);
        const val = hvm.term_val(term);

        const tag_name = getTagName(tag);
        const is_value = isValue(tag);
        const is_elim = isEliminator(tag);

        // Determine children based on tag
        var children_buf: [16]hvm.Val = undefined;
        var num_children: usize = 0;

        switch (tag) {
            hvm.LAM, hvm.LIN, hvm.AFF, hvm.SLF, hvm.BRI => {
                children_buf[0] = val;
                num_children = 1;
            },
            hvm.APP, hvm.SUP, hvm.ALL, hvm.SIG, hvm.ANN, hvm.P02, hvm.EQL, hvm.RED, hvm.ERR => {
                children_buf[0] = val;
                children_buf[1] = val + 1;
                num_children = 2;
            },
            hvm.SWI => {
                children_buf[0] = val;
                children_buf[1] = val + 1;
                children_buf[2] = val + 2;
                num_children = 3;
            },
            hvm.CO0, hvm.CO1 => {
                children_buf[0] = val;
                num_children = 1;
            },
            else => {
                if (tag >= hvm.C00 and tag <= hvm.C15) {
                    const arity = hvm.ctr_arity(tag);
                    for (0..arity) |i| {
                        children_buf[i] = val + @as(hvm.Val, @truncate(i));
                    }
                    num_children = arity;
                }
            },
        }

        return .{
            .tag = tag,
            .tag_name = tag_name,
            .ext = ext,
            .val = val,
            .loc = loc,
            .children = children_buf[0..num_children],
            .is_value = is_value,
            .is_eliminator = is_elim,
        };
    }
};

fn getTagName(tag: hvm.Tag) []const u8 {
    return switch (tag) {
        hvm.APP => "APP",
        hvm.VAR => "VAR",
        hvm.LAM => "LAM",
        hvm.CO0 => "CO0",
        hvm.CO1 => "CO1",
        hvm.SUP => "SUP",
        hvm.DUP => "DUP",
        hvm.REF => "REF",
        hvm.ERA => "ERA",
        hvm.RED => "RED",
        hvm.LET => "LET",
        hvm.USE => "USE",
        hvm.EQL => "EQL",
        hvm.MAT => "MAT",
        hvm.SWI => "SWI",
        hvm.NUM => "NUM",
        hvm.ANN => "ANN",
        hvm.BRI => "BRI",
        hvm.TYP => "TYP",
        hvm.ALL => "ALL",
        hvm.SIG => "SIG",
        hvm.SLF => "SLF",
        hvm.LIN => "LIN",
        hvm.AFF => "AFF",
        hvm.ERR => "ERR",
        hvm.P02 => "P02",
        else => "???",
    };
}

fn isValue(tag: hvm.Tag) bool {
    return switch (tag) {
        hvm.LAM, hvm.SUP, hvm.ERA, hvm.NUM, hvm.TYP => true,
        else => tag >= hvm.C00 and tag <= hvm.C15,
    };
}

fn isEliminator(tag: hvm.Tag) bool {
    return switch (tag) {
        hvm.APP, hvm.MAT, hvm.SWI, hvm.CO0, hvm.CO1, hvm.P02 => true,
        else => false,
    };
}

// =============================================================================
// Global debugger instance (for integration with reducer)
// =============================================================================

var global_debugger: ?*Debugger = null;

/// Set the global debugger instance
pub fn setGlobalDebugger(dbg: *Debugger) void {
    global_debugger = dbg;
}

/// Get the global debugger instance
pub fn getGlobalDebugger() ?*Debugger {
    return global_debugger;
}

/// Record an event to the global debugger (if active)
pub fn recordGlobalEvent(event: DebugEvent) void {
    if (global_debugger) |dbg| {
        dbg.recordEvent(event) catch {};
    }
}

/// Check if debugger wants to pause
pub fn shouldPause() bool {
    if (global_debugger) |dbg| {
        return dbg.pause_requested;
    }
    return false;
}

// =============================================================================
// Helper functions for creating events
// =============================================================================

pub fn makeBetaEvent(app_loc: hvm.Val, lam_loc: hvm.Val, depth: u32, itrs: u64) DebugEvent {
    return .{
        .kind = .beta_reduction,
        .term = hvm.got(app_loc),
        .term2 = hvm.got(lam_loc),
        .location = app_loc,
        .stack_depth = depth,
        .interaction_num = itrs,
        .label1 = null,
        .label2 = null,
        .message = "Beta reduction: (\\x.body) arg -> body[x:=arg]",
    };
}

pub fn makeDupSupEvent(dup_loc: hvm.Val, sup_loc: hvm.Val, dup_lab: hvm.Ext, sup_lab: hvm.Ext, depth: u32, itrs: u64) DebugEvent {
    const is_annihilation = dup_lab == sup_lab;
    return .{
        .kind = if (is_annihilation) .dup_annihilation else .dup_commutation,
        .term = hvm.term_new(hvm.CO0, dup_lab, dup_loc),
        .term2 = hvm.term_new(hvm.SUP, sup_lab, sup_loc),
        .location = dup_loc,
        .stack_depth = depth,
        .interaction_num = itrs,
        .label1 = dup_lab,
        .label2 = sup_lab,
        .message = if (is_annihilation)
            "Annihilation: labels match, O(1) collapse"
        else
            "COMMUTATION: different labels, potential exponential blowup!",
    };
}

// =============================================================================
// Visualization export
// =============================================================================

/// Export term as DOT graph for Graphviz
pub fn exportDOT(allocator: Allocator, term: hvm.Term) ![]const u8 {
    var buf = ArrayList(u8).empty;
    const w = buf.writer(allocator);

    try w.writeAll("digraph HVM {\n");
    try w.writeAll("  node [shape=record];\n");

    try exportDOTNode(allocator, &buf, term, 0);

    try w.writeAll("}\n");

    return buf.toOwnedSlice(allocator);
}

fn exportDOTNode(allocator: Allocator, buf: *ArrayList(u8), term: hvm.Term, id: u32) !u32 {
    const tag = hvm.term_tag(term);
    const ext = hvm.term_ext(term);
    const val = hvm.term_val(term);
    const w = buf.writer(allocator);

    const tag_name = getTagName(tag);

    try std.fmt.format(w, "  n{d} [label=\"{{{s}|ext={d}|val={d}}}\"];\n", .{ id, tag_name, ext, val });

    var next_id = id + 1;

    // Add edges to children
    switch (tag) {
        hvm.LAM, hvm.LIN, hvm.AFF => {
            const child = hvm.got(val);
            try std.fmt.format(w, "  n{d} -> n{d} [label=\"body\"];\n", .{ id, next_id });
            next_id = try exportDOTNode(allocator, buf, child, next_id);
        },
        hvm.APP => {
            const func = hvm.got(val);
            const arg = hvm.got(val + 1);
            try std.fmt.format(w, "  n{d} -> n{d} [label=\"func\"];\n", .{ id, next_id });
            next_id = try exportDOTNode(allocator, buf, func, next_id);
            try std.fmt.format(w, "  n{d} -> n{d} [label=\"arg\"];\n", .{ id, next_id });
            next_id = try exportDOTNode(allocator, buf, arg, next_id);
        },
        hvm.SUP => {
            const lft = hvm.got(val);
            const rgt = hvm.got(val + 1);
            try std.fmt.format(w, "  n{d} -> n{d} [label=\"left\"];\n", .{ id, next_id });
            next_id = try exportDOTNode(allocator, buf, lft, next_id);
            try std.fmt.format(w, "  n{d} -> n{d} [label=\"right\"];\n", .{ id, next_id });
            next_id = try exportDOTNode(allocator, buf, rgt, next_id);
        },
        else => {},
    }

    return next_id;
}

/// Export term as JSON for web visualizer
pub fn exportJSON(allocator: Allocator, term: hvm.Term) ![]const u8 {
    var buf = ArrayList(u8).empty;
    try exportJSONNode(allocator, &buf, term);
    return buf.toOwnedSlice(allocator);
}

fn exportJSONNode(allocator: Allocator, buf: *ArrayList(u8), term: hvm.Term) !void {
    const tag = hvm.term_tag(term);
    const ext = hvm.term_ext(term);
    const val = hvm.term_val(term);
    const w = buf.writer(allocator);

    try w.writeAll("{");
    try std.fmt.format(w, "\"tag\":\"{s}\",\"ext\":{d},\"val\":{d}", .{ getTagName(tag), ext, val });

    switch (tag) {
        hvm.LAM, hvm.LIN, hvm.AFF => {
            try w.writeAll(",\"body\":");
            try exportJSONNode(allocator, buf, hvm.got(val));
        },
        hvm.APP => {
            try w.writeAll(",\"func\":");
            try exportJSONNode(allocator, buf, hvm.got(val));
            try w.writeAll(",\"arg\":");
            try exportJSONNode(allocator, buf, hvm.got(val + 1));
        },
        hvm.SUP => {
            try w.writeAll(",\"left\":");
            try exportJSONNode(allocator, buf, hvm.got(val));
            try w.writeAll(",\"right\":");
            try exportJSONNode(allocator, buf, hvm.got(val + 1));
        },
        else => {},
    }

    try w.writeAll("}");
}
