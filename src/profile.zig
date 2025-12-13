const std = @import("std");
const hvm = @import("hvm.zig");
const cost = @import("cost.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// HVM4 Profiler - Performance Analysis & Optimization Guidance
// =============================================================================
//
// This module provides profiling capabilities for HVM4 programs:
// 1. Per-interaction timing and counting
// 2. Heap access pattern analysis
// 3. Hotspot detection
// 4. Memory allocation tracking
// 5. Superposition branch analysis
//
// =============================================================================

/// Interaction types for profiling
pub const InteractionType = enum(u8) {
    beta_reduction,     // APP-LAM
    dup_annihilate,     // DUP-SUP same label
    dup_commute,        // DUP-SUP different label
    dup_lam,            // DUP-LAM
    dup_num,            // DUP-NUM
    dup_ctr,            // DUP-CTR
    switch_zero,        // SWI with 0
    switch_succ,        // SWI with non-0
    match_ctr,          // MAT-CTR
    op_compute,         // P02-NUM-NUM
    ref_expand,         // REF expansion
    ann_decay,          // ANN decay
    let_bind,           // LET binding
    other,              // Other interactions
};

/// Profile data for a single interaction type
pub const InteractionProfile = struct {
    count: u64 = 0,
    total_time_ns: u64 = 0,
    min_time_ns: u64 = std.math.maxInt(u64),
    max_time_ns: u64 = 0,
    heap_allocations: u64 = 0,
    heap_accesses: u64 = 0,
};

/// Location hotspot info
pub const Hotspot = struct {
    location: hvm.Val,
    access_count: u64,
    interaction_type: InteractionType,
    is_bottleneck: bool,
};

/// Profiler state
pub const Profiler = struct {
    allocator: Allocator,

    // Per-interaction-type profiles
    interaction_profiles: [16]InteractionProfile,

    // Heap access tracking
    heap_access_counts: []u32,
    heap_size: u64,

    // Hotspots (top N most accessed locations)
    hotspots: ArrayList(Hotspot),
    max_hotspots: usize,

    // Global stats
    total_interactions: u64,
    total_allocations: u64,
    total_time_ns: u64,
    peak_heap_usage: u64,

    // State
    is_profiling: bool,
    start_time: i128,

    pub fn init(allocator: Allocator, heap_size: u64) !Profiler {
        const counts = try allocator.alloc(u32, heap_size);
        @memset(counts, 0);

        return .{
            .allocator = allocator,
            .interaction_profiles = [_]InteractionProfile{.{}} ** 16,
            .heap_access_counts = counts,
            .heap_size = heap_size,
            .hotspots = ArrayList(Hotspot).empty,
            .max_hotspots = 100,
            .total_interactions = 0,
            .total_allocations = 0,
            .total_time_ns = 0,
            .peak_heap_usage = 0,
            .is_profiling = false,
            .start_time = 0,
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.allocator.free(self.heap_access_counts);
        self.hotspots.deinit(self.allocator);
    }

    /// Start profiling
    pub fn start(self: *Profiler) void {
        self.is_profiling = true;
        self.start_time = std.time.nanoTimestamp();
        self.resetStats();
    }

    /// Stop profiling
    pub fn stop(self: *Profiler) void {
        if (self.is_profiling) {
            self.total_time_ns = @intCast(std.time.nanoTimestamp() - self.start_time);
            self.is_profiling = false;
            self.computeHotspots();
        }
    }

    /// Reset all statistics
    pub fn resetStats(self: *Profiler) void {
        for (&self.interaction_profiles) |*p| {
            p.* = .{};
        }
        @memset(self.heap_access_counts, 0);
        self.hotspots.clearRetainingCapacity();
        self.total_interactions = 0;
        self.total_allocations = 0;
        self.total_time_ns = 0;
    }

    /// Record an interaction
    pub fn recordInteraction(self: *Profiler, itype: InteractionType, time_ns: u64, heap_allocs: u64, heap_accesses: u64) void {
        if (!self.is_profiling) return;

        const idx = @intFromEnum(itype);
        var prof = &self.interaction_profiles[idx];

        prof.count += 1;
        prof.total_time_ns += time_ns;
        prof.min_time_ns = @min(prof.min_time_ns, time_ns);
        prof.max_time_ns = @max(prof.max_time_ns, time_ns);
        prof.heap_allocations += heap_allocs;
        prof.heap_accesses += heap_accesses;

        self.total_interactions += 1;
    }

    /// Record heap access
    pub fn recordHeapAccess(self: *Profiler, location: hvm.Val) void {
        if (!self.is_profiling) return;
        if (location < self.heap_size) {
            self.heap_access_counts[location] +|= 1;
        }
    }

    /// Record allocation
    pub fn recordAllocation(self: *Profiler, size: u64) void {
        if (!self.is_profiling) return;
        self.total_allocations += size;

        const current_usage = hvm.get_len();
        self.peak_heap_usage = @max(self.peak_heap_usage, current_usage);
    }

    /// Compute hotspots from access counts
    fn computeHotspots(self: *Profiler) void {
        self.hotspots.clearRetainingCapacity();

        // Find top N most accessed locations
        var threshold: u32 = 0;

        // First pass: find threshold
        var max_count: u32 = 0;
        for (self.heap_access_counts) |count| {
            max_count = @max(max_count, count);
        }

        // Threshold at 10% of max
        threshold = max_count / 10;

        // Second pass: collect hotspots
        for (self.heap_access_counts, 0..) |count, i| {
            if (count > threshold and self.hotspots.items.len < self.max_hotspots) {
                self.hotspots.append(self.allocator, .{
                    .location = @intCast(i),
                    .access_count = count,
                    .interaction_type = .other,
                    .is_bottleneck = count > max_count / 2,
                }) catch continue;
            }
        }

        // Sort by access count (descending)
        std.mem.sort(Hotspot, self.hotspots.items, {}, struct {
            fn lessThan(_: void, a: Hotspot, b: Hotspot) bool {
                return a.access_count > b.access_count;
            }
        }.lessThan);
    }

    /// Get profile report
    pub fn getReport(self: *Profiler) ProfileReport {
        return .{
            .profiles = self.interaction_profiles,
            .hotspots = self.hotspots.items,
            .total_interactions = self.total_interactions,
            .total_allocations = self.total_allocations,
            .total_time_ns = self.total_time_ns,
            .peak_heap_usage = self.peak_heap_usage,
        };
    }
};

/// Profile report summary
pub const ProfileReport = struct {
    profiles: [16]InteractionProfile,
    hotspots: []const Hotspot,
    total_interactions: u64,
    total_allocations: u64,
    total_time_ns: u64,
    peak_heap_usage: u64,

    /// Get interaction breakdown
    pub fn getBreakdown(self: ProfileReport) [16]f64 {
        var percentages: [16]f64 = [_]f64{0} ** 16;

        if (self.total_interactions == 0) return percentages;

        for (self.profiles, 0..) |p, i| {
            percentages[i] = @as(f64, @floatFromInt(p.count)) / @as(f64, @floatFromInt(self.total_interactions)) * 100.0;
        }

        return percentages;
    }

    /// Get average time per interaction type
    pub fn getAverageTimes(self: ProfileReport) [16]f64 {
        var avg_times: [16]f64 = [_]f64{0} ** 16;

        for (self.profiles, 0..) |p, i| {
            if (p.count > 0) {
                avg_times[i] = @as(f64, @floatFromInt(p.total_time_ns)) / @as(f64, @floatFromInt(p.count));
            }
        }

        return avg_times;
    }

    /// Find the bottleneck interaction type
    pub fn findBottleneck(self: ProfileReport) ?InteractionType {
        var max_time: u64 = 0;
        var bottleneck: ?InteractionType = null;

        for (self.profiles, 0..) |p, i| {
            if (p.total_time_ns > max_time) {
                max_time = p.total_time_ns;
                bottleneck = @enumFromInt(i);
            }
        }

        return bottleneck;
    }

    /// Check if oracle problem detected
    pub fn hasOracleProblem(self: ProfileReport) bool {
        const dup_commute_profile = self.profiles[@intFromEnum(InteractionType.dup_commute)];
        const dup_annihilate_profile = self.profiles[@intFromEnum(InteractionType.dup_annihilate)];

        // Oracle problem likely if commutations >> annihilations
        if (dup_annihilate_profile.count == 0) return dup_commute_profile.count > 1000;

        const ratio = @as(f64, @floatFromInt(dup_commute_profile.count)) /
            @as(f64, @floatFromInt(dup_annihilate_profile.count));

        return ratio > 10.0; // More than 10x commutations vs annihilations
    }

    /// Get optimization suggestions
    pub fn getSuggestions(self: ProfileReport, allocator: Allocator) ![]const []const u8 {
        var suggestions = ArrayList([]const u8).empty;

        if (self.hasOracleProblem()) {
            try suggestions.append(allocator, "High DUP-SUP commutation ratio detected. Consider using linear types.");
        }

        const breakdown = self.getBreakdown();

        // Check for excessive beta reductions
        if (breakdown[@intFromEnum(InteractionType.beta_reduction)] > 80) {
            try suggestions.append(allocator, "High beta reduction rate. Consider partial evaluation or inlining.");
        }

        // Check for slow reference expansions
        const avg_times = self.getAverageTimes();
        if (avg_times[@intFromEnum(InteractionType.ref_expand)] > 1000) {
            try suggestions.append(allocator, "Slow REF expansions. Consider function inlining.");
        }

        // Check for high memory usage
        if (self.peak_heap_usage > self.total_allocations / 2) {
            try suggestions.append(allocator, "High peak memory usage. Consider lazy evaluation.");
        }

        return suggestions.toOwnedSlice(allocator);
    }
};

// =============================================================================
// Format & Display
// =============================================================================

/// Format profile report as string
pub fn formatReport(allocator: Allocator, report: ProfileReport) ![]u8 {
    var buf = ArrayList(u8).empty;
    const writer = buf.writer(allocator);

    try writer.print("=== HVM4 Profile Report ===\n\n", .{});

    // Summary
    try writer.print("Total Interactions: {d}\n", .{report.total_interactions});
    try writer.print("Total Time: {d:.2}ms\n", .{@as(f64, @floatFromInt(report.total_time_ns)) / 1_000_000.0});
    try writer.print("Peak Heap Usage: {d} terms\n\n", .{report.peak_heap_usage});

    // Per-interaction breakdown
    try writer.print("Interaction Breakdown:\n", .{});
    const breakdown = report.getBreakdown();
    const names = [_][]const u8{
        "beta_reduction", "dup_annihilate", "dup_commute", "dup_lam",
        "dup_num", "dup_ctr", "switch_zero", "switch_succ",
        "match_ctr", "op_compute", "ref_expand", "ann_decay",
        "let_bind", "other", "", "",
    };

    for (report.profiles, 0..) |p, i| {
        if (p.count > 0) {
            try writer.print("  {s}: {d} ({d:.1}%)\n", .{ names[i], p.count, breakdown[i] });
        }
    }

    // Hotspots
    if (report.hotspots.len > 0) {
        try writer.print("\nHotspots (top {d}):\n", .{report.hotspots.len});
        for (report.hotspots[0..@min(10, report.hotspots.len)]) |h| {
            try writer.print("  loc {d}: {d} accesses", .{ h.location, h.access_count });
            if (h.is_bottleneck) {
                try writer.print(" [BOTTLENECK]", .{});
            }
            try writer.print("\n", .{});
        }
    }

    // Oracle check
    if (report.hasOracleProblem()) {
        try writer.print("\n⚠️  WARNING: Oracle problem detected!\n", .{});
    }

    return buf.toOwnedSlice(allocator);
}

// =============================================================================
// Integration with Cost Model
// =============================================================================

/// Profile with cost analysis
pub fn profileWithCost(allocator: Allocator, term: hvm.Term) !struct { report: ProfileReport, cost: cost.CostAnalysis } {
    var profiler = try Profiler.init(allocator, 1024 * 1024);
    defer profiler.deinit();

    profiler.start();
    _ = hvm.reduce(term);
    profiler.stop();

    const cost_analysis = cost.estimateCost(term);
    const report = profiler.getReport();

    return .{ .report = report, .cost = cost_analysis };
}

// =============================================================================
// Public API
// =============================================================================

/// Quick profile of a term reduction
pub fn profile(allocator: Allocator, term: hvm.Term) !ProfileReport {
    var profiler = try Profiler.init(allocator, 1024 * 1024);
    defer profiler.deinit();

    profiler.start();
    _ = hvm.reduce(term);
    profiler.stop();

    return profiler.getReport();
}

/// Profile with interaction counting
pub fn profileInteractions(term: hvm.Term) struct { result: hvm.Term, interactions: u64 } {
    const start_itrs = hvm.get_itrs();
    const result = hvm.reduce(term);
    const end_itrs = hvm.get_itrs();

    return .{
        .result = result,
        .interactions = end_itrs - start_itrs,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "profiler init and deinit" {
    const allocator = std.testing.allocator;

    var profiler = try Profiler.init(allocator, 1024);
    defer profiler.deinit();

    try std.testing.expect(!profiler.is_profiling);
}

test "profiler start stop" {
    const allocator = std.testing.allocator;

    var profiler = try Profiler.init(allocator, 1024);
    defer profiler.deinit();

    profiler.start();
    try std.testing.expect(profiler.is_profiling);

    profiler.stop();
    try std.testing.expect(!profiler.is_profiling);
}

test "record interaction" {
    const allocator = std.testing.allocator;

    var profiler = try Profiler.init(allocator, 1024);
    defer profiler.deinit();

    profiler.start();
    profiler.recordInteraction(.beta_reduction, 100, 2, 4);
    profiler.recordInteraction(.beta_reduction, 200, 3, 6);
    profiler.stop();

    const report = profiler.getReport();
    const beta_profile = report.profiles[@intFromEnum(InteractionType.beta_reduction)];

    try std.testing.expectEqual(@as(u64, 2), beta_profile.count);
    try std.testing.expectEqual(@as(u64, 300), beta_profile.total_time_ns);
}

test "profile report breakdown" {
    const allocator = std.testing.allocator;

    var profiler = try Profiler.init(allocator, 1024);
    defer profiler.deinit();

    profiler.start();
    profiler.recordInteraction(.beta_reduction, 100, 0, 0);
    profiler.recordInteraction(.op_compute, 100, 0, 0);
    profiler.stop();

    const report = profiler.getReport();
    const breakdown = report.getBreakdown();

    try std.testing.expectEqual(@as(f64, 50.0), breakdown[@intFromEnum(InteractionType.beta_reduction)]);
}
