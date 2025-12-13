const std = @import("std");
const hvm = @import("hvm.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// =============================================================================
// HVM4 Cost Semantics - Interaction-based cost model
// =============================================================================
//
// This module provides a cost model for HVM4 programs based on the interaction
// calculus. Each interaction rule has an associated cost, and the total cost
// of a reduction is the sum of all interaction costs.
//
// The cost model helps:
// 1. Predict program performance
// 2. Detect exponential blowup patterns (high commutation cost)
// 3. Compare optimization strategies
// 4. Guide supercompilation decisions
//
// =============================================================================

/// Cost weights for different interaction types
pub const CostModel = struct {
    /// Beta reduction: (λx.body) arg → body[x:=arg]
    /// This is the fundamental reduction, constant time
    beta: u32 = 1,

    /// Application + Superposition distribution
    /// (f &{a,b}) → &{(f a'), (f b')} with lazy duplication
    app_sup: u32 = 2,

    /// DUP + SUP annihilation (same label)
    /// CO0(&L{a,b}) where labels match → a, CO1 → b
    /// This is O(1), very cheap
    dup_annihilate: u32 = 1,

    /// DUP + SUP commutation (different labels)
    /// CO0(&L{a,b}) where labels differ → creates 2x2 grid
    /// THIS IS THE EXPENSIVE ONE - potential exponential
    dup_commute: u32 = 8,

    /// DUP + LAM
    /// CO0(λx.body) → creates two lambdas sharing body
    dup_lam: u32 = 3,

    /// DUP + Constructor
    /// CO0(#Pair{a,b}) → distributes over fields
    dup_ctr: u32 = 2,

    /// Pattern match
    /// ~{case0, case1, ...} (Ci{fields}) → casei applied to fields
    pattern_match: u32 = 2,

    /// Numeric switch
    /// (?x z s) 0 → z, (?x z s) n → s (n-1)
    numeric_switch: u32 = 1,

    /// Binary operation
    /// (+ a b) where a, b are numbers
    binary_op: u32 = 1,

    /// Erasure
    /// Interaction with ERA
    erasure: u32 = 1,

    /// Heap allocation (per word)
    alloc: u32 = 1,

    /// Reference resolution
    ref_call: u32 = 1,

    /// Linear type unwrap
    linear_unwrap: u32 = 0, // Free - just removes marker

    /// Type annotation decay
    type_decay: u32 = 0, // Free - just removes annotation

    /// Create default cost model
    pub fn default() CostModel {
        return .{};
    }

    /// Create "cheap" cost model (ignores allocation)
    pub fn cheap() CostModel {
        return .{
            .alloc = 0,
            .erasure = 0,
            .linear_unwrap = 0,
            .type_decay = 0,
        };
    }

    /// Create "oracle-aware" cost model (heavy penalty for commutation)
    pub fn oracleAware() CostModel {
        return .{
            .dup_commute = 100, // Very expensive - this is the oracle problem
        };
    }
};

/// Result of cost analysis
pub const CostAnalysis = struct {
    /// Total weighted cost
    total_cost: u64,
    /// Breakdown by interaction type
    breakdown: Breakdown,
    /// Estimated time complexity class
    complexity: Complexity,
    /// Warning if oracle risk detected
    oracle_warning: bool,
    /// Commutation ratio (commutations / annihilations)
    commutation_ratio: f32,

    pub const Breakdown = struct {
        beta: u64 = 0,
        app_sup: u64 = 0,
        dup_annihilate: u64 = 0,
        dup_commute: u64 = 0,
        dup_lam: u64 = 0,
        dup_ctr: u64 = 0,
        pattern_match: u64 = 0,
        numeric_switch: u64 = 0,
        binary_op: u64 = 0,
        erasure: u64 = 0,
        alloc: u64 = 0,
        ref_call: u64 = 0,
    };

    pub const Complexity = enum {
        /// O(1) - constant
        constant,
        /// O(log n) - logarithmic
        logarithmic,
        /// O(n) - linear
        linear,
        /// O(n log n) - linearithmic
        linearithmic,
        /// O(n²) - quadratic
        quadratic,
        /// O(n³) - cubic
        cubic,
        /// O(2^n) - exponential (oracle problem!)
        exponential,
        /// Unknown
        unknown,
    };
};

/// Cost tracker - accumulates costs during reduction
pub const CostTracker = struct {
    model: CostModel,
    breakdown: CostAnalysis.Breakdown,
    interaction_count: u64,

    pub fn init(model: CostModel) CostTracker {
        return .{
            .model = model,
            .breakdown = .{},
            .interaction_count = 0,
        };
    }

    pub fn reset(self: *CostTracker) void {
        self.breakdown = .{};
        self.interaction_count = 0;
    }

    /// Record a beta reduction
    pub fn recordBeta(self: *CostTracker) void {
        self.breakdown.beta += self.model.beta;
        self.interaction_count += 1;
    }

    /// Record APP+SUP distribution
    pub fn recordAppSup(self: *CostTracker) void {
        self.breakdown.app_sup += self.model.app_sup;
        self.interaction_count += 1;
    }

    /// Record DUP+SUP annihilation
    pub fn recordDupAnnihilate(self: *CostTracker) void {
        self.breakdown.dup_annihilate += self.model.dup_annihilate;
        self.interaction_count += 1;
    }

    /// Record DUP+SUP commutation (oracle risk!)
    pub fn recordDupCommute(self: *CostTracker) void {
        self.breakdown.dup_commute += self.model.dup_commute;
        self.interaction_count += 1;
    }

    /// Record DUP+LAM
    pub fn recordDupLam(self: *CostTracker) void {
        self.breakdown.dup_lam += self.model.dup_lam;
        self.interaction_count += 1;
    }

    /// Record DUP+CTR
    pub fn recordDupCtr(self: *CostTracker) void {
        self.breakdown.dup_ctr += self.model.dup_ctr;
        self.interaction_count += 1;
    }

    /// Record pattern match
    pub fn recordMatch(self: *CostTracker) void {
        self.breakdown.pattern_match += self.model.pattern_match;
        self.interaction_count += 1;
    }

    /// Record numeric switch
    pub fn recordSwitch(self: *CostTracker) void {
        self.breakdown.numeric_switch += self.model.numeric_switch;
        self.interaction_count += 1;
    }

    /// Record binary operation
    pub fn recordBinaryOp(self: *CostTracker) void {
        self.breakdown.binary_op += self.model.binary_op;
        self.interaction_count += 1;
    }

    /// Record erasure
    pub fn recordErasure(self: *CostTracker) void {
        self.breakdown.erasure += self.model.erasure;
        self.interaction_count += 1;
    }

    /// Record allocation
    pub fn recordAlloc(self: *CostTracker, words: u32) void {
        self.breakdown.alloc += @as(u64, self.model.alloc) * words;
    }

    /// Record reference call
    pub fn recordRefCall(self: *CostTracker) void {
        self.breakdown.ref_call += self.model.ref_call;
        self.interaction_count += 1;
    }

    /// Get total cost
    pub fn totalCost(self: *const CostTracker) u64 {
        const b = self.breakdown;
        return b.beta + b.app_sup + b.dup_annihilate + b.dup_commute +
            b.dup_lam + b.dup_ctr + b.pattern_match + b.numeric_switch +
            b.binary_op + b.erasure + b.alloc + b.ref_call;
    }

    /// Get full analysis
    pub fn analyze(self: *const CostTracker) CostAnalysis {
        const total = self.totalCost();
        const b = self.breakdown;

        // Calculate commutation ratio
        const annihilations = b.dup_annihilate / @max(self.model.dup_annihilate, 1);
        const commutations = b.dup_commute / @max(self.model.dup_commute, 1);
        const ratio = if (annihilations > 0)
            @as(f32, @floatFromInt(commutations)) / @as(f32, @floatFromInt(annihilations))
        else if (commutations > 0)
            999.0 // All commutations, no annihilations - very bad
        else
            0.0;

        // Estimate complexity
        const complexity = estimateComplexity(self.interaction_count, commutations, ratio);

        return .{
            .total_cost = total,
            .breakdown = b,
            .complexity = complexity,
            .oracle_warning = commutations > 10 or ratio > 0.5,
            .commutation_ratio = ratio,
        };
    }
};

fn estimateComplexity(interactions: u64, commutations: u64, ratio: f32) CostAnalysis.Complexity {
    // If we have many commutations relative to total, likely exponential
    if (ratio > 1.0 or commutations > 100) {
        return .exponential;
    }

    // If no commutations and moderate interactions, likely polynomial
    if (commutations == 0) {
        if (interactions < 100) return .constant;
        if (interactions < 10000) return .linear;
        return .quadratic;
    }

    // Some commutations - could be quadratic or worse
    if (ratio > 0.1) return .quadratic;
    return .linearithmic;
}

// =============================================================================
// Static Cost Estimation (before reduction)
// =============================================================================

/// Estimate cost of reducing a term (without actually reducing)
pub fn estimateCost(term: hvm.Term, model: CostModel) CostEstimate {
    var estimate = CostEstimate{
        .model = model,
        .min_cost = 0,
        .max_cost = 0,
        .likely_cost = 0,
        .sup_count = 0,
        .dup_count = 0,
        .app_count = 0,
        .oracle_risk = false,
    };

    analyzeTermCost(term, &estimate, 0, false);

    // Calculate likely cost
    estimate.likely_cost = estimate.min_cost;
    if (estimate.oracle_risk) {
        // If oracle risk, max could be exponential
        estimate.max_cost = estimate.min_cost * (@as(u64, 1) << @min(estimate.sup_count, 20));
    } else {
        estimate.max_cost = estimate.min_cost * 2; // Some margin
    }

    return estimate;
}

pub const CostEstimate = struct {
    model: CostModel,
    min_cost: u64,
    max_cost: u64,
    likely_cost: u64,
    sup_count: u32,
    dup_count: u32,
    app_count: u32,
    oracle_risk: bool,
};

fn analyzeTermCost(term: hvm.Term, estimate: *CostEstimate, depth: u32, in_dup: bool) void {
    if (depth > 1000) return; // Prevent stack overflow

    const tag = hvm.term_tag(term);
    const val = hvm.term_val(term);

    switch (tag) {
        hvm.APP => {
            estimate.app_count += 1;
            estimate.min_cost += estimate.model.beta; // Assume will beta-reduce
            analyzeTermCost(hvm.got(val), estimate, depth + 1, in_dup);
            analyzeTermCost(hvm.got(val + 1), estimate, depth + 1, in_dup);
        },
        hvm.LAM => {
            if (in_dup) {
                estimate.min_cost += estimate.model.dup_lam;
            }
            analyzeTermCost(hvm.got(val), estimate, depth + 1, in_dup);
        },
        hvm.SUP => {
            estimate.sup_count += 1;
            if (in_dup) {
                // SUP inside DUP - check for oracle risk
                estimate.oracle_risk = true;
                estimate.min_cost += estimate.model.dup_commute;
            }
            analyzeTermCost(hvm.got(val), estimate, depth + 1, in_dup);
            analyzeTermCost(hvm.got(val + 1), estimate, depth + 1, in_dup);
        },
        hvm.CO0, hvm.CO1 => {
            estimate.dup_count += 1;
            analyzeTermCost(hvm.got(val), estimate, depth + 1, true);
        },
        hvm.LIN, hvm.AFF => {
            // Linear types prevent oracle problem
            analyzeTermCost(hvm.got(val), estimate, depth + 1, false); // Reset in_dup
        },
        hvm.MAT => {
            estimate.min_cost += estimate.model.pattern_match;
            analyzeTermCost(hvm.got(val), estimate, depth + 1, in_dup);
        },
        hvm.SWI => {
            estimate.min_cost += estimate.model.numeric_switch;
            analyzeTermCost(hvm.got(val), estimate, depth + 1, in_dup);
            analyzeTermCost(hvm.got(val + 1), estimate, depth + 1, in_dup);
            analyzeTermCost(hvm.got(val + 2), estimate, depth + 1, in_dup);
        },
        hvm.P02 => {
            estimate.min_cost += estimate.model.binary_op;
            analyzeTermCost(hvm.got(val), estimate, depth + 1, in_dup);
            analyzeTermCost(hvm.got(val + 1), estimate, depth + 1, in_dup);
        },
        hvm.REF => {
            estimate.min_cost += estimate.model.ref_call;
        },
        hvm.ERA => {
            estimate.min_cost += estimate.model.erasure;
        },
        else => {
            // Handle constructors
            if (tag >= hvm.C00 and tag <= hvm.C15) {
                if (in_dup) {
                    estimate.min_cost += estimate.model.dup_ctr;
                }
                const arity = hvm.ctr_arity(tag);
                for (0..arity) |i| {
                    analyzeTermCost(hvm.got(val + @as(hvm.Val, @truncate(i))), estimate, depth + 1, in_dup);
                }
            }
        },
    }
}

// =============================================================================
// Global cost tracker (for integration with reducer)
// =============================================================================

var global_tracker: ?*CostTracker = null;

pub fn setGlobalTracker(tracker: *CostTracker) void {
    global_tracker = tracker;
}

pub fn getGlobalTracker() ?*CostTracker {
    return global_tracker;
}

// =============================================================================
// Utility functions
// =============================================================================

/// Format cost analysis as human-readable string
pub fn formatAnalysis(allocator: Allocator, analysis: CostAnalysis) ![]const u8 {
    var buf = ArrayList(u8).empty;
    const w = buf.writer(allocator);

    try w.writeAll("=== Cost Analysis ===\n");
    try std.fmt.format(w, "Total cost: {d}\n", .{analysis.total_cost});
    try w.writeAll("\nBreakdown:\n");

    const b = analysis.breakdown;
    if (b.beta > 0) try std.fmt.format(w, "  Beta reductions:    {d}\n", .{b.beta});
    if (b.app_sup > 0) try std.fmt.format(w, "  APP+SUP dist:       {d}\n", .{b.app_sup});
    if (b.dup_annihilate > 0) try std.fmt.format(w, "  DUP annihilations:  {d}\n", .{b.dup_annihilate});
    if (b.dup_commute > 0) try std.fmt.format(w, "  DUP commutations:   {d} (!)\n", .{b.dup_commute});
    if (b.dup_lam > 0) try std.fmt.format(w, "  DUP+LAM:            {d}\n", .{b.dup_lam});
    if (b.dup_ctr > 0) try std.fmt.format(w, "  DUP+CTR:            {d}\n", .{b.dup_ctr});
    if (b.pattern_match > 0) try std.fmt.format(w, "  Pattern matches:    {d}\n", .{b.pattern_match});
    if (b.numeric_switch > 0) try std.fmt.format(w, "  Numeric switches:   {d}\n", .{b.numeric_switch});
    if (b.binary_op > 0) try std.fmt.format(w, "  Binary ops:         {d}\n", .{b.binary_op});
    if (b.erasure > 0) try std.fmt.format(w, "  Erasures:           {d}\n", .{b.erasure});
    if (b.alloc > 0) try std.fmt.format(w, "  Allocations:        {d}\n", .{b.alloc});
    if (b.ref_call > 0) try std.fmt.format(w, "  Reference calls:    {d}\n", .{b.ref_call});

    try std.fmt.format(w, "\nCommutation ratio: {d:.2}\n", .{analysis.commutation_ratio});

    const complexity_str = switch (analysis.complexity) {
        .constant => "O(1)",
        .logarithmic => "O(log n)",
        .linear => "O(n)",
        .linearithmic => "O(n log n)",
        .quadratic => "O(n²)",
        .cubic => "O(n³)",
        .exponential => "O(2^n) - ORACLE RISK!",
        .unknown => "Unknown",
    };
    try std.fmt.format(w, "Estimated complexity: {s}\n", .{complexity_str});

    if (analysis.oracle_warning) {
        try w.writeAll("\n⚠️  WARNING: High commutation ratio detected!\n");
        try w.writeAll("    This may indicate oracle problem risk.\n");
        try w.writeAll("    Consider using linear types to prevent exponential blowup.\n");
    }

    return buf.toOwnedSlice(allocator);
}
