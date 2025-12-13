# HVM4 (Zig)

A Zig implementation of HVM4 - the Higher-Order Virtual Machine based on Interaction Calculus, with the new term layout, type system, and SupGen primitives.

## Features

### Core Runtime
- **HVM4 Architecture**: New 64-bit term layout `[8-bit tag][24-bit ext][32-bit val]`
- **50+ Term Types**: Constructors (C00-C15), primitives (P00-P15), stack frames, and more
- **17 Numeric Primitives**: ADD, SUB, MUL, DIV, MOD, AND, OR, XOR, LSH, RSH, NOT, EQ, NE, LT, LE, GT, GE
- **Optimal Duplication**: Label-based annihilation/commutation for optimal sharing
- **Stack Frame Evaluation**: Typed frames for WNF reduction (F_APP, F_MAT, F_SWI, F_OP2, etc.)
- **Lazy Collapse**: BFS enumeration of infinite superposition structures
- **Auto-Dup Foundation**: Variable use counting with label recycling
- **Configurable Runtime**: Adjustable heap, stack, workers, and optional reference counting

### Type System
- **Linear/Affine Types**: Prevent oracle problem with usage tracking (`\!x` linear, `\?x` affine)
- **Dependent Types**: Π (ALL), Σ (SIG), Self (SLF) types with proper verification
- **Type Annotations**: `{term : Type}` with convertibility checking (beta+eta equivalence)
- **Structural Equality**: `(=== a b)` for term comparison

### Performance
- **SIMD + Parallel**: Vectorized batch operations with multi-threaded execution
- **Parallel Reduction**: Work-stealing infrastructure for concurrent term reduction
- **SoA Memory Layout**: Structure-of-Arrays heap with SIMD tag scanning (~19B terms/sec)
- **Lock-Free Reduction**: AtomicHeap with CAS-based parallel reduction
- **Supercombinators**: Pre-compiled S, K, B, C combinators and Church numeral optimizations
- **Compiled Reducer**: Direct-threaded dispatch with comptime-generated handler tables (5x speedup)
- **Metal GPU Acceleration**: Pure Zig Metal bindings for macOS GPU compute (Apple Silicon)
- **GPU-Resident Reduction**: Full reduction loop on GPU with no CPU round-trips (2x sustained throughput)
- **Heterogeneous GPU Batching**: Mixed redex types processed in single dispatch
- **Async GPU Pipelining**: Triple-buffered command submission for overlapped execution

### Superposition Applications
- **SupGen Primitives**: Superposition-based program enumeration for discrete search
- **SAT Solver**: Boolean satisfiability via superposition (`src/sat.zig`)
- **Program Synthesis**: Grammar-based enumeration with spec-driven search (`src/synthesis.zig`)
- **Safe-Level Analysis**: Static and runtime detection of oracle problem patterns

### Development Tools
- **Linearity Checker**: Static analysis for oracle prevention (`src/linearity.zig`)
- **Cost Model**: Interaction costs and complexity estimation (`src/cost.zig`)
- **Debugger**: Step tracing, breakpoints, DOT/JSON export (`src/debug.zig`)
- **Profiler**: Per-interaction timing, hotspot detection, oracle warnings (`src/profile.zig`)
- **Supercompiler**: Partial evaluation and specialization (`src/supercomp.zig`)

## Building

Requires Zig 0.15+:

```bash
# Debug build (with safety checks)
zig build

# Release build (optimized for performance)
zig build -Doptimize=ReleaseFast
```

## Usage

```bash
# Run an HVM file
./zig-out/bin/hvm4 run examples/test.hvm

# Evaluate an expression
./zig-out/bin/hvm4 eval "(+ #21 #21)"

# Run tests
./zig-out/bin/hvm4 test

# Run parser tests
./zig-out/bin/hvm4 parse

# Run benchmarks
./zig-out/bin/hvm4 bench

# Test Metal GPU (macOS only)
./zig-out/bin/hvm4 gpu

# Show syntax examples
./zig-out/bin/hvm4 examples
```

## Syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `#N` | Number literal | `#42` |
| `'c'` | Character literal | `'x'` |
| `*` | Erasure | `*` |
| `\x.body` | Lambda | `\x.x` |
| `\!x.body` | Linear lambda (exactly once) | `\!x.x` |
| `\?x.body` | Affine lambda (at most once) | `\?x.*` |
| `(f x)` | Application | `((\x.x) #42)` |
| `(op a b)` | Binary operation | `(+ #3 #4)` |
| `&L{a,b}` | Superposition | `&0{#1,#2}` |
| `!&L{x,y}=v;k` | Duplication | `!&0{a,b}=sup;a` |
| `(?n z s)` | Switch/match | `(?#0 #100 \p.#200)` |
| `{t : T}` | Type annotation | `{#42 : Type}` |
| `(=== a b)` | Structural equality | `(=== #42 #42)` |
| `@LOG(v k)` | Debug logging | `@LOG(#42 result)` |
| `Type` | Type universe | `Type` |

## Example

```
// examples/test.hvm
(+ #21 #21)
```

```bash
$ ./zig-out/bin/hvm4 eval "(+ #21 #21)"
#42

$ ./zig-out/bin/hvm4 eval "(* (+ #2 #3) (- #10 #4))"
#30
```

## Architecture

### Term Layout (HVM4)

```
64-bit term: [8-bit tag][24-bit ext][32-bit val]

- tag: Term type (0x00-0xFF)
- ext: Extension field (label, constructor ID, primitive ID)
- val: Value field (heap location or immediate value)
```

### Term Types

| Category | Tags | Description |
|----------|------|-------------|
| Core | APP, VAR, LAM, SUP, ERA, REF | Basic lambda calculus |
| Collapse | CO0, CO1 | Duplication projections |
| Constructors | C00-C15 | Arity 0-15 constructors |
| Primitives | P00-P15 | Arity 0-15 primitives |
| Pattern Match | MAT, SWI | Constructor and numeric matching |
| Stack Frames | F_APP, F_MAT, F_SWI, F_OP2, etc. | Evaluation frames |
| Special | DUP, LET, USE, EQL, RED | Advanced features |
| Type System | ANN, BRI, TYP, ALL, SIG, SLF | Type annotations and dependent types |
| Linearity | LIN, AFF, ERR | Linear/affine types and type errors |

## Performance

Optimized following [VictorTaelin's IC techniques](https://gist.github.com/VictorTaelin/4f55a8a07be9bd9f6d828227675fa9ac):

- Cached heap/stack pointers in local variables
- Branch prediction hints (`@branchHint`) for hot paths
- Inlined critical interactions (APP-LAM, CO0/CO1-SUP, P02-NUM)
- Comptime dispatch table for interaction rules
- Stack frame-based WNF evaluation (no recursion)
- Batch heap allocations in interaction rules
- SIMD vectorized batch operations (8-wide vectors)
- Multi-threaded parallel execution (configurable workers)
- **Ultra-fast `reduce_fast` with fused beta-reduction chains**
- **Massively parallel interaction processing (310x speedup)**

### Interaction Net Benchmarks (ReleaseFast, Apple M4 Pro)

| Interaction | Ops/sec | Description |
|-------------|---------|-------------|
| DUP+NUM | ~158M | Trivial number duplication |
| MAT+CTR | ~143M | Pattern matching on constructors |
| Beta reduction | ~141M | APP+LAM interaction |
| DUP+LAM | ~135M | Lambda duplication |
| CO0+SUP annihilation | ~143M | Same-label collapse (optimal) |
| DUP+SUP commutation | ~122M | Different-label (creates 4 nodes) |
| Deep nested β (depth=10) | ~177M | Stress test |
| APP+SUP | ~119M | Superposition distribution |
| SWI+NUM | ~108M | Numeric switch |
| DUP+CTR | ~114M | Constructor duplication |

### Ultra-Fast Reduce Benchmarks

| Benchmark | Ops/sec | vs Serial |
|-----------|---------|-----------|
| reduce_fast: Beta reduction | ~149M | 1.06x |
| reduce_fast: CO0+SUP annihilation | ~176M | 1.23x |
| reduce_fast: DUP+LAM | ~183M | 1.35x |
| reduce_fast: Deep nested β | ~222M | **1.58x** |
| reduce_fast: P02 arithmetic | ~135M | 1.0x |

### Compiled Reducer Benchmarks

| Benchmark | Ops/sec | vs Serial |
|-----------|---------|-----------|
| reduce_compiled: Beta reduction | ~183M | 1.5x |
| reduce_beta_chain: Deep nested β | ~618M | **5.0x** |

### Massively Parallel Benchmarks

| Benchmark | Ops/sec | vs Serial |
|-----------|---------|-----------|
| SIMD batch add | ~2.7B | 20x |
| SIMD batch multiply | ~7.9B | 58x |
| SIMD 8-wide interactions | ~5.8B | **43x** |
| Parallel SIMD add (12 threads) | ~16B | 118x |
| Parallel SIMD multiply (12 threads) | ~14B | 105x |
| Pure parallel computation | ~31B | **227x** |
| **Parallel beta reduction** | **~42B** | **310x** |

### Metal GPU Benchmarks (Apple M4 Pro)

| Benchmark | Ops/sec | Notes |
|-----------|---------|-------|
| GPU batch add | ~544M | Simple arithmetic |
| GPU batch multiply | ~2.3B | 18x vs CPU SIMD |
| GPU heap transfer | ~2.3B terms/sec | Shared memory |
| GPU APP-LAM | ~224M | Beta reduction |
| GPU OP2-NUM | ~204M | 16 arithmetic ops |
| GPU SUP-CO0 | ~248M | Annihilation |
| **GPU-Resident Heterogeneous** | ~211M | Mixed types, single dispatch |
| **GPU-Resident Multi-Step** | ~380M | No CPU round-trips |
| **GPU-Resident Sustained** | **~424M** | **2x vs single-shot** |

### API

```zig
const hvm = @import("hvm.zig");

// Configuration
const config = hvm.Config{
    .heap_size = 4 << 30,           // 4GB heap
    .stack_size = 1 << 26,          // 64MB stack
    .num_workers = 64,              // 64 threads
    .enable_refcount = true,        // Optional RC
    .enable_label_recycling = true, // Recycle labels
};
var state = try hvm.State.initWithConfig(allocator, config);

// SIMD batch operations
hvm.batch_add(a, b, results);
hvm.batch_mul(a, b, results);
hvm.parallel_batch_add(a, b, results);
hvm.parallel_batch_mul(a, b, results);

// Safety analysis (oracle problem detection)
const analysis = hvm.analyze_safety(term);
if (analysis.level == .unsafe) {
    // Handle potential exponential blowup
}

// Runtime monitoring
hvm.reset_commutation_counter();
_ = hvm.reduce(term);
if (hvm.commutation_limit_reached()) {
    // Too many DUP+SUP commutations
}

// Memory management
state.resetHeap();  // Arena-style reset
const stats = state.getStats();

// GPU-Resident Reduction (macOS Metal)
const metal = @import("metal.zig");
var gpu = try metal.MetalGPU.init();
defer gpu.deinit();

// Allocate GPU-resident state (16M terms, 1M redexes)
try gpu.allocGpuResidentState(16 * 1024 * 1024, 1024 * 1024);

// Upload heap once
try gpu.uploadGpuHeap(heap_data);

// Run multiple reduction steps without CPU round-trips
const stats = try gpu.gpuReduceResident(redexes, alloc_ptr, max_steps);
// stats.interactions, stats.allocations, stats.new_redexes

// Or heterogeneous batch (mixed APP-LAM, CO0-SUP, OP2-NUM, ERA)
const batch_stats = try gpu.gpuInteractHeterogeneous(mixed_redexes, alloc_ptr);

// Download results once at end
try gpu.downloadGpuHeap(result_data);
```

Run benchmarks with:
```bash
./zig-out/bin/hvm4 bench
```

## References

- [HVM4 Technical Overview](https://gist.github.com/VictorTaelin/71e9b2ffeba3a9402afba76f8735d0b8)
- [HVM4 Commit History](https://gist.github.com/VictorTaelin/57d8082c7ffb97bd4755dd39815bd0cc)
- [IC Optimization to C](https://gist.github.com/VictorTaelin/ab20d7ec33ba7395ddacab58079fe20c)
- [VictorTaelin's X updates](https://x.com/VictorTaelin/status/1971591584916393984)

## License

MIT
