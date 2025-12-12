# HVM4 (Zig)

A Zig implementation of HVM4 - the Higher-Order Virtual Machine based on Interaction Calculus. Upgraded from HVM3 with the new term layout and expanded type system.

## Features

- **HVM4 Architecture**: New 64-bit term layout `[8-bit tag][24-bit ext][32-bit val]`
- **50+ Term Types**: Constructors (C00-C15), primitives (P00-P15), stack frames, and more
- **17 Numeric Primitives**: ADD, SUB, MUL, DIV, MOD, AND, OR, XOR, LSH, RSH, NOT, EQ, NE, LT, LE, GT, GE
- **Stack Frame Evaluation**: Typed frames for WNF reduction (F_APP, F_MAT, F_SWI, F_OP2, etc.)
- **Lazy Collapse**: BFS enumeration of infinite superposition structures
- **Auto-Dup Foundation**: Variable use counting and fresh label generation
- **SIMD + Parallel**: Vectorized batch operations with multi-threaded execution

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
./zig-out/bin/hvm3 run examples/test.hvm

# Evaluate an expression
./zig-out/bin/hvm3 eval "(+ #21 #21)"

# Run tests
./zig-out/bin/hvm3 test

# Run parser tests
./zig-out/bin/hvm3 parse

# Run benchmarks
./zig-out/bin/hvm3 bench

# Show syntax examples
./zig-out/bin/hvm3 examples
```

## Syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `#N` | Number literal | `#42` |
| `'c'` | Character literal | `'x'` |
| `*` | Erasure | `*` |
| `\x.body` | Lambda | `\x.x` |
| `(f x)` | Application | `((\x.x) #42)` |
| `(op a b)` | Binary operation | `(+ #3 #4)` |
| `&L{a,b}` | Superposition | `&0{#1,#2}` |
| `!&L{x,y}=v;k` | Duplication | `!&0{a,b}=sup;a` |
| `(?n z s)` | Switch/match | `(?#0 #100 \p.#200)` |

## Example

```
// examples/test.hvm
(+ #21 #21)
```

```bash
$ ./zig-out/bin/hvm3 eval "(+ #21 #21)"
#42

$ ./zig-out/bin/hvm3 eval "(* (+ #2 #3) (- #10 #4))"
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

## Performance

Optimized following [VictorTaelin's IC techniques](https://gist.github.com/VictorTaelin/4f55a8a07be9bd9f6d828227675fa9ac):

- Cached heap/stack pointers in local variables
- Branch prediction hints (`@branchHint`) for hot paths
- Inlined critical interactions (APP-LAM, CO0/CO1-SUP, P02-NUM)
- Stack frame-based WNF evaluation
- SIMD vectorized batch operations (4-wide vectors)
- Multi-threaded parallel execution (12 cores on M4 Pro)

### Benchmark Results (ReleaseFast, Apple M4 Pro)

| Benchmark | Ops/sec | Notes |
|-----------|---------|-------|
| Single-threaded arithmetic | 95M | P02 binary primitives |
| Beta reduction (λ application) | **185M** | Inlined APP-LAM |
| CO0+SUP annihilation | 157M | Collapse projections |
| SIMD batch add | 1.3B | Vectorized, single-thread |
| SIMD batch multiply | 3.1B | Vectorized, single-thread |
| **Parallel SIMD add** | **12.4B** | **131x speedup** |
| **Parallel SIMD multiply** | **14.1B** | Multi-threaded + SIMD |

### Batch Operations API

For maximum performance on bulk numeric operations:

```zig
const hvm = @import("hvm.zig");

// SIMD batch operations (single-threaded)
hvm.batch_add(a, b, results);    // results[i] = a[i] + b[i]
hvm.batch_mul(a, b, results);    // results[i] = a[i] * b[i]
hvm.batch_sub(a, b, results);    // results[i] = a[i] - b[i]

// Parallel SIMD (multi-threaded, 12 cores)
hvm.parallel_batch_add(a, b, results);
hvm.parallel_batch_mul(a, b, results);
```

Run benchmarks with:
```bash
./zig-out/bin/hvm3 bench
```

## References

- [HVM4 Technical Overview](https://gist.github.com/VictorTaelin/71e9b2ffeba3a9402afba76f8735d0b8)
- [HVM4 Commit History](https://gist.github.com/VictorTaelin/57d8082c7ffb97bd4755dd39815bd0cc)
- [IC Optimization to C](https://gist.github.com/VictorTaelin/ab20d7ec33ba7395ddacab58079fe20c)
- [VictorTaelin's X updates](https://x.com/VictorTaelin/status/1971591584916393984)

## License

MIT
