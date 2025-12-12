# HVM3 (Zig)

A Zig implementation of [HVM3](https://github.com/HigherOrderCO/HVM3) - the Higher-Order Virtual Machine based on Interaction Calculus.

## Features

- Bit-packed 64-bit term representation for efficient memory usage
- All core interaction rules: APP-LAM, DUP-SUP, DUP-LAM, etc.
- Full arithmetic operators (+, -, *, /, %, ==, !=, <, >, &, |, ^, <<, >>)
- Superpositions and duplications for optimal sharing
- Parser for `.hvm` files

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
$ ./zig-out/bin/hvm3 run examples/test.hvm
Result: #42
```

## Performance

The implementation is heavily optimized for speed with inlined hot paths, direct memory access, and compile-time debug check removal.

### Benchmark Results (ReleaseFast, Apple M4 Pro)

| Benchmark | Ops/sec |
|-----------|---------|
| Arithmetic operations | 112M |
| Beta reduction (λ application) | 205M |
| DUP+SUP annihilation | 214M |

### Debug vs Release Comparison

| Benchmark | Debug | ReleaseFast | Speedup |
|-----------|-------|-------------|---------|
| Arithmetic | 15M ops/s | 112M ops/s | 7.4x |
| Beta reduction | 19M ops/s | 205M ops/s | 10.6x |
| DUP+SUP annihilation | 23M ops/s | 214M ops/s | 9.5x |

Run benchmarks with:
```bash
./zig-out/bin/hvm3 bench
```

## License

MIT
