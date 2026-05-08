# Tinygrad

Tinygrad is a minimalist deep learning framework (~4,300 lines of core code) created by George Hotz. It sits between micrograd (educational scalar engine) and PyTorch (industrial framework) -- small enough to read in a weekend, powerful enough to train real models. The entire design revolves around one idea: a computation graph of operations that gets lazily compiled into fused GPU kernels.

Source: https://github.com/tinygrad/tinygrad

## Core Design Philosophy

Three constraints shape every decision in tinygrad:

1. **Stay small.** The core must remain under ~10k lines. No external dependencies for the engine. If a feature bloats the codebase, it doesn't ship.
2. **Lazy by default.** Operations are recorded, not executed. Execution is deferred until explicitly requested (`.realize()`, `.numpy()`). This gives the compiler a full view of the computation before generating code.
3. **Backends are thin.** Each hardware target (CUDA, Metal, OpenCL, AMD, CPU, WebGPU, ...) only needs to implement ~25 low-level primitives. The compiler does the heavy lifting.

## Architecture Overview

Tinygrad has four layers, each feeding into the next:

```
 Tensor API          (what the user writes)
    │
    ▼
 UOp Graph           (lazy IR -- records what to compute)
    │
    ▼
 Scheduler + Codegen (fuses ops, generates device-specific kernels)
    │
    ▼
 Runtime             (allocates memory, launches kernels on device)
```

### Layer 1: Tensor API (Frontend)

The `Tensor` class provides a PyTorch-like interface -- `Tensor.randn()`, `t1 + t2`, `t.reshape()`, `t.sum()`, etc. But unlike PyTorch, these calls do **no computation**. Every operation creates a new node in a lazy computation graph.

Internally, a `Tensor` is a thin wrapper around a single `UOp`:

```python
class Tensor:
    def __init__(self, ...):
        self.uop = ...  # the underlying UOp graph node
```

When you write `c = a + b`, Python calls `Tensor.__add__`, which creates a new `UOp` of type `ADD` pointing at the UOps inside `a` and `b`, wraps it in a new `Tensor`, and returns it. No addition happens. The graph just grows.

### Layer 2: UOp Graph (Intermediate Representation)

The **UOp** (micro-operation) is the fundamental abstraction -- not the Tensor. This is the key architectural difference from PyTorch.

Each UOp is an immutable node with:
- An **op type** from the `Ops` enum (ADD, MUL, LOAD, STORE, RESHAPE, REDUCE_AXIS, ...)
- A **src** tuple pointing to input UOps (the edges of the DAG)
- A **dtype** (data type)
- Arbitrary metadata

UOps are **deduplicated via weak-reference caching**. If you create the exact same operation twice (same op, same inputs, same dtype), you get back the same object. This means identical subexpressions are automatically shared -- a form of common subexpression elimination that happens for free.

The UOp categories cover the full range of deep learning operations:

| Category | Examples | Purpose |
|---|---|---|
| Elementwise math | ADD, MUL, EXP2, LOG2, SQRT, SIN, CAST | Pointwise operations |
| Memory | LOAD, STORE, BUFFER, COPY | Reading/writing device memory |
| Movement | RESHAPE, EXPAND, PERMUTE, PAD, SHRINK, FLIP | Shape manipulation without data copy |
| Reduction | REDUCE_AXIS | Sum, max, etc. along axes |
| Control | RANGE, IF, SINK | Loop structure, dependencies |
| Special | WMMA | Hardware-specific ops (tensor cores) |

The entire computation -- forward pass, loss, backward pass -- is a single DAG of these nodes.

### Layer 3: Scheduler + Compiler

When `.realize()` is called, the UOp DAG must become executable code. This happens in two phases.

**Scheduling** breaks the DAG into discrete kernels:

1. Walk the graph from the SINK (root) node
2. Analyze data dependencies between operations
3. Decide which operations can be **fused** into a single kernel
4. Produce a list of `ExecItem` objects, each representing one GPU kernel launch

**Kernel fusion** is the primary optimization. Consider:

```python
a = Tensor.randn(1024, 1024)
b = Tensor.randn(1024, 1024)
c = (a + b) * 2.0
c.realize()
```

Without fusion, this would be two kernels: one for addition, one for multiplication. The addition kernel writes ~4MB to memory, and the multiplication kernel reads it back. With fusion, tinygrad generates a **single kernel** that does `(a[i] + b[i]) * 2.0` in one pass -- the intermediate result never touches memory. This is a massive win because GPU compute is fast but memory bandwidth is the bottleneck.

The scheduler decides fusion boundaries based on:
- Whether the output of one operation is the sole input to the next
- Memory access patterns (can both operations share the same loop structure?)
- Device limits (register pressure, shared memory size)

**Codegen** then converts each fused kernel from UOps into device-specific source code:

```
UOp graph
  → Pattern-matcher rewrites (simplify, optimize)
  → Loop structure (RANGE nodes become for-loops)
  → Memory indexing (compute strides, offsets)
  → Device-specific rendering (C, CUDA, Metal, LLVM IR, PTX, WGSL, ...)
  → Compilation to binary
```

The compiler is built around **PatternMatcher** -- a declarative rule engine. Each optimization pass is a set of pattern-matching rules:

```python
PatternMatcher([
    (UPat(Ops.ADD, src=(UPat.var("x"), UPat.const(0))), lambda x: x),  # x + 0 → x
    (UPat(Ops.MUL, src=(UPat.var("x"), UPat.const(1))), lambda x: x),  # x * 1 → x
    ...
])
```

This makes the compiler modular and readable -- each pass is data-driven, not a tangle of if-statements.

### Layer 4: Runtime

Each backend provides:
- **Allocator**: Manages device memory (with LRU caching for buffer reuse)
- **Compiler**: Compiles rendered source to binary (results are cached)
- **Runtime**: Launches compiled kernels on the device

The runtime is accessed through `Device["METAL"]`, `Device["CUDA"]`, etc. At execution time, the engine:
1. Allocates output buffers
2. Launches each `ExecItem`'s compiled kernel
3. Handles cross-device copies if needed

A JIT layer caches compiled schedules. First run pays compilation cost; subsequent runs with the same shapes reuse cached binaries and even batch multiple kernels into a single "device graph" submission to reduce CPU-GPU synchronization overhead.

## Autograd

Tinygrad's autograd is **operation-centric**: gradient rules are defined per UOp type, not per tensor. There is no `.grad` attribute on tensors accumulating gradients during the forward pass. Instead:

1. The forward pass builds a UOp DAG
2. `backward()` walks the DAG in reverse topological order
3. For each UOp, a PatternMatcher (`pm_gradient`) applies the appropriate gradient rule
4. New UOps are created representing the backward computation
5. These backward UOps are themselves part of the lazy graph and get compiled/fused like everything else

Key gradient rules:

| Forward Op | Backward Rule |
|---|---|
| ADD(a, b) | grad flows unchanged to both a and b |
| MUL(a, b) | grad * b flows to a, grad * a flows to b |
| REDUCE_AXIS | grad is expanded back to original shape |
| RESHAPE | grad is reshaped back (inverse reshape) |
| PERMUTE | grad is permuted back (inverse permutation) |
| PAD | grad is shrunk (inverse of pad) |
| EXP2 | grad * out (where out = exp2(input)) |

The backward pass produces new UOps, meaning gradients are lazy too. The gradient computation is fused and optimized by the same compiler pipeline as the forward pass. This is a significant advantage over eager autograd -- the compiler can see both forward and backward operations and optimize across them.

### Comparison with micrograd's autograd

In micrograd, each `Value` node stores a `_backward` closure that directly mutates `.grad` on its parents. The backward pass calls these closures. It's simple and direct, but:
- Closures capture Python scope (slow, no optimization)
- Gradients are computed eagerly (no fusion opportunity)
- Each node must store its backward function (memory overhead)

In tinygrad, gradients are just more UOps in the same graph. The compiler doesn't distinguish between "forward" and "backward" operations -- it fuses and optimizes all of them uniformly.

## Operation-Centric vs Tensor-Centric Graphs

This is the fundamental architectural divide between tinygrad and micrograd/PyTorch.

**Micrograd (tensor-centric):** Nodes are `Value` objects (scalars). Edges are implicit in `_prev` sets. The operation is just a label (`_op`). The graph answers: "what values exist, and what produced them?"

**Tinygrad (operation-centric):** Nodes are `UOp` objects (operations). The inputs (`src` tuple) point to other UOps. There is no separate "tensor" node in the graph -- a tensor is just the UOp that computes it. The graph answers: "what operations need to run, and in what order?"

Why does this matter?

1. **Fusion is natural.** Operations are first-class, so merging two adjacent operations into one kernel is a graph rewrite -- collapse two nodes into one. In a tensor-centric graph, you'd need to look "through" the tensors to find fusable operation pairs.

2. **No redundant storage.** A tensor-centric graph creates nodes for intermediate values that may never materialize in memory. In an operation-centric graph, intermediate results are compiler temporaries inside fused kernels.

3. **Compiler friendliness.** The UOp graph is essentially an IR (intermediate representation) -- it maps naturally to the kind of graphs that compilers already know how to optimize (SSA form, dataflow graphs).

The tradeoff is usability: tensor-centric graphs are more intuitive for users who think in terms of "I have a matrix, I transform it." Tinygrad hides this behind the `Tensor` API so users never touch UOps directly.

## Movement Operations and Views

A critical optimization: **movement operations don't move data.**

When you call `t.reshape(2, 3)` or `t.permute(1, 0)` or `t.expand(4, 3)`, tinygrad creates a UOp node recording the shape change, but no memory is allocated and no data is copied. These are "view" operations -- they change how the same underlying buffer is indexed.

The compiler resolves views during codegen by computing the correct stride/offset arithmetic. A transposed matrix doesn't get physically transposed; instead, the generated kernel swaps the loop index order.

This means chains of reshapes, permutations, and expansions are essentially free -- they're just metadata that the compiler folds into memory addressing.

## Multi-Backend via Renderers

The codegen pipeline is parameterized by a **Renderer** class. Each device provides its own:

| Backend | Renderer | Target |
|---|---|---|
| CPU | CStyleRenderer | C code |
| CUDA | CUDARenderer | CUDA C |
| METAL | MetalRenderer | Metal Shading Language |
| AMD | AMDRenderer | AMD ISA |
| NV | NVRenderer | PTX |
| CL | OpenCLRenderer | OpenCL C |
| WEBGPU | WGSLRenderer | WGSL (WebGPU) |
| LLVM | LLVMRenderer | LLVM IR |

The renderer only handles the final step -- converting optimized UOps into text/binary. All optimization passes (fusion, simplification, loop structuring) are shared across backends. This is how tinygrad supports many devices without code duplication.

Adding a new backend requires:
1. A renderer (~200 lines) that maps UOps to the target language
2. A runtime (~200 lines) that compiles and launches kernels
3. An allocator (~100 lines) for device memory

## Putting It All Together: Full Execution Trace

```python
a = Tensor([1.0, 2.0, 3.0])    # UOp: CONST → BUFFER
b = Tensor([4.0, 5.0, 6.0])    # UOp: CONST → BUFFER
c = (a + b).relu()              # UOp: ADD → WHERE (relu = max(0, x))
loss = c.sum()                  # UOp: REDUCE_AXIS

loss.backward()                 # Adds gradient UOps to the graph
optimizer.step()                # Adds parameter update UOps

# Nothing has executed yet. The entire forward + backward + update is one UOp DAG.

loss.realize()  # NOW:
# 1. Scheduler analyzes the full DAG
# 2. Fuses add+relu into one kernel, reduce into another
# 3. Gradient kernels are fused where possible
# 4. Codegen produces (e.g.) 3 Metal kernels
# 5. Kernels are compiled to GPU binary
# 6. Runtime launches kernels in dependency order
# 7. Result buffer is ready on device
```

The key insight: because everything is lazy, the compiler sees the **entire training step** as one graph and optimizes globally. Eager frameworks like PyTorch must optimize each operation in isolation (though `torch.compile` partially addresses this).

## Comparison with micrograd and PyTorch

| | micrograd | tinygrad | PyTorch |
|---|---|---|---|
| Graph type | Tensor-centric | Operation-centric | Tensor-centric (eager), Operation-centric (compile) |
| Execution | Eager | Lazy | Eager (default) |
| Data unit | Scalar `Value` | N-dimensional `UOp` | N-dimensional `Tensor` |
| Autograd | Closures on nodes | PatternMatcher rules on UOps | C++ engine with per-op backward functions |
| Fusion | None | Automatic via scheduler | None (eager) / torch.compile |
| Backends | Python only | ~10 backends | CUDA, CPU (primarily) |
| Compiler | None | Built-in (UOp → kernel) | Separate (TorchInductor) |
| Code size | ~100 lines | ~4,300 lines | ~2M+ lines |
| Purpose | Education | Hackable framework | Production |

## What picograd Takes From Tinygrad

The path from micrograd to a real framework requires three things that tinygrad demonstrates:

1. **Replacing scalars with tensors.** The C tensor library in `/src/tensor/tensor.c` is this step -- an n-dimensional array with shape, stride, and flat data storage. Tinygrad's UOps operate on these.

2. **Lazy evaluation + compilation.** Instead of executing each Python operation immediately, record the operations and compile them into efficient kernels. This is where the real performance comes from.

3. **Minimal but complete op set.** Tinygrad proves that ~25 low-level operations (load, store, add, mul, reduce, ...) are sufficient to express all of deep learning. Everything else is composed from these.

## Resources

- Tinygrad repo: https://github.com/tinygrad/tinygrad
- George Hotz tinygrad streams: https://www.youtube.com/@gaborhotz
- "You can be mass produced" (tinygrad philosophy talk): https://www.youtube.com/watch?v=ci6iFqRSJcQ
- Tinygrad documentation: https://docs.tinygrad.org/
