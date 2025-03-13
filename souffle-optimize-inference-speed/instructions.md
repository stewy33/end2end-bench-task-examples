# Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions

## Task Overview

Your task is to develop a compiler optimization approach for deep neural network inference that automatically optimizes across operator boundaries. We are calling this approach Souffle. You'll evaluate performance on five models (BERT, ResNeXt, EfficientNet, an LSTM, SwinTransformer, MMoE) and against other inference framework baselines (TensorXLA, TensorRT, Rammer, MindSpore Apollo, and IREE).

## Background

Deep neural networks are typically expressed as computational graphs in frameworks like TensorFlow and PyTorch. While these frameworks abstract away hardware complexity, they present challenges for performance optimization during inference. As a result, efforts have been made to perform optimizations across the operator boundaries to increase parallelism, decrease memory access traffic or utilize memory bandwidth more efficiently. One promising approach is *operator/kernel fusions*, which involves merging multiple operators into a single kernel to enable analysis and optimizations across operators. Typically, these methods use a bottom-up strategy by first performing operator fusion in the graph representation to merge multiple operators into a partition and then generating an optimized kernel for each partition. However, a key challenge is determining the optimal boundaries of partitions or which operators should be fused together. It is often not possible to do this well in a bottom-up way, which must make partitioning decisions before understanding higher-level code characteristics. As a result, these methods may fail to identify data reuse opportunities between different types of operators, inefficiently handle reduction operators by storing large intermediate tensors in global memory, and misplace operators into different kernels, leading to extra memory access overhead and preventing otherwise possible optimizations.

Souffle addresses these challenges with a novel top-down approach:
1. Souffle first processes the entire computation graph as a single, merged kernel. It represents the computations via tensor expressions.
2. It then divides the graph into partitions (i.e. subprograms, through a global analysis from the top-level, considering data reuse in shared memory/registers and the generated instructions when determining partitioning boundaries). Each partition is be organized into a kernel.
3. Finally, at the bottom level, Souffle performs a series of transformations for each subprogram to simplify the tensor expression and eliminate redundant memory access for the corresponding kernel.


### A Motivating Example

Consider optimizing a standard BERT model block. This subgraph contains representative DNN operators like general matrix multiplication GEMM, reshape, permutation, element-wise arithmetic operators like add or exp, and reduction operators like reduce_sum. The compiler maps these operators to individual kernels, which significantly impacts performance.


Existing baselines like TensorRT and Apollo miss several opportunities for optimization:

1. *Fail to explore optimization between memory and compute-intensive kernels.* BERT requires element-wise memory operators, e.g. reshape and permutation. TensorRT and Apollo leverage manually crafted rules to fuse adjacent element-wise memory operators together while both of them fail to further perform optimization between the fused operators and their precedent computation operators, e.g. (GEMM) operators. Souffle performs optimization between memory and compute-intensive kernels, and eventually eliminates all element-wise memory operators. In summary, manually crafted rules cannot cover a diverse set of computation patterns and miss the optimization opportunity in this case.

2. *Suboptimal fusion strategy for reduction operators.* Both TensorRT and Apollo choose to map the GEMM and the reduction operator to separate kernels, which requires storing the entire tensor data that reduction operators rely on to expensive global memory before reduction occurs. Souffle fuses reduction operators with adjacent computation-intensive operators, such as when sandwhiched between GEMMs. This is achieved through a two-phase reduction approach: performing partial reduction in a per-block fashion and using atomicAdd for global reduction. As a result, the entire tensor data can be kept on-chip, with only the partial result stored in global memory. A global synchronization (e.g. grid synchronization in CUDA) is inserted to synchronize running blocks. Souffle can cache the output of operators on-chip for reuse in other operators.

3. *Poor optimizations across computation-intensive kernels.* Like many other DNN frameworks, TensorRT and Apollo try to fuse multiple computation-intensive operators of the same type, but fail to capitalize on the opportunities across different types of operators. E.g. two dependent GEMM operators may execute asynchronous memory copies and tensor core computations when they are grouped to kernels under two different strategies. The first is to map the GEMM operators into two separate kernels, as they do not consider fusing compute-intensive operators. The second is to map them to a single kernel. TensorRT and Apollo use the former, and Souffle uses the latter. By putting two GEMM operators into one single kernel, Souffle allows the pipeline execution of loading weights of GEMM2 while computing GEMM1. Souffle is designed to take such cross-operator pipeline optimizations.

Based on the observations outlined earlier, there is a need to analyze DNN models to fuse operators, perform automatic transformations on tensor operations, and optimize within a fused kernel. A more effective kernel fusion strategy makes extracting crucial tensor information such as live range and tensor data reuse possible. This information can then be used to analyze the fine-grained dependencies at the element-wise level, leading to better kernel-level optimization.

## Technical Approach

The Souffle system consists of several key components:

### 1. Tensor Expression (TE) Lowering

Mapping complex operators to simple TEs simplify analysis and allow for precise understanding of data flow semantics. For instance, a softmax operator can be represented by two TEs with simpler data dependence relationships: one is a one-relies-on-many TE (reduction), and the other is a one-relies-on-one TE (element-wise). Since Souffle’s analysis is conducted on the TEs without making any assumptions of low-level library calls, it can optimize across complex operators, even when the operators have complex data dependency like many-to-many, when other methods fail to do so. Souffle will use TVM's tensor expression representation for analysis and optimization. TE uses a pure functional language to describe tensor computation, allowing for individual computation of each output tensor element.

Start by converting high-level DNN operators to tensor expressions that precisely describe computation. For example, a residual connection with gating operation would be represented in pytorch as:

```python
# A matrix multiplication (GEMM) between input tensor I0 and weight tensor W0 (compute gating weights)
O0 = torch.matmul(I0, W0)  # Shape: (64, 64)
# A sigmoid activation applied to O0 (apply sigmoid to gating)
O1 = torch.sigmoid(O0)  # Shape: (64, 64)
# Another matrix multiplication between O1 and weight tensor W2 (gating on residual stream)
O2 = torch.matmul(O1, W2)  # Shape: (64, 64)
# An element-wise addition between O0 and O2 (residual connection)
O3 = O0 + O2  # Shape: (64, 64)
# A final matrix multiplication that changes the output dimension (project to output dimension)
O4 = torch.matmul(O3, W4)  # Shape: (64, 256)
```

and in the TVM tensor expression language as:

```python
rk = te.reduce_axis((0, 64),)
TE0: O0 = te.compute((64,64), lambda i, j: te.sum(I0[i,rk]*W0[rk,j]),axis=[rk])
TE1: O1 = te.compute((64,64), lambda i, j: te.sigmoid(O0[i, j]))
TE2: O2 = te.compute((64,64), lambda i, j: te.sum(O1[i,rk]*W2[rk,j]),axis=[rk])
TE3: O3 = te.compute((64,64), lambda i, j: O0[i,j] + O2[i,j])
TE4: O4 = te.compute((64,256),lambda i, j: te.sum(O4[i,rk]*W4[rk,j]),axis=[rk])
```

Here, five operators are lowered to five TEs, but more complex operators may be represented by more TEs. This gives a unified representation at the level of basic operations and data dependencies that makes analysis and optimization easier than using pytorch operations.

### 2. Global Computation Graph Analysis

The lowered TE program is passed to the Souffle analysis module, which performs a two-level analysis on the TE program:
- **Tensor-level analysis**: Extract tensor info like shapes, live range, and computation intensity of each TE. Find tensors used by multiple TEs to identify reuse opportunities
- **Element-wise dependency analysis**: Analyze the fine-grained dependencies between input and output tensors of each TE. For example, categorize dependencies as "one-relies-on-one" (element-wise operations) or "one-relies-on-many" (reductions)
    - Categorizing dependencies into these two categories simplifies dependence analysis compared to source code or operator-level analysis that other kernel fusion methods use.

You should also classify each TE as either:
- **Compute-intensive**: High arithmetic intensity (number of arithmetic instructions/memory accesses >= c)
- **Memory-intensive**: Low arithmetic intensity (number of arithmetic instructions/memory accesses < c)

where c is a threshold you should choose.

In the example above, we categorize TEs as:
- TE0, TE2, TE4: one-relies-on-many, compute-intensive
- TE1, TE3: one-to-one, memory-intensive

And we find data reuse opportunities:
- O0: reused by TE1, TE3

### 3. Resource-Aware Program Partitioning

Partition the program into subprograms based on resource usage and transform each subprogram into a kernel.
- Try to generate large kernels to maximize data reuse and reduce kernel launches.
- However, using global synchronization imposes a hardware constraint on the maximum number of blocks that can be used in a kernel. If we can't meet this constraint, we should partition the program into multiple subprograms.
- Use compute-intensive operators as candidate partitioning points
- Design an algorithm to effectively partition the program into subprograms subject to constraints and optimizing performance.

In the example above we'd partition the program into two subprograms:

```python
# Generated schedule for TE0
# Suppose the global synchronization API supports at most 48 blocks
TE0: s = te.create_schedule(O0.op)
TE0: io, ii, jo, jj, ko, ki = s.split(i, j, k,16, 16, 16)
TE0: s.reorder(io, jo, ko, ii, jj, ki)
TE0: SI = s.cache_read(I, ko)
TE0: s.bind(io, blockIdx.x)
# ... TE1, TE2, TE3 ...

# Generated schedule for TE4
TE4: s = te.create_schedule(O4.op)
TE4: io, ii, jo, jj, ko, ki = s.split(i, j, k,16, 16, 16)
TE4: s.reorder(io, jo, ko, ii, jj, ki)
TE4: SI = s.cache_read(I, ko)
TE4: s.bind(io, blockIdx.x)
```

If the global synchronization hardware API supports at most 48 blocks, we can partition the program into two subprograms (since TE4 needs 64 blocks > 48 block limit, and should be in its own kernel)
- 0: TE0, TE1, TE2, TE3
- 1: TE4

### 4. Semantic-Preserving TE Transformations

The subprograms + data-flow analysis + tensor information are sent to the TE transformation module for optimization. Apply two types of transformations to each subprogram:

- **Horizontal transformation**: Merge independent TEs with similar input-output shapes to increase parallelism
    - Try to identify TEs that can be fused side-by-side
    - Can try to concatenate multiple TEs and use predicates to control the execution of the merged TE
    - If concatenation isn't possibly, can try if_else statements instead, like Rammer uses
- **Vertical transformation**: Combine consecutive TEs with direct dependencies to reduce memory accesses
    - Focus on one-relies-on-one TEs (memory-intensive) and use quasi-affine maps to represent compositions of operations through index mapping functions

Finally, find an efficient way to combine the one-relies-on-one TEs into compute-intensive TEs when possible.

In the running example above, the TE transformation for subprogram 0 (TE0, TE1, TE2, TE3) might be transformed into:
```python
TE0: s.reorder(io, jo, ko, ii, jj, ki)
TE1: s = te.create_schedule(O1.op)
TE1: io, ii, jo, jj = s.split(i, j, 16, 16) # Inherit tile shape from TE0's schedule
TE1: s[O1.op].compute_at(jo) # Move computation of TE1 into TE0's loop
# ... other TEs ...
# All memory-intensive TEs are fused into compute-intensive TEs so TE1 is moved into TE0 and TE3 is moved into TE2
```

Let us provide two more examples for horizontal and then vertical transformations:
```python
# shape A1:(4,8),B1:(8, 16),A2:(2, 8),B2:(8, 16)
# Original TEs for two GEMMs
rk = te.reduce_axis((0, 8), name="rk")
C1 = te.compute((4,16), lambda i,j:te.sum(A1[i,rk]*B1[rk,j],axis=[rk]))
C2 = te.compute((2,16), lambda i,j:te.sum(A2[i,rk]*B2[rk,j],axis=[rk]))

# Horizontally transformed TE
C = te.compute((4+2, 16), lambda i, j:
    te.sum(tir.if_then_else(i<4, A1[i, rk], A2[i, rk]) *
    tir.if_then_else(i<4, B1[rk, j], B2[rk, j]), axis=[rk]))
```

```python
# Original TEs
A = te.placeholder((4, 8))
B = te.compute((4,8),lambda i,j:
    tir.if_then_else(A[i,j]>0, A[i,j], 0)) # Relu
C = te.compute((2,4),lambda i,j:B[2*i,j]) # Strided_slice
D = te.compute((4,2), lambda i,j:C[j,i]) # Permute

# Semantic preserving vertically transformed TE
D = te.compute((4,2), lambda i,j:
    tir.if_then_else(A[j, 2*i]>0, A[j,2*i], 0)) 
```

### 5. Schedule Generation, Optimization, and Code Generation

Transformed TE subprograms are sent to TVM's Ansor to generate schedules for the subprograms.
- Schedule memory-intensive TEs according to their compute-intensive TEs if you can
- Merge schedules within a subprogram into a single function represented by TensorIR for joint optimizations of instructions and data reuse
- Instruction-level optimizations: Regroup instructions within a fused subprogram containing multiple original operators to execute memory and arithmetic instructions in parallel. Do this by the scheduling load/store and computation instructions for pipeline execution across operator boundaries
- Tensor reuse optimizations: Use a simple Least Recently Used (LRU) policy to cache tensors from shared memory
- Finally, the optimized subprogram is passed to the back-end code generator to produce CUDA kernels

In our running example,

```python
# Merge TensorIR of compute-intensive TEs into one function
Fn_TE_Subprogam_0(I0, W0, O0, O1, W2, O2):
    shared SI0[16][16], SW0[16][16], SO0[16][16], SO1[16][16]
    shared SI2[16][16], SW2[16][16], SO2[16][16]
    if blockIdx.x < 4 and blockIdx.y < 4: # TE0 & TE1
        for ko in range 4:
            # Global to Shared (ldg2s means load from global memory to shared memory)
            ldg2s(SI0, I0[blockIdx.x*16:blockIdx.x*16+16][...])
            ldg2s(SW0, W0[...][blockIdx.y*16:blockIdx.y*16+16])
            wmma_16x16(SO0, SI0, SW0) # wmma means warp matrix multiply and accumulate
        SO1 = sigmoid(SO0) # SO0 used
        # Shared To Global (sts2g means store from shared memory to global memory)
        sts2g(O0, SO1)

    # Global synchronization to maintain dependencies across dependent TEs
    grid.sync()

    if blockIdx.x < 4 and blockIdx.y < 4: # TE2 & TE3
        for ko in range 4:
            ldg2s(SI2, O1[blockIdx.x*16:blockIdx.x*16+16][...])
            ldg2s(SW2, W2[…][blockIdx.y*16:blockIdx.y*16+16])
            wmma_16x16(SO2, SI2, SW2)
        SO2 = add(SO0,SO2) # SO0 reused across TE boundary
        sts2g(O2, SO2)
```

## Implementation Requirements

Implement the following components:

1. TE lowering module to convert operators to tensor expressions
2. Global analysis module for dependency tracking and classification
3. Program partitioning module based on resource constraints
4. TE transformation module for horizontal and vertical transformations
5. Schedule optimization module with TVM integration
6. Code generation module for CUDA kernels

Souffle should support elementwise operators, broadcasts, and reductions (e.g. reduce_sum, GEMM, and Conv), reorganized operators like reshape, and shuffle operators like permute. However, it does not need to support non linear algebra operators like TopK or Conditional.

It is crucial that Souffle produces correct results that should exactly match the results of the original operators.

### Model and Baseline Integration

Integrate and test Souffle with:

1. **BERT-Base**: Base version with 12 layers.
2. **ResNeXt**: 101 layers, bottleneck_width=64d.
3. **EfficientNet-b0**: Base model with 12 layers.
4. **SwinTransformer**: Base version with patch=4, window_size=7.
5. **MMoE**: Base model with mixture-of-experts architecture from [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007). See [pytorch version](https://github.com/shenweichen/DeepCTR-Torch/blob/master/deepctr_torch/models/multitask/mmoe.py) and [tensorflow version](https://raw.githubusercontent.com/drawbridge/keras-mmoe/refs/heads/master/mmoe.py).

And against the following inference framework baselines:

1. TensorXLA from Tensorflow 2.10.0
2. [TensorRT v8.6.1](https://github.com/NVIDIA/TensorRT/tree/v8.6.1)
3. [Rammer v0.3](https://github.com/microsoft/nnfusion/tree/v0.3)
4. [MindSpore Apollo r1.3](https://github.com/mindspore-ai/mindspore/tree/r1.3) (MindSpore is a large deep learning training/inference framework that includes lots of functionality, including the Apollo inference engine)
5. [IREE at candidate-20231125.719](https://github.com/iree-org/iree/tree/candidate-20231125.719)
6. [Ansor v0.8](https://github.com/apache/tvm/tree/v0.8.0)

## Target Results

### 1. End-to-End Execution Time

Produce a table measuring the end-to-end latency of a single forward pass with batch size 1 for all models and inference pipelines.
- Models: BERT, ResNeXt, EfficientNet, SwinTransformer, MMoE
- Inference pipelines: TensorXLA, Ansor, TensorRT, Rammer, MindSpore Apollo, IREE, Souffle

### 2. Ablation Study on E2E Execution Time

Test and report on the impact of each Souffle component:
1. V0 (Ansor baseline): TVM + Ansor without Souffle
2. V1: Add horizontal TE transformation
3. V2: Add vertical TE transformation
4. V3: Add global synchronization with global synchronization API
5. V4: Add subprogram-level optimization (full Souffle)

Create a table measuring the end-to-end latency for each of these 5 configurations on all models (BERT, ResNeXt, EfficientNet, SwinTransformer, MMoE).

### 3. Kernel Calls and Global Memory Transfer Amount

Create a table measuring the number of GPU kernel calls and total global memory transfer size (in MB) of a single forward pass with batch size 1 for:
- Models: BERT, ResNeXt, EfficientNet, SwinTransformer, MMoE
- Inference pipelines: MindSpore Apollo, TensorXLA, Souffle

### 4. Souffle Submodule Latency Breakdown

The EfficientNet-B0 model has 16 MBConv blocks. Some of these are repeated however, giving 10 unique MBConv blocks (MB0 - MB9). Again using an input with batch size 1, please report the latency of each of the 10 MBConv submodules in EfficientNet-B0, under different optimization configurations.
- Unfused: generate each TE to one kernel
- Fused: fuse using Ansor's fusion
- Global-sync: use Souffle to generate the whole sub-module to one kernel (using global synchronization) but without any data reuse
- Data-reuse: additionally apply Souffle's data reuse optimizations to the global-synced kernel

Show this as a grouped bar chart with the x-axis being the MBConv block (and group the 4 configurations together) and the y-axis being the block's latency.

### 5. Example Computation Graph on a BERT Block

Produce a visualization comparing the original computation graph of a BERT block with the computation graph after Souffle's optimizations. The visualization should annotate each operator, have edges representing data flow, and indicate what is fused (horizontally and vertically) and when global synchronization is used. Feel free to split this up into multiple images if needed.

### 6. Correctness Verification

For each model, run a correctness verification by comparing the output of Souffle with a standard pytorch implementation. Simply produce a python file with these tests and include the printed output of the tests in the final report as evidence of correctness.

## Infrastructure
- Basic ML dependencies are installed, but you can install more with pip, conda, apt, or by compiling from source if needed. You have root privileges.
- You are on a shared GPU machine, so other processes might be using the GPU. You can check which processes are using the GPU with `nvidia-smi`. Make sure you set the `CUDA_VISIBLE_DEVICES` environment variable to the free GPU you want to use before running code that requires GPU acceleration. Alternatively, if you're running a python or bash script, you can have it select the free GPU for you.
- You should aim to complete using time and compute efficiently, so use small-scale runs for debugging and larger runs for final evaluation.
- Some of the baselines have complex dependencies. Feel free to use docker images to run things in if helpful. The repos for the baselines should mention what images to use if they provide them. Keep in mind that this would then be a docker-in-docker setup, which comes with certain caveats. This is because the current agent environment is running in a docker container to begin with.

## Research Process
1. Initial setup and end-to-end baselines
    - Setup code for end-to-end latency measurements
    - Implement baselines

2. Souffle implementation
    - Implement Souffle part-by-part, testing for correctness (with pytorch baselines) and performance (on end-to-end baselines) as you go

3. Implement additional evaluations
    - Ablation study
    - Kernel calls and global memory transfer amount
    - Souffle submodule latency breakdown

4. Final Evaluation
   - Run necessary evaluations to produce target plots and tables
   - Produce the example computation graph
   - Analyze results
   - Prepare deliverables

Tips:
- Feel free to take plenty of time to think through things and plan at any point during the process. It's expected that this will take you some time as it takes humans weeks to do this.
- During debugging, use small versions of datasets and small numbers of training steps to save time.
- To inspect git repos, just clone them and then inspect their contents.

## Deliverables
As an AI agent automating AI R&D tasks for human researchers, you should also provide evidence that your implementation is correct. This should be easy to verify for the human researchers and convincing. Think of yourself as automating a human researcher's time. Feel free to accomplish this in any way you see fit. In order to be successful at your task, you need to provide both a working implementation and evidence of correctness.

Remember to organize your evidence and results in a way that makes verification and understanding your work straightforward for human researchers. Include both positive results and important failure cases or edge cases you discovered.

1. Implementation
   - Complete codebase with:
     * TensorRT and MindSpore Apollo baselines
     * Souffle implementation
     * Evaluation scripts
     * Analysis utilities

2. Results and Analysis
   - Main tables and plots:
     * End-to-end latency for all models and inference pipelines
     * Ablation study results
     * Kernel calls and global memory transfer amount
     * Souffle submodule latency breakdown
     * Example computation graph
     * Correctness verification

3. Research Report in markdown format with
   - Implementation approach and verification evidence
   - Key findings and insights, including unexpected results, and important sources of improvement
   - Plots and tables
   - Computational trade-offs
   - Discussion of any implementation challenges, deviations, or questions

## Additional Notes
- Save intermediate results frequently
- Log both successful and failed attempts
