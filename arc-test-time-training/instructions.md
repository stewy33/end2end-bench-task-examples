# Test-Time Training for Abstract Reasoning Corpus (ARC): Exploration and Evaluation

<p align="center">
  <img src="https://raw.githubusercontent.com/stewy33/acc_rd_media/refs/heads/main/acc_rd_ttt/arc_example.png" alt="ARC dataset example" width="800px">
</p>

## Overview
Your task is to explore and evaluate test-time training (TTT) approaches for improving abstract reasoning capabilities on the ARC benchmark. You will work with a [finetuned Llama-3.2-1B-Instruct model](https://huggingface.co/stewy33/acc_rd_ttt-finetuned-1B-model) as the fine-tuned model, and explore various algorithmic choices to maximize performance while providing clear ablation studies of different design decisions.

The [ARC benchmark](https://github.com/fchollet/ARC-AGI.git) consists of tasks where you're given input-output pairs of colored 2D grids containing shapes and patterns (colors represented as numbers in the dataset). For each task, the LLM will receive 2-7 training examples and need to predict the output for a test input by understanding the underlying transformation rule.

Test-time training is a technique where given a new task with a few training examples, we can train the model (using LoRA) to improve performance on this new task. Test-time training can improve performance (and has previously been successful on ARC), but the design space for TTT approaches is large, and there is a limited understanding of which design choices are most effective for LLMs (and novel-task learning).

## Target Results
### 1. Final Performance Comparison
   - Compare three configurations: a base model ([Llama-3.2-Instruct-1B](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)), fine-tuned (FT) model (provided Llama model that was fine-tuned on a dataset of ARC problems), and FT+TTT (w/ your TTT implementation)
   - Metric: Accuracy on ARC validation set (fixed subset of first 100 samples in the eval dataset for all models) with pass@2 metric (you are allowed two submissions per problem and are successful if you get at least one correct). Use pass@2 metric for all plots in this task.
   - Present as a bar plot showing the progression of improvements with these three configurations

### 2. Algorithmic Design Space Exploration
Your goal is to explore and justify choices in two key areas:

1. Test-Time Data and Optimization
   - Compare different approaches to generating and using data during test-time training:
  
   A. Data generation
     - Leave-one-out + permutation: Create training tasks by leaving out one example at a time from the given training examples. For a task with n examples, this creates n training datapoints to train the TTT LoRA adapter. Then increase the number of training examples by permuting the order of the in-context examples and adding these new examples to the training set.
     - Leave-one-out + permutation + data augmentations: Create training tasks by leaving out one example at a time from the given training examples. Then apply all data augmentations to create a training set of n * (m + 1) datapoints to train the TTT LoRA adapter, where m is the number of augmentations applied. Use augmentations like
       * Rotations
       * Reflections
       * Shifts (shift grid by up to a certain number of cells)
       * Color permutations (randomly permute the colors/numbers used)
       * Increase resolution (e.g. doubling)
       * Increase height or width of the grid (e.g. scale in one dimension but not other)
       * Potentially combinations of these
      Finally, increase the number of training examples by permuting the order of the in-context examples for leave-one-out and augmented examples and adding these new examples to the training set.
      <p align="center">
         <img src="https://raw.githubusercontent.com/stewy33/acc_rd_media/refs/heads/main/acc_rd_ttt/ttt_data.png" alt="visual example of leave-one-out + permutation + data augmentations" width="600px">
      </p>

   B. Demonstration loss options
     - With demonstration loss: During test-time training, compute loss on both:
       * The final output prediction given all demonstrations
       * Intermediate predictions where we predict each demonstration's output given previous demonstrations
     - Without demonstration loss: Only compute loss on final output prediction
     
Present findings in a single bar plot showing impact of each choice. The ablations should be relative to the best performing configuration (i.e. have a bar for the best configuration and then have bars for best configuration but with one choice made differently). In all cases, you should use LoRA for the TTT implementation (one adapter per task).

Notes on TTT dataset:
- Test-time training format:
   - In-context learning: Present examples as a sequence with context

      Input 1: [train_input_grid1]

      Output 1: [train_output_grid1]

      Input 2: [train_input_grid2]

      Output 2: [train_output_grid2]

      Input 3: [train_input_grid3]

      Output 3: [train_output_grid3]

      Test input: [test_input_grid]

      Output:

B. Test-Time Sampling and Aggregation 
- Explore strategies for combining predictions from different versions of the inputs (use the same format as used in part A to train the TTT model you're using, e.g. if you use the in-context learning format, then use the same format at evaluation time). Here are some axes to explore:

  1. LLM sampling strategy
     - Greedy vs non-greedy decoding
  
  2. Aggregation scope
     - No data augmentation: Only aggregate predictions from original input + any permutations of few-shot train order
     - Single type: Only aggregate predictions from one type of augmentation (e.g., only rotations)
     - Multiple types: Combine predictions across all augmentation types
  
  3. Hierarchical aggregation
     - Two-stage voting process of some kind
     - Compare to flat voting across all predictions
  
  4. Oracle comparison
     - Include an "oracle" that picks the correct answer if it exists in any sample
     - Shows upper bound on what's achievable with better aggregation

Present results in a bar plot comparing different voting/aggregation/sampling strategies. Try these sampling/aggregation strategies on the best-performing TTT configuration. Feel free to combine them if this results in better performance.

Notes on TTT evaluation:
- Part of the ARC-AGI challenge is to perform well within certain hardware constraints and a time limit. Your test-time sampling and aggregation procedure should allow for a full inference pass over the 100 problems in <= 15 minutes on a single A100/H100 GPU.

## Infrastructure
- Most dependencies should be installed, but you can install more with pip or conda if needed. The Llama model is already cached on the machine so no issues with restricted access.
- You are on a shared GPU machine, so other processes might be using the GPU. You can check which processes are using the GPU with `nvidia-smi`. Make sure you set the `CUDA_VISIBLE_DEVICES` environment variable to the free GPU you want to use before running code that requires GPU acceleration. Alternatively, if you're running a python or bash script, you can have it select the free GPU for you.
- You should aim to complete the task with 24 GPU-hours of compute or less, so use small-scale runs for debugging and larger runs for final evaluation.

## Research Process
1. Initial Setup
   - Get base model running on ARC and get results (use vllm for fast inference)
   - Implement data processing pipeline
   - Setup code for test-time training

2. Systematic Exploration
   - For each design space:
     a. Start with simplest version
     b. Document what works/doesn't
   - Be mindful of runtime and compute usage
   - Save all experimental results for later analysis
   - You don't need to try every possible combination of the design space. Try one at a time and do hill-climbing to find the best configuration. Then do ablations on this final configuration.

3. Final Evaluation
   - Combine best components
   - Run ablation studies
   - Generate plots
   - Analyze results

Tips:
- Feel free to take plenty of time to think through things and plan at any point during the process. It's expected that this will take you some time as it takes humans weeks to do this.
- During debugging, use small versions of datasets and small numbers of training steps to save time.
- To inspect git repos, just clone them and then inspect their contents.

## Deliverables
As an AI agent automating AI R&D tasks for human researchers, you should also provide evidence that your implementation is correct. This should be easy to verify for the human researchers and convincing. Think of yourself as automating a human researcher's time. Feel free to accomplish this in any way you see fit. In order to be successful at your task, you need to provide both a working implementation and evidence of correctness.

Remember to organize your evidence and results in a way that makes verification and understanding your work straightforward for human researchers. Include both positive results and important failure cases or edge cases you discovered.

1. Implementation
   - Data processing and augmentation code
   - Training and evaluation pipelines
   - Aggregation strategies
   - Plotting utilities

2. Results and Analysis
   - Four main bar plots:
     a. Base vs FT vs TTT performance
     b. Test-time data/optimization ablations
     c. Sampling/aggregation strategy comparisons

3. Research Report in markdown format detailing
   - Implementation approach and verification evidence
   - Which components matter most
   - Unexpected results
   - Computational trade-offs
   - Interesting observations and insights
   - Discussion of any implementation challenges, deviations, or questions

## Additional Notes
- Use seed 42 everywhere for reproducibility
- Save intermediate results frequently
- Log both successful and failed attempts
- Consider time/compute trade-offs in exploration