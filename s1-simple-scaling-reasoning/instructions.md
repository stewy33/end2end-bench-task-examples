# s1: Simple test-time scaling by distillation

## Overview
Your task is to develop a simple and cost-effective approach to achieve test-time scaling and strong reasoning performance from a standard language model. To do this, you will use a small dataset of 1,000 difficult questions paired with reasoning traces (from Gemini Thinking Experimental) to distill chain-of-thought reasoning ability into a [Qwen2.5-3B-Instruction model](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct). In contrast to models like OpenAI's o1, which use many samples of reinforcement learning training to achieve emergent chain-of-thought reasoning, you will take a distillation-focused approach, distilling this ability from a RL-trained reasoning model into a standard LLM at much lower cost.

After training, you will combine this model with several test-time scaling approaches to control the number of tokens it generates before generating a final answer. Your goal is to systematically evaluate how different test-time computation strategies affect performance on difficult reasoning tasks -- and to find a strong approach in this distillation setting.

## Target Results
### 1. Test-Time Scaling Plots

   Compare different approaches for controlling computation at test-time and their scaling performance:

   1. Truncation
      - Enforce a maximum number of thinking tokens by appending the end-of-thinking token delimiter and "Final Answer:" to early exit the thinking stage.
      - Truncate at [500, 1000, 2000, 4000, 8000, infinite] thinking tokens

   2. Budget Forcing
      - Enforce a maximum token count by appending the end-of-thinking token delimiter and "Final Answer:" to early exit the thinking stage
      - If the model terminates early, replace/suppress the end-of-thinking token up to n times and append "Wait", or other continuation words/phrases to the model's current reasoning trace to encourage the model to reflect on its current generation. Experiment with a couple choices of n. Note that it only makes sense to use higher n for higher/infinite truncation values (otherwise the truncation cap will happen before the end-of-thinking token would need to be suppressed). So do some observation to see when you actually need n > 1 and only do that for those cases.
      - Truncate at [500, 1000, 2000, 4000, 8000, infinite] thinking tokens
      - Play around with a couple values of n
   
   3. Class-Conditional Control (tell the model in the prompt how long it should think for)
      - Try both "short thinking" or "long thinking" prompts and present results with both
      - Do not truncate the reasoning trace (this treats two prompts as two coarse settings for controlling the number of thinking tokens)
   
   Create a three-panel scatter plot comparing the performance of the different test-time scaling methods on the first 100 problems of MATH500, the full AIME24 dataset, and the first 100 problems of GPQA-Diamond. The x-axis should be the average number of thinking tokens used by the model for each method. The y-axis should be the accuracy of the model for each method.

   If you observe any places where test-time scaling isn't happening, please prepare plots/tables showing these cases in a separate section. And provide a brief discussion of why you think this is happening.

### 2. Quantitative Scaling Metrics Table

   Take your results from part 1 and prepare a table measuring 'Control', 'Budget Error', 'Scaling', and 'Max Performance' for each method and dataset. Here are what these metrics mean:
   - Control (only applies to truncation and budget forcing, use 50% for class-conditional): the percent of generations whose number of thinking tokens exceeds the truncation budget + 10 slack tokens (in the cases when a truncation budget is used).
   - Budget Error: for log-spaced counts of [500, 1000, 2000, 4000, and 8000], calculate average absolute value of the percent difference between the average number of thinking tokens used by the method and the log-spaced counts.
   - Scaling: Take all pairs of runs within each method and calculate the average slope of accuracy wrt avg thinking tokens used. In other words,
   $$
      \sum_{tokens(b) > tokens(a)} \frac{acc(b) - acc(a)}{tokens(b) - tokens(a)}
   $$
   - Max Performance: the maximum accuracy achieved by each method for each dataset

### 3. Example Samples
   - Show a few representative samples across methods and datasets in the final report
   - Include paths to the full evaluation samples as well

## Data Sources & Evaluation

### Distillation Dataset Curation
- You will begin with a [large initial collection of 59K questions](https://huggingface.co/datasets/stewy33/acc_rd_s1-full_59k_dataset). Some information about this dataset
  - Is a collection of (prompt, reasoning, answer_attempt, ground_truth_solution) tuples with reasoning traces from Gemini Flash Thinking 2.0, a reasoning model from Google.
  - Contains questions from many sources: NuminaMATH, MATH, AIME 1983-2024, OlympicArena, OmniMath, AGIEval, GPQA, Stanford Statistics PhD qualifying exams, etc.
  - Already filtered out questions with basic quality issues (e.g. non-English samples, samples with images, etc.).
  - Includes feature labels on whether a couple Qwen models got the question correct and the MSC first-level category/domain of the question.

- Now go through a rigorous filtering process to arrive at final training dataset of 1,000 questions. This process will filter for Quality, Difficulty, and Diversity.
  1. Quality filter:
    * Remove questions where API errors happened
    * Remove questions that Gemini got wrong
    * Remove questions with formatting issues like ASCII art diagrams, non-existent image references, etc.
    * Decontaminate distillation dataset against the evaluation questions described further below (use 8-grams to check for contamination). Also, you should deduplicate the distillation dataset using the same 8-gram procedure.

  2. Diversity bias: Each question is classified using the first level of the [Mathematics Subject Classification (MSC) scheme](https://en.wikipedia.org/wiki/Mathematics_Subject_Classification). Devise a filtering/sampling scheme the makes the training set roughly uniform across the MSC first-level categories.

  3. Difficulty filter and bias:
    * Remove all questions that the Qwen variants can already solve (Qwen 7B and Qwen 32B), as a proxy for difficulty.
    * Keep all AIME and GQPA problems, and all MATH problems with a Gemini thinking trace longer than 5600 tokens. For all other domains, sample problems uniformly across domains but bias towards problems with longer Gemini reasoning traces. Make sure there are no duplicate problems and no problems that show up in the eval set.

- Make sure your final 1,000 problems simultaneously satisfy the quality, diversity, and difficulty desiderata as specified above.
- You will do simple SFT on this data with full-parameter finetuning (no LoRA) with Qwen2.5-3B-Instruct in bfloat16 precision.

### Evaluation Datasets
1. [MATH500](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark) test split (use first 100/500 problems)
2. [AIME24](https://huggingface.co/datasets/simplescaling/aime24_nofigures) (use all problems)
3. [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) (use first 100 problems)

## Infrastructure
- Basic ML dependencies are installed, but you can install more with pip or conda if needed.
- You are on a shared GPU machine, so other processes might be using the GPU. You can check which processes are using the GPU with `nvidia-smi`. Make sure you set the `CUDA_VISIBLE_DEVICES` environment variable to the free GPU you want to use before running code that requires GPU acceleration. Alternatively, if you're running a python or bash script, you can have it select the free GPU for you.
- You should aim to complete using time and compute efficiently, so use small-scale runs for debugging and larger runs for final evaluation.

## Research Process
1. Initial setup and dataset filtering
   - Understand the initial dataset that will be filtered. Inspect its fields and look at some examples
   - Build data filtering and sampling pipeline (use vllm for fast local inference when applicable)

2. SFT training
   - Setup code for SFT training
   - Train the model

3. Evaluation setup and test-time scaling methods
   - Setup code for evaluation (use vllm for fast inference). It will be key to get answer checking working correctly, since different datasets have different answer formats you might need to deal with. This may require some inspection of the evaluation datasets to understand their answer formats.
   - Integrate test-time scaling methods into vllm evaluation pipeline.

4. Final Evaluation
   - Run necessary evaluations to produce target plots and tables
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
     * Data curation pipeline
     * SFT training pipeline
     * All test-time scaling methods
     * Evaluation scripts
     * Analysis utilities

2. Results and Analysis
   - Main plots:
     * Accuracy vs computation for each dataset
     * Method comparison plots
   - Reasoning traces:
     * A few diverse examples
     * Include metadata (dataset, correct/incorrect, computation used)

3. Research Report in markdown Format with
   - Implementation approach and verification evidence
   - Key findings and insights, including unexpected results
   - Plots and tables as well as some example generations
   - Computational trade-offs
   - Discussion of any implementation challenges, deviations, or questions

## Additional Notes
- Use seed 42 EVERYWHERE for reproducibility
- Save intermediate results frequently
- Log both successful and failed attempts
- Consider time/compute trade-offs in exploration/design choices
