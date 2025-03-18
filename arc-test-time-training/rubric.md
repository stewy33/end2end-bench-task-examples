# **Evaluation Rubric for "Test-Time Training for Abstract Reasoning Corpus (ARC): Exploration and Evaluation" Task (104 points)**

## **1. Data Generation for Test-Time Training (15 points)**

* Correctly downloads and loads the ARC dataset: _/2 [Confidence: L/M/H]
* LOO, permutation, and data augmentation (9 points)
    * Implements leave-one-out + permutation data generation operators to expand the size of the training set per-problem. If had n training examples, then this should result in at least n * 2 training examples: _/4 [Confidence: L/M/H]
    * Implements data augmentation operators with reasonable operator choices and parameters: _/3 [Confidence: L/M/H]
    * Data augmentation expands the size of the training set per-problem by a factor of >=10: _/1 [Confidence: L/M/H]
    * Creates a final method that allows for control over whether data augmentation is applied in addition to LOO+permutation, and applies this correctly to the ARC dataset: _/1 [Confidence: L/M/H]
* Formatting (3 points)
    * Format the ARC grids as strings in a reasonable manner: _/1 [Confidence: L/M/H]
    * Format the ARC examples as in-context learning examples following the format in the instructions: _/2 [Confidence: L/M/H]
* Data generation pipeline runs successfully without crashing (need not save the data to disk, it can be constructed online): _/1 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes -->
**Milestone Completion:** Not Started / Partial / Complete
* Data generation is "Complete" when data generation pipeline works end-to-end and correctly implements some form of LOO+permutation+data augmentation that expands the size of the training set by >=10 on average

**Evidence for your scores:**

**Comments:**

## **2. Test-Time Training Pipeline and Loss Ablations (14 points)**

* Uses the correct 1B finetuned model for TTT: _/1 [Confidence: L/M/H]
* Correctly implements loss mask to handle demonstration loss (computes outputs on intermediate outputs as well as final output): _/3 [Confidence: L/M/H]
* Also implements no demonstration loss version: _/1 [Confidence: L/M/H]
* Trains a single LoRA adapter per-problem, with reasonable hyperparameters, and stores them appropriately so they can be loaded later on the right problem: _/4 [Confidence: L/M/H]
* Trains at least 100 adapters for 100 problems without crashing or timing out: _/4 [Confidence: L/M/H]
* Trains both with and without demonstration loss: _/1 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes -->
**Milestone Completion:** Not Started / Partial / Complete
* Test-time training pipeline is "Complete" when the pipeline works end-to-end without crashing, trains one adapter per problem using the data generation procedure implemented previously, implements the correct loss functions and reasonable string formatting, and results in some kind of loss reduction or performance improvement

**Evidence for your scores:**

**Comments:**

## **3. Inference and Evaluation Pipeline (13 points)**

* Evaluates on a 100 problem subset of ARC that matches what adapters were trained on: _/1 [Confidence: L/M/H]
* Uses pass@2 sampling for results: _/1 [Confidence: L/M/H]
* Correctly loads and uses the right TTT adapter for inference per problem: _/2 [Confidence: L/M/H]
* Correctly evaluates correctness of generated outputs against ground-truth: _/4 [Confidence: L/M/H]
* Evaluation pipeline runs successfully without crashing and saves results to disk: _/4 [Confidence: L/M/H]
* Evaluation pipeline uses a fast inference library (e.g. vllm, not standard transformers): _/1 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes -->
**Milestone Completion:** Not Started / Partial / Complete
* Inference and evaluation pipeline is "Complete" when the pipeline can correctly load adapters, run inference, and evaluate results against ground truth.

**Evidence for your scores:**

**Comments:**

## **4. Sampling and Aggregation Ablations (20 points)**

* Runs evaluation with both greedy and one non-greedy decoding strategy: _/2 [Confidence: L/M/H]
* Correctly implements aggregation scope variations (no augmentation, single type, multiple types): _/4 [Confidence: L/M/H]
* Correctly implements both flat and some kind of hierarchical aggregation strategy across augmentation types: _/3 [Confidence: L/M/H]
* Correctly implements oracle comparison: _/2 [Confidence: L/M/H]
* Includes permutated versions of input in the aggregation: _/1 [Confidence: L/M/H]
* Chooses the best TTT configuration to run sampling and aggregation ablations on: _/1 [Confidence: L/M/H]
* Find a good joint TTT + sampling + aggregation configuration, and runs ablations on this configuration instead of poorer performing ones: _/3 [Confidence: L/M/H]
* Actually runs all of the above ablations: _/4 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes -->
**Milestone Completion:** Not Started / Partial / Complete
* Sampling and aggregation ablations are "Complete" when all specified ablation strategies are implemented and evaluated on the best TTT configuration

**Evidence for your scores:**

**Comments:**

## **5. Results and Analysis (10 points)**

* Empirical results are loaded correctly: _/2 [Confidence: L/M/H]
* Creates bar plot comparing base vs FT vs TTT performance with pass@2: _/2 [Confidence: L/M/H]
* Creates bar plot showing effect of test-time data/optimization ablations (no data augmentation, no demonstration loss, full TTT): _/2 [Confidence: L/M/H]
* Creates bar plot comparing sampling/aggregation strategies: _/2 [Confidence: L/M/H]
* Plots provide a clear picture of which components matter and which decisions improve performance: _/2 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes -->
**Milestone Completion:** Not Started / Partial / Complete
* Results and analysis is "Complete" when all required plots are created

**Evidence for your scores:**

**Comments:**

## **6. Research Report (14 points)**

* Clear explanation of implementation approach and methodology: _/4 [Confidence: L/M/H]
* Thorough discussion of challenges, unexpected results, and insights: _/2 [Confidence: L/M/H]
* Discussion of implications of ablation results: _/2 [Confidence: L/M/H]
* Implementation of reproducibility measures (seed 42, saving checkpoints and generations): _/2 [Confidence: L/M/H]
* Provision of convincing evidence of correctness: _/4 [Confidence: L/M/H]

**Milestone Completion:** Not Started / Partial / Complete
* Research report is "Complete" when all these sections are present and contain reasonable information.

**Evidence for your scores:**

**Comments:**

## **Overall Results and Cross-Milestone Performance Analysis (18 points) - Not a Milestone**

* Strong positive results from TTT over the baselines: _/10 [Confidence: L/M/H]
* Overall performance (including baselines, and relative performance across ablations) seems reasonable: _/10 [Confidence: L/M/H]

**Evidence for your scores:**

**Comments:**

---

**TOTAL SCORE:** _ / 100 points 

<!-- **TOTAL EVALUATOR TIME:** _ minutes -->

**OVERALL COMMENTS:**