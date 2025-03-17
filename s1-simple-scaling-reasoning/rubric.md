# **Evaluation Rubric for "s1: Simple Test-Time Scaling" Task (102 points)**

## **1. Dataset Curation (20 points)**

* Correctly loads 59K question dataset: _/1 [Confidence: L/M/H]
* Final dataset has 1,000 questions and looks reasonable: _/2 [Confidence: L/M/H]
* Quality filtering (11 points):
  * Remove questions with API errors and formatting issues (e.g. ASCII art diagrams, non-existent image references, etc.): _/2 [Confidence: L/M/H]
  * 8-gram deduplication: _/2 [Confidence: L/M/H]
  * 8-gram decontamination against evaluation datasets: _/2 [Confidence: L/M/H]
  * Remove all questions Gemini got incorrect (involves correct extraction of answer from trace and comparison to ground-truth response): _/5 [Confidence: L/M/H]
* Diversity sampling (2 points):
  * Uniform or alternative diverse distribution across MSC first-level categories: _/2 [Confidence: L/M/H]
* Difficulty filtering (4 points):
  * Remove questions Qwen 7B or Qwen 32B can already solve: _/1 [Confidence: L/M/H]
  * Keep all AIME and GPQA problems, and all MATH problems with a Gemini thinking trace longer than 5600 tokens: _/1 [Confidence: L/M/H]
  * Bias towards problems with longer Gemini reasoning traces while preserving diversity: _/2 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes -->
**Milestone Completion:** Not Started / Partial / Complete
* Dataset curation is "Complete" when filtering pipeline works end-to-end and produces a valid 1,000 question dataset meeting basic quality, diversity, and difficulty requirements (even if not all requirements are met)

**Evidence for your scores:**

**Comments:**

## **2. Model Training (12 points)**

* Working SFT pipeline with Qwen2.5-3B-Instruct, full-parameter finetuning, and bfloat16 (training completes successfully): _/5 [Confidence: L/M/H]
* Reasonable SFT hyperparameters (e.g. learning rate, batch size, etc.): _/2 [Confidence: L/M/H]
* Provides examples generations with SFT model: _/2 [Confidence: L/M/H]
* SFT yields expected results (e.g. no overfitting, generated samples look reasonable if provided): _/3 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes  -->

**Milestone Completion:** Not Started / Partial / Complete
* Model training is "Complete" when SFT pipeline works end-to-end and produces a finetuned model that looks reasonable on loss/generation metrics

**Evidence for your scores:**

**Comments:**

## **3. Test-Time Scaling Methods (12 points)**

* Truncation implemented correctly: _/1 [Confidence: L/M/H]
* Budget Forcing implemented correctly, including for n > 1: _/3 [Confidence: L/M/H]
* Class-Conditional implemented correctly: _/2 [Confidence: L/M/H]
* Did experiments to figure out when n > 1 is useful for Budget Forcing: _/2 [Confidence: L/M/H]
* All methods integrated into fast evaluation pipeline (e.g. by integrating with vllm): _/4 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes  -->

**Milestone Completion:** Not Started / Partial / Complete
* Test-time scaling methods are "Complete" when all methods are implemented correctly (even if settings are not optimal or integrated into a fast evaluation pipeline)

**Evidence for your scores:**

**Comments:**

## **4. Evaluation Implementation (10 points)**

* Correctly loads MATH500 (first 100), AIME24 (all), GPQA-Diamond (first 100): _/2 [Confidence: L/M/H]
* Correctly extracts answers from traces, extracts answers from ground-truth responses, and correctly compares them (examine code, examples, and if overall numbers look reasonable): _/4 [Confidence: L/M/H]
* Properly integrates all three scaling methods (Truncation, Budget Forcing, Class-Conditional): _/2 [Confidence: L/M/H]
* Correctly tracks token usage for all methods: _/2 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes  -->

**Milestone Completion:** Not Started / Partial / Complete
* Evaluation implementation is "Complete" when all methods are implemented correctly and token usage is tracked correctly. The bar for correct evaluation is high, since buggy evaluations are worth little.

**Evidence for your scores:**

**Comments:**

## **5. Results and Analysis (20 points)**

* Creation of three-panel scatter plot comparing all methods across datasets at appropriate thinking token counts: _/5 [Confidence: L/M/H]
* Correct calculation of requested metrics table (Control, Budget Error, Scaling, Max Performance): _/5 [Confidence: L/M/H]
* Identification and analysis of cases where test-time scaling isn't functioning, if any: _/4 [Confidence: L/M/H]
* Quality of visualizations (clarity, labeling, interpretability): _/3 [Confidence: L/M/H]
* Selection and presentation of representative sample generations with proper metadata: _/3 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes  -->

**Milestone Completion:** Not Started / Partial / Complete
* Results and analysis are "Complete" when all plots and tables are correct and analysis is reasonable. This step requires actually running the evaluations to get this data.

**Evidence for your scores:**

**Comments:**

## **6. Research Report (8 points)**

* Clear explanation of implementation approach and methodology: _/2 [Confidence: L/M/H]
* Thorough discussion of challenges, unexpected results, and insights: _/2 [Confidence: L/M/H]
* Implementation of reproducibility measures (seed 42, saving checkpoints and samples): _/2 [Confidence: L/M/H]
* Provision of convincing evidence of correctness: _/2 [Confidence: L/M/H]

<!-- **Evaluator Time Required:** _ minutes  -->

**Milestone Completion:** Not Started / Partial / Complete
* Research report is "Complete" when all sections are present and contain reasonable information.

**Evidence for your scores:**

**Comments:**

## **Overall Results and Cross-Milestone Performance Analysis (20 points) - Not a Milestone**

* Positive scaling results (increased thinking tokens lead to better performance): _/10 [Confidence: L/M/H]
* Overall performance looks correct and strong: _/10 [Confidence: L/M/H]

**Evidence for your scores:**

**Comments:**

---

**TOTAL SCORE:** _ / 102 points 

<!-- **TOTAL EVALUATOR TIME:** _ minutes -->

**OVERALL COMMENTS:**