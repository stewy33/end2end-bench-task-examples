# End2End-Bench Task Examples

We provide instructions, evaluation rubrics, and gold-standard replication reports for several End2EndBench tasks in this repository.

We also provide some sample agent trajectories on the s1: Simple test-time scaling task at [https://www.stewyslocum.com/end2end-bench-task-examples/agent_trajectories](https://www.stewyslocum.com/end2end-bench-task-examples/agent_trajectories).

### Tasks:
* **The Surprising Effectiveness of Test-Time Training for Abstract Reasoning**
    * [Original Paper](https://arxiv.org/abs/2411.07279): Shows that test-time training significantly improves LLMs' performance on the ARC-AGI benchmark. Involves investigating a large space of design choices in how to apply test-time training to this setting effectively. Requires dataset generation, test-time training implementation, hill-climbing on a space of design choices, and evaluation of resulting performance.
        * Unpublished, but well-received online, a good deal of citations in a short time, and a popular github codebase.
    * [End2End-Bench Task Instructions, Gold-Standard Replication Report, and Evaluation Rubric](arc-test-time-training/)

* **s1: Simple test-time scaling**
    * [Original Paper](https://arxiv.org/abs/2501.19393): Tries to find a simple and minimal approach to distilling strong test-time CoT scaling behavior to improve reasoning in open-source models. This paper spends a lot of time on dataset construction to build a tiny dataset of only 1000 examples that significantly boosts performance. This task involves data curation, model training, experimentation with various test-time scaling and sampling methods, and evaluation of performance on math domains.
        * Unpublished, but well-received online, tons of citations in a short time, and thousands of stars on github.
    * [End2End-Bench Task Instructions, Gold-Standard Replication Report, and Evaluation Rubric](s1-simple-test-time-scaling/)

* **Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions**
    * [Original Paper](https://dl.acm.org/doi/10.1145/3617232.3624858): Develops a compiler, called Souffle, to speed up deep learning inference by intelligently fusing and parallelizing tensor expressions into CUDA kernels, even across operator boundaries. This task involves graph analysis, compiler design, a deep understanding of pipelining and inefficiencies in existing inference code, CUDA programming, and performance evaluation.
        * Published at ACM ASPLOS 2024, a top systems conference
    * [End2End-Bench Task Instructions, Gold-Standard Replication Report, and Evaluation Rubric](souffle-optimize-inference-speed/)