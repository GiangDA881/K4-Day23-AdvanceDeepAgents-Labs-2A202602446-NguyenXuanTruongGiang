# A Survey of LLM Agents and Tool Use: Foundations, Reasoning, Benchmarks, and Security Trends

## TL;DR
- Large language model (LLM) agents leverage external tools and APIs through frameworks like ReAct, Toolformer, and Gorilla, shifting static text generation into interactive problem-solving [1][2][3].
- Advanced reasoning frameworks such as Plan-and-Solve, Reflexion, and depth-first search decision trees (DFSDT) enable agents to handle multi-step tasks across thousands of APIs [4][5][6].
- Comprehensive benchmarks like WebArena, ToolBench, and AgentBench highlight significant performance gaps between frontier LLMs and human capabilities in real-world environments [7][8][9].
- Recent work underscores critical multi-agent security vulnerabilities—including indirect prompt injection, cascading tool attacks, and the security-collaboration trade-off [10][11][12].

## Background
Large language model (LLM) agents combine foundational language generation capabilities with external tools, memory modules, and execution environments to solve complex multi-step tasks. Traditionally, LLMs operated in a closed-loop parametric regime, relying strictly on internal weights to generate answers, which often led to hallucinations and factual obsolescence. The integration of tool use allows models to query search engines, invoke calculators, execute code interpreters, and interact with professional APIs. Early milestones in this domain established how models can be prompted or fine-tuned to interleave reasoning with tool execution, paving the way for autonomous agents capable of dynamic problem-solving [1][2][3].

## Foundations and Core Tool-Use Architectures
The foundational design of LLM agents rests on prompt-based paradigms and specialized fine-tuning strategies that teach models when and how to invoke external utilities. A pivotal advance is the ReAct framework, which prompts LLMs to generate verbal reasoning traces (thoughts) alongside task-specific actions in an interleaved manner [1]. By expanding the action space to include a language thought space, ReAct enables models to create, maintain, and adjust high-level plans while reducing error propagation in multi-hop question answering [1]. 

Complementing prompt-based architectures, self-supervised and retriever-aware fine-tuning approaches address tool selection accuracy and API hallucinations. Toolformer demonstrates that language models can be trained to autonomously decide which APIs to call, what arguments to pass, and how to incorporate results into future token prediction using a self-supervised loss [2]. To address syntax and parameter errors when interfacing with massive API collections, Gorilla fine-tunes LLaMA-based models using Retriever-Aware Training (RAT), allowing models to adapt to dynamic API documentation updates and outperforming zero-shot GPT-4 in API functionality accuracy [3].

## Reasoning, Planning, and Execution Frameworks
Beyond single-step API calls, complex workflows require structured planning, error handling, and multi-tool coordination. Plan-and-Solve (PS) prompting addresses zero-shot reasoning by prompting models to decompose tasks into smaller subtasks before execution, while PS+ introduces detailed instructions on variable extraction and intermediate calculation to mitigate math errors [5]. To handle trial-and-error execution in dynamic environments, Reflexion introduces verbal reinforcement learning, enabling agents to reflect on task feedback and store linguistic memories in an episodic memory buffer for subsequent attempts [4].

For complex software and AI ecosystems, frameworks scale tool integration to thousands of endpoints. ToolLLM introduces an open-source benchmark and training framework covering over 16,000 real-world RESTful APIs, utilizing a depth-first search-based decision tree (DFSDT) algorithm to evaluate multiple reasoning traces and prevent dead ends [6]. Similarly, HuggingGPT employs ChatGPT as a central controller to orchestrate expert AI models hosted on Hugging Face using language as a generic interface, executing multi-stage task planning, model selection, and dependency resolution [13].

## Benchmarks, Datasets, and Evaluation Metrics
Evaluating LLM agents requires rigorous, reproducible environments that test multi-turn interactions, API correctness, and end-to-end task success. API-Bank provides a runnable evaluation system with 73 real-world API tools across 1,000 domains, revealing that while GPT-4 achieves high success in basic API calls, its performance drops when integrating planning, retrieval, and execution [14]. To measure realistic digital workflows, WebArena builds fully operational web applications across e-commerce and collaborative domains, showing that the best GPT-4 agent achieves an end-to-end task success rate of 14.41% compared to 78.24% for humans [7].

Other notable benchmarks target specific dimensions of agent capability. ToolBench uses an efficient machine evaluator (ToolEval) achieving high human agreement (87.1% pass rate) to measure multi-tool execution [8]. ToolQA assesses question-answering over external reference corpora, showing that standard agents struggle with complex multi-hop tool composition [15]. AgentBench unifies evaluation across 8 diverse environments—including operating systems, databases, and games—exposing persistent performance gaps in multi-turn interactive settings [9].

## Trends, Security Risks, and Open Problems
Recent developments over the past two years highlight a transition toward multi-agent collaboration ecosystems alongside escalating security vulnerabilities. Multi-agent systems introduce complex threat vectors, such as indirect prompt injection via tool outputs, inter-agent message poisoning, and cascading orchestrator manipulation [11]. Studies on agent security benchmarks and multi-agent systems reveal that defensive mechanisms often incur a "security-collaboration trade-off," where active memory vaccines and prompt guards reduce attack success rates but can inadvertently diminish benign collaboration efficiency [10][12]. 

Simultaneously, operational bottlenecks such as execution latency and token bloat present major engineering challenges. Trajectory analyses on SWE-bench show that accumulated token usage can exceed one million tokens per task due to redundant tool outputs and verbose build logs [16]. To mitigate these costs, client-side optimization frameworks utilize sample-efficient search strategies and execution pipelines to navigate the Pareto frontier between task accuracy, latency, and monetary expense [17].

## References
[1] Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[3] Gorilla: Large Language Model Connected with Massive APIs. web. https://arxiv.org/abs/2305.15334 (2023-05-25)
[4] Reflexion: Language Agents with Verbal Reinforcement Learning. arxiv. https://arxiv.org/abs/2303.11366 (2023-03-20)
[5] Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. arxiv. https://arxiv.org/abs/2305.04091 (2023-05-26)
[6] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arxiv. https://arxiv.org/abs/2307.16789 (2023-07-31)
[7] WebArena: A Realistic Web Environment for Building Autonomous Agents. web. https://proceedings.iclr.cc/paper_files/paper/2024/file/4410c0711e9154a7a2d26f9b3816d1ef-Paper-Conference.pdf (n.d.)
[8] ToolBench: An Open-Source Benchmark for Evaluating Tool-Using Agents. web. https://github.com/openbmb/toolbench (n.d.)
[9] AgentBench: A Comprehensive Benchmark to Evaluate LLM-as-Agent. web. https://github.com/THUDM/AgentBench (n.d.)
[10] Multi-Agent Security Tax: Trading Off Security and Collaboration Capabilities in Multi-Agent Systems. hf-search. https://huggingface.co/papers/2502.19145 (2025-02-27)
[11] Beyond Single-Model Injection: A Threat Model and Defense Architecture for Prompt Injection in Multi-Agent Systems. web. https://arxiv.org/abs/2609.22949 (2026-09-19)
[12] ACIARENA: Toward Unified Evaluation for Agent Cascading Injection. web. https://aclanthology.org/2026.acl-long.457.pdf (n.d.)
[13] HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. web. https://papers.neurips.cc/paper_files/paper/2023/file/77c33e6a367922d003ff102ffb92b658-Paper-Conference.pdf (2023-11-18)
[14] API-bank: A comprehensive benchmark for tool-augmented LLMs. web. https://arxiv.org/abs/2304.08244 (2023-10-25)
[15] ToolQA: A Dataset for LLM Question Answering with External Tools. web. https://proceedings.neurips.cc/paper_files/paper/2023/file/9cb2a7495900f8b602cb10159246a016-Paper-Datasets_and_Benchmarks.pdf (n.d.)
[16] Reducing Cost of LLM Agents with Trajectory Reduction. web. https://dl.acm.org/doi/pdf/10.1145/3797084 (n.d.)
[17] AgentOpt v0.1 Technical Report: Client-Side Optimization for LLM-Based Agent. web. https://arxiv.org/abs/2604.06296 (n.d.)
