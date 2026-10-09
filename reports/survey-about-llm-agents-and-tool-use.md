# Survey on Large Language Model Agents and Tool Use: Architectures, Benchmarks, and Open-World Challenges

## TL;DR
- Large language model agents combine internal reasoning traces with external tool invocation through paradigms like ReAct and modular neuro-symbolic routers such as MRKL [1][2].
- Efficient API utilization and reasoning have advanced via self-supervised tool annotation (Toolformer), dedicated API fine-tuning (Gorilla), and decoupled execution blueprints (ReWOO) [3][4][5].
- Tool learning and multi-agent frameworks (API-Bank, WebArena, MetaGPT) highlight performance gaps in multi-step workflows, where end-to-end task success drops significantly compared to single-step accuracy [6][7][8].
- Robustness evaluations reveal vulnerabilities to benchmark exploitation, state tampering, and indirect security attacks, driving recent shifts toward test-time compute scaling and end-to-end reinforcement learning [9][10][11].

## Background
Large language model (LLM) agents represent autonomous systems where a foundational language model acts as the core controller, perceiving environments, maintaining memory, planning multi-step trajectories, and invoking external tools such as calculators, web search engines, databases, and application programming interfaces (APIs). Traditionally, standard LLMs suffer from static pre-training cutoffs, hallucination of factual details, and an inability to perform exact mathematical computations or interact dynamically with real-time software. Bridging this gap has become central to artificial intelligence research, transforming static text generators into active problem-solvers capable of executing software workflows and retrieving external knowledge on demand [1][3][2].

Foundational architectural paradigms such as ReAct established the standard of interleaving verbal reasoning traces and task-specific actions to dynamically create, maintain, and adjust high-level plans [1]. Concurrently, MRKL systems proposed modular, neuro-symbolic architectures connecting language models to specialized symbolic and neural modules via a router [2]. These foundational frameworks addressed key limitations of early models, enabling systems to decide when to invoke external tools, what arguments to pass, and how to incorporate execution observations into subsequent reasoning steps [1][3][2].

## Architectural Paradigms and Tool Integration
Building upon basic interleaving frameworks, subsequent research focused on optimizing how LLMs acquire, select, and interact with tool libraries. Toolformer demonstrated that language models can be trained in a self-supervised manner to decide which APIs to call, when to call them, and how to incorporate results by filtering API calls based on whether they reduce future token perplexity [3]. To address API selection hallucinations and massive library scaling, Gorilla fine-tuned a 7B LLaMA model on APIBench using Abstract Syntax Tree (AST) sub-tree matching, surpassing GPT-4 in functional accuracy by 20.43% on zero-shot API invocation tasks [4].

To mitigate intermediate token redundancy and propagation errors inherent in continuous interleaving, ReWOO introduced a decoupled paradigm featuring a Planner, Worker, and Solver [5]. By generating an interdependent plan blueprint upfront, ReWOO achieved a 5x token efficiency improvement and a 64% reduction in token usage on public reasoning benchmarks [5]. Furthermore, the transition toward modular agent architectures has been supported by standardized interoperability frameworks like the Model Context Protocol (MCP), which formalizes secure skill acquisition and progressive disclosure without requiring full model retraining [12][13].

## Tool Learning, Environment Interaction, and Multi-Agent Collaboration
Evaluating agents in realistic interactive settings requires specialized benchmarks and environments. API-Bank established a comprehensive runnable evaluation suite of 73 tools and 314 dialogues, revealing that while GPT-3.5 achieves high accuracy on basic API calls, its effectiveness drops by 21% in retrieve-and-call settings and a further 17% in plan-retrieve-and-call settings [6]. Similarly, ToolQA and WebArena provided long-horizon interactive environments—such as e-commerce platforms and discussion forums—where GPT-4 achieved an end-to-end task success rate of 14.41%, compared to 78.24% for humans [14][7].

To tackle complex multi-step workflows, multi-agent collaborative frameworks have emerged. MetaGPT enforces Standard Operating Procedures (SOPs) with structured documents and diagrams, achieving 85.9% Pass@1 on HumanEval and reducing human revision costs compared to unstructured chat frameworks [8]. AutoGen provides generic conversation interfaces with automated reply mechanisms and Docker-based container isolation, achieving 69.48% accuracy on complex level-5 mathematical problem solving [15]. Moreover, automated orchestration systems like AOrchestra dynamically instantiate specialized sub-agents to manage complex task distributions [16].

## Benchmarks, Robustness, and Evaluation Vulnerabilities
As agent capabilities expand, rigorous evaluation has revealed severe vulnerabilities and diagnostic challenges in existing testbeds. Automated auditing of prominent benchmarks—including SWE-bench, WebArena, OSWorld, and GAIA—revealed that nearly all can be exploited for near-perfect scores without actually solving tasks, such as by overwriting test hooks in container environments or loading public validation answers directly from local files [9]. In SWE-bench Verified, differential patch testing (PatchDiff) demonstrated that 7.8% of plausible model-generated patches fail developer-written tests outside modified files, inflating reported resolution rates by 6.4 absolute percent points [17].

Specialized diagnostic benchmarks further dissect agent failure modes. ToolBH identified that proprietary models like Gemini-1.5-Pro and GPT-4o achieve low multi-level hallucination diagnostic scores (45.3 and 37.0 out of 100), primarily due to errors in solvability detection and instrumental reasoning [18]. CostBench demonstrated that dynamic blocking events and cost changes drastically degrade agent performance, with exact match ratios falling significantly under strict cost-constraint scenarios [19]. Furthermore, safety benchmarks like SafeToolBench and security audits on Model Context Protocol implementations (MCP Security Bench) underscore persistent risks in privacy leaks, physical injury, and indirect prompt injection during tool execution [20][21].

## Trends and open problems
The past two years have witnessed a major paradigm shift toward reinforcement learning for end-to-end agent training and test-time compute scaling. Methods such as Agent-R1, CM2, and L-Zero (L0) employ reinforcement learning with verifiable rewards (RLVR), checklist rewards, and REPL code-as-action execution paradigms to train small language models (such as Qwen2.5-7B-Instruct) to surpass previous zero-shot baselines [22][10][23]. Simultaneously, scaling test-time compute through parallel sampling, sequential revision, and verifiers has demonstrated that increasing diversified rollouts and targeted reflection substantially improves agent task performance [11].

Despite these breakthroughs, significant open problems remain unsolved. Safety alignment for tool-using agents continues to struggle against complex agentic threats, including tool-initiated attacks and multi-stage prompt injections, prompting defenses like MCP-Guard [24][25]. Furthermore, long-horizon reliability, execution latency, and robustness against dynamic environment failures remain key barriers to deploying autonomous agents in open-world production settings [26][27].

## References
[1] Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning. arxiv. https://arxiv.org/abs/2205.00445 (2022-05-02)
[3] Toolformer: Language Models Can Teach Themselves To Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[4] Gorilla: Large Language Model Connected with Massive APIs. arxiv. https://arxiv.org/abs/2305.15334 (2023-05-25)
[5] ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models. arxiv. https://arxiv.org/abs/2305.18323 (2023-05-23)
[6] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. hf-search. https://huggingface.co/papers/2311.11944 (2023-11-20)
[7] WebArena: A Realistic Web Environment for Building Autonomous Agents. web. https://ar5iv.labs.arxiv.org/html/2307.13854 (2023-07-23)
[8] MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework. web. https://arxiv.org/html/2308.00352v6 (2024-10-21)
[9] How We Broke Top AI Agent Benchmarks. web. https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/ (n.d.)
[10] Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning. hf-search. https://huggingface.co/papers/2511.14460 (2025-11-18)
[11] Scaling Test-time Compute for LLM Agents. web. https://paperswithcode.co/paper/2506.12928 (n.d.)
[12] Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward. arxiv. https://arxiv.org/abs/2602.12430 (2026-02-12)
[13] A survey of agent interoperability protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP). hf-search. https://huggingface.co/papers/2505.02279 (2025-05-04)
[14] ToolQA: A Dataset for LLM Question Answering with External Tools. web. https://arxiv.org/html/2306.13304v1 (2023-06-23)
[15] AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. web. https://arxiv.org/html/2308.08155v2 (2023-10-03)
[16] AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration. hf-daily. https://huggingface.co/papers/2602.03786 (2026-02-03)
[17] Are “Solved Issues” in SWE-bench Really Solved Correctly? An Empirical Study. web. https://dl.acm.org/doi/abs/10.1145/3744916.3764576 (2026-09-11)
[18] A Multi-level Hallucination Diagnostic Benchmark for Tool-augmented Large Language Models. web. https://aclanthology.org/2024.emnlp-main.637.pdf (n.d.)
[19] CostBench: Evaluating Multi-Turn Cost-Optimal Planning and Adaptation in Dynamic Environments for LLM Tool-Use Agents. web. https://aclanthology.org/2026.acl-long.584.pdf (n.d.)
[20] SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool Utilization Safety in LLMs. web. https://aclanthology.org/2025.findings-emnlp.958.pdf (n.d.)
[21] MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents. hf-search. https://huggingface.co/papers/2510.15994 (2025-10-14)
[22] CM2: Reinforcement Learning with Checklist Rewards for Multi-Turn and Multi-Step Agentic Tool Use. hf-search. https://huggingface.co/papers/2602.12268 (2026-02-12)
[23] L0: Reinforcement Learning to Become General Agents. web. https://arxiv.org/html/2506.23667v1 (n.d.)
[24] Agent Safety Alignment via Reinforcement Learning. arxiv. https://arxiv.org/abs/2507.08270 (2025-07-11)
[25] MCP-Guard: A Multi-Stage Defense-in-Depth Framework for Securing Model Context Protocol in Agentic AI. hf-search. https://huggingface.co/papers/2508.10991 (2026-01-08)
[26] AGENCYBENCH: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-Word Contexts. web. https://aclanthology.org/2026.acl-long.337.pdf (n.d.)
[27] ToolGym: an Open-world Tool-using Environment for Scalable Agent Testing and Data Curation. hf-search. https://huggingface.co/papers/2601.06328 (2026-01-09)
