# World Models in AI and Machine Learning: A Comprehensive Survey

## TL;DR
- World models serve as internal environment simulators and predictive models in reinforcement learning and robotics, tracing their roots to early recurrent neural network formulations [1] and modern architectures like Dreamer and MuZero [2][3].
- Latent dynamics approaches, such as the Dreamer family and Recurrent State-Space Models (RSSMs), enable data-efficient policy optimization and planning directly from images and complex observations [2][4][5].
- Multimodal and video foundation world models (e.g., Sora, Genie, V-JEPA 2) scale generative transformers and joint-embedding predictive architectures over large internet and robotic datasets to capture spatial-temporal consistency [6][7][8][9].
- Recent agentic and benchmark developments (2023-2026) highlight crucial challenges such as compounding roll-out errors, hallucination, and the perception-functionality gap, requiring dedicated testing protocols and modular self-improving frameworks [10][11][12][13].

## Background
World models are internal predictive representations of past and present states and futures, inspired by cognitive systems where biological perception is governed by predictions of future sensory data given motor actions [1]. Formally, a world model learns a transition function $p(s_{t+1} | s_t, a_t)$ over environment states $s_t$ and actions $a_t$, allowing an agent to simulate and evaluate multiple hypothetical trajectories without interacting directly with the physical environment. Early conceptualizations trace back to early recurrent neural network architectures and differentiable environment models from the early 1990s [1]. 

The modern renaissance of world models began with the integration of variational autoencoders (VAEs) and Mixture Density Network Recurrent Neural Networks (MDN-RNNs), demonstrating that agents could learn compact spatio-temporal representations and evolve effective policies inside their own internal "dream environments" before deployment in tasks such as CarRacing-v0 [14]. These frameworks address core computational bottlenecks in artificial intelligence: sample inefficiency in reinforcement learning, the high cost of real-world physical exploration, and the need for robust planning under uncertainty.

## Latent Dynamics and Model-Based Reinforcement Learning
Model-based reinforcement learning relies heavily on learning environment dynamics in compact latent spaces rather than high-dimensional pixel spaces. The Deep Planning Network (PlaNet) introduced a purely model-based agent that learns environment dynamics from images using the Recurrent State-Space Model (RSSM), which combines deterministic and stochastic transition components essential for long-term latent prediction [5]. By performing fast online planning in this latent space, PlaNet solved continuous control tasks from pixels using up to 200x fewer episodes than strong model-free algorithms [5].

Building upon latent dynamics, the Dreamer family revolutionized visual reinforcement learning. DreamerV1 introduced latent imagination for learning behaviors directly from images [2]. DreamerV2 leveraged categorical/discrete representations in latent space, achieving human-level performance on Atari benchmarks [15]. DreamerV3 further advanced generalization as a general algorithm that outperformed specialized methods across over 150 diverse tasks using a single fixed hyperparameter set, notably collecting diamonds in Minecraft from scratch without human data within 100M environment steps [4]. Alternatively, MuZero integrated learned recurrent dynamics directly into Monte Carlo Tree Search (MCTS), predicting future rewards, policies, and values end-to-end to achieve superhuman performance in board games (Go, chess, shogi) and state-of-the-art results across Atari 2600 games without knowing the rules [3]. More recent variations, such as the Residual-Action World Model (ResWM), reformulate control as incremental residual adjustments rather than absolute actions to stabilize long-horizon planning and improve sample efficiency [16].

## Visual, Multimodal, and Foundation World Models
In parallel with reinforcement learning, generative video models and joint-embedding predictive architectures have emerged as large-scale foundation world models. Sora pioneered joint training on variable-duration videos and images using a transformer architecture operating on spacetime patches of latent codes, exhibiting emergent 3D consistency, long-range object permanence, and realistic physical persistence under occlusions [6]. Sora 2 extended these capabilities with synchronized audio, dialogue, and physically accurate modeling of buoyancy, rigidity, and elastic collisions [17].

Interactive and action-conditioned world models further bridge perception and control. Genie is an 11B parameter foundation world model that generates endless action-controllable virtual environments from unlabelled internet videos without ground-truth action labels [7]. Genie 3 advanced this paradigm to real-time interactive generation at 24 frames per second at 720p resolution, incorporating promptable world events and extended visual memory [9]. In robotics and physical understanding, architectures like V-JEPA 2 employ self-supervised joint-embedding predictive training over millions of hours of video and robotic interaction datasets, predicting representations in latent space rather than reconstructing raw pixels to achieve 65% to 80% success rates on zero-shot robotic pick-and-place tasks [8]. However, physical evaluation studies indicate that current video generation models often rely on case-based generalization—mimicking closest training exemplars—rather than abstracting true physical rules, displaying a clear attribute prioritization hierarchy where color and size take precedence over shape and velocity [18].

## Trends, Benchmarks, and Open Problems
The evaluation and deployment of world models between 2023 and 2026 revealed significant challenges, notably the perception-functionality gap: perceptual quality and visual realism are only weakly correlated with functional utility for decision-making [10]. To address this, benchmarks such as HalluWorld and WorldMark were established to evaluate hallucination, temporal coherence, physical consistency, and causal reasoning [10][11]. Research shows that hallucination in world models is predictable and preventable through structured constraint mechanisms [19], while transformer-based models can maintain strong performance on interactions like Atari 100k [20].

Recent research has also expanded into agentic world modeling and recursive self-improvement frameworks. Studies on agentic world modeling establish foundational taxonomies, scaling laws, and synthetic environment generation for agentic reinforcement learning [12][21]. Advanced architectures like Long-WAM scale the context window of world-action models [22], while Memento 3 explores model-based recursive self-improvement through reflective rulebooks in persistent memory [13]. Despite these advances, compounding errors during long-horizon rollouts, environment drift, and out-of-distribution generalization remain central open problems [10].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Dream to Control: Learning Behaviors by Latent Imagination. hf-search. https://huggingface.co/papers/1912.01603 (2019-12-03)
[3] Mastering Atari, Go, chess and shogi by planning with a learned model. hf-search. https://huggingface.co/papers/1911.08265 (2019-11-19)
[4] Mastering Diverse Domains through World Models. web. https://arxiv.org/abs/2301.04104 (n.d.)
[5] Learning Latent Dynamics for Planning from Pixels. hf-search. https://huggingface.co/papers/1811.04551 (2019-06-04)
[6] Sora: Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[7] Genie: Generative Interactive Environments. hf-search. https://huggingface.co/papers/2402.15391 (2024-02-23)
[8] V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. web. https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/ (2025-06-11)
[9] Genie 3: A new frontier for world models. web. https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ (2025-08-05)
[10] State of World Models 2026: Taxonomy, Benchmarks and Open Challenges. web. https://world-models.io/reports/state-of-world-models-2026/state-of-world-models-2026-v1.0.pdf (n.d.)
[11] HalluWorld: A Controlled Benchmark for Hallucination via Reference World Models. web. https://arxiv.org/html/2605.19341v1 (n.d.)
[12] Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond. hf-search. https://huggingface.co/papers/2604.22748 (2026-04-24)
[13] Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks. hf-daily. https://huggingface.co/papers/2610.11794 (2026-10-08)
[14] Recurrent World Models Facilitate Policy Evolution. arxiv. https://arxiv.org/abs/1809.01999 (2018-09-04)
[15] Mastering Atari with Discrete World Models. hf-search. https://huggingface.co/papers/2010.02193 (2022-02-12)
[16] ResWM: Residual-Action World Model for Visual RL. hf-search. https://huggingface.co/papers/2603.11110 (2026-03-11)
[17] Sora 2 is here. web. https://openai.com/index/sora-2/ (2025-09-30)
[18] How Far Is Video Generation from World Model: A Physical Simulation Perspective. web. https://arxiv.org/html/2411.02385v1 (2024-11-04)
[19] Hallucination in World Models is Predictable and Preventable. web. https://arxiv.org/html/2606.27326 (2026-06-25)
[20] Transformer-based World Models Are Happy With 100k Interactions. web. https://ar5iv.labs.arxiv.org/html/2303.07109 (n.d.)
[21] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning. hf-search. https://huggingface.co/papers/2602.10090 (2026-02-10)
[22] Long-WAM: Scaling the Context of World-Action Models. hf-daily. https://huggingface.co/papers/2610.10528 (2026-10-07)
