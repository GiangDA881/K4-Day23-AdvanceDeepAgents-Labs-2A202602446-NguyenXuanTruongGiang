# A Survey of Video and Multimodal Generation: Foundations, Architectures, Benchmarks, and Recent Trends

## TL;DR
- Video generation has transitioned from cascaded 2D/3D U-Net diffusion models to scalable Diffusion Transformer (DiT) architectures capable of generating high-resolution, multi-second sequences up to 13B parameters [1][2][3][4].
- Multimodal conditioning techniques—including cascaded spatial encoders, plug-and-play motion modules, and training-free attention control—enable precise image-to-video (I2V) and structure-guided video editing [5][6][7][8][9].
- Comprehensive benchmark suites like VBench and automated metrics like VideoScore have standardized multi-dimensional evaluation across temporal consistency, motion dynamics, and human alignment [10][11].
- Recent open-source models (2023–2025) such as Open-Sora 2.0, CogVideoX, and HunyuanVideo 1.5 incorporate causal 3D VAEs, rectified flow, and spatiotemporal sparsity (SSTA) to achieve production-grade efficiency and fidelity [12][13][3][14].

## Background
Video and multimodal generation aims to synthesize temporally coherent, visually realistic moving imagery conditioned on text, images, audio, or spatial constraints. Historically dominated by recurrent networks and generative adversarial networks (GANs) restricted to short clips and low resolutions, the field experienced a major paradigm shift with the success of diffusion models and autoregressive transformers [15][16][1]. Video generation poses unique challenges compared to image synthesis due to the immense dimensionality of spatial-temporal data, where maintaining long-term physical realism, object permanence, and temporal consistency requires modeling complex motion dynamics across frames [17][18]. The emergence of large-scale curated video datasets and high-capacity transformer architectures has transformed video generators from constrained domain models into general-purpose world simulators [17][5][19].

## Foundations and Text-to-Video Architectures
Early foundational approaches adapted text-to-image (T2I) models for video by introducing temporal layers or factorizing spatial-temporal operations. Make-A-Video decomposed spatial-temporal tensors through pseudo-3D convolutions and temporal attention layers trained on unsupervised video data without requiring paired text-video datasets [16]. Similarly, Imagen Video employed a cascade of 7 video diffusion models totaling 11.6B parameters, integrating spatial and temporal super-resolution models with v-prediction parameterization [1]. Autoregressive formulations like VideoPoet treated video generation as a zero-shot multimodal sequence modeling task using a decoder-only transformer processing text, images, videos, and audio [15].

With the scaling success of Diffusion Transformers (DiTs), the field shifted away from U-Net backbones toward transformer blocks operating on compressed latent spaces. OpenAI's Sora introduced spacetime patch representations combined with DiTs, enabling the joint processing of variable resolutions, aspect ratios, and durations up to one minute [17]. Open-Sora and Open-Sora 2.0 advanced open-source democratization by pairing Spatial-Temporal Diffusion Transformers (STDiT) with highly compressive 3D VAEs and hybrid dual-stream/single-stream blocks inspired by FLUX [12][2][13]. These architectures utilize 3D Rotary Position Embedding (RoPE) to capture positional relationships across time, height, and width coordinates, providing robust foundations for scalable video synthesis [12][4].

## Multimodal Conditioning, Image-to-Video, and Control
Controlling video generation requires integrating diverse modalities such as static images, sparse spatial cues, and audio. Image-to-video (I2V) frameworks bridge static perception and dynamic generation; for instance, Stable Video Diffusion (SVD) scales latent video diffusion models using a three-stage training protocol and supports explicit motion control via camera motion-specific LoRA modules [5]. I2VGen-XL employs a cascaded architecture featuring two hierarchical encoders (a fixed CLIP encoder and a learnable content encoder) at a base stage ($448 \times 256$) followed by a refinement stage to achieve high-resolution $1280 \times 720$ output [6].

For personalized motion dynamics and structural guidance, plug-and-play frameworks like AnimateDiff inject lightweight motion modules into frozen T2I models, supporting MotionLoRA for specific camera movements without full model retraining [7]. Training-free control pipelines such as ControlVideo and ControlNet adapt spatial conditioning signals (e.g., Canny edges, depth maps, human poses) to video by introducing fully cross-frame interaction in self-attention modules and interleaved-frame smoothers to eliminate flickering [8][9]. Furthermore, DiTCtrl explores attention control in multi-modal diffusion transformers to enable multi-prompt, longer video generation with smooth transitions [20].

## Benchmarks, Evaluation, and Datasets
Evaluating video generation quality requires metrics that capture both individual frame appearance and temporal continuity. Fréchet Video Distance (FVD) extended FID by replacing frame-level networks with an Inflated 3D Convnet (I3D) pre-trained on Kinetics-400, proving superior correlation with human judgment regarding temporal distortions [18]. Traditional action recognition and video understanding datasets such as UCF101 (13.3K videos), ActivityNet (20K videos), and Kinetics (over 300K short videos) have served foundational roles in pre-training and feature extraction [19].

To address the multi-dimensional nature of generative video, VBench established a comprehensive benchmark suite decomposing quality into 16 fine-grained dimensions across video quality and video-condition consistency, showing high correlation with human perception [10]. VBench++ further expanded this framework to support adaptive image suites and multi-task evaluation for both text-to-video and image-to-video settings [21]. Complementing static test suites, automatic metrics like VideoScore simulate fine-grained human feedback by training on the large-scale VIDEOFEEDBACK dataset (37.6K videos), achieving high Spearman correlation and pairwise comparison accuracy across GenAI-Bench and VBench [11].

## Trends and open problems
Recent advances (2023–2025) have focused on scaling parameter capacity, improving inference efficiency, and unifying multimodal tasks in open-source ecosystems. Models such as CogVideoX (up to 5B parameters) incorporate 3D causal VAEs and Expert Adaptive LayerNorm (AdaLN) to eliminate future leakage and temporal flicker in 10-second continuous videos [3]. HunyuanVideo scaled open-source parameter counts to 13B using a Causal 3D VAE and unified full attention, outperforming proprietary models in professional human evaluations of motion dynamics [4]. Its successor, HunyuanVideo 1.5, introduced Selective and Sliding Tile Attention (SSTA) to achieve a 1.87x inference speedup by dynamically pruning redundant spatiotemporal tokens [14]. Lightricks' LTX-Video supports synchronized audio-video generation at native resolutions up to 4K with distilled and quantized variants [22].

Despite these breakthroughs, significant open problems remain. The quadratic computational complexity of 3D full attention in video DiTs consumes over 80% of computational resources, prompting training-free acceleration frameworks like Sparse VideoGen (SVG) and RainFusion [23]. Physical realism, long video coherence, mitigation of visual artifacts, and precise semantic alignment without hallucination continue to be active areas of research as the community pushes toward real-time, interactive world simulators.

## References
[1] IMAGEN VIDEO: HIGH DEFINITION VIDEO GENERATION WITH DIFFUSION MODELS. web. https://imagen.research.google/video/paper.pdf (n.d.)
[2] Open-Sora: Democratizing Efficient Video Production for All. web. https://arxiv.html/2412.20404 (2024-12-29)
[3] Text-to-Video Diffusion Models with An Expert Transformer. arxiv. https://arxiv.org/abs/2408.06072 (2025-03-26)
[4] HunyuanVideo: A Systematic Framework For Large Video Generative Models. web. https://arxiv.org/abs/2412.03603 (2024-12-03)
[5] Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. web. https://arxiv.org/abs/2311.15127 (2023-11-25)
[6] I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models. web. https://ar5iv.labs.arxiv.org/abs/2311.04145 (2023-11-09)
[7] AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. web. https://arxiv.org/abs/2307.04725 (2023-07-11)
[8] ControlVideo: Training-free Controllable Text-to-Video Generation. hf-search. https://huggingface.co/papers/2305.13077 (2023-05-22)
[9] ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models. web. https://huggingface.co/lllyasviel/ControlNet (n.d.)
[10] VBench: Comprehensive Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2311.17982 (2023-11-29)
[11] VIDEOSCORE: Building Automatic Metrics to Simulate Fine-grained Human Feedback. web. https://aclanthology.org/2024.emnlp-main.127.pdf (n.d.)
[12] Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k. web. https://arxiv.org/abs/2503.09642 (2025-03-23)
[13] Open-Sora: Democratizing Efficient Video Production for All. web. https://arxiv.org/html/2412.20404 (n.d.)
[14] HunyuanVideo 1.5 Technical Report. web. https://arxiv.org/abs/2511.18870 (2025-11-24)
[15] VideoPoet: A Large Language Model for Zero-Shot Video Generation. arxiv. https://arxiv.org/abs/2312.14125 (2023-12-21)
[16] Make-A-Video: Text-to-Video Generation without Text-Video Data. web. https://arxiv.org/abs/2209.14792 (n.d.)
[17] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[18] Towards Accurate Generative Models of Video: A New Metric & Challenges. arxiv. https://arxiv.org/abs/1812.01717 (2019-03-27)
[19] Rethinking Temporal Information for Video Continual Learning. web. https://openaccess.thecvf.com/content/CVPR2023W/CLVision/papers/Alssum_Just_a_Glimpse_Rethinking_Temporal_Information_for_Video_Continual_Learning_CVPRW_2023_paper.pdf (n.d.)
[20] DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation. hf-search. https://huggingface.co/papers/2412.18597 (2024-12-24)
[21] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2411.13503 (2024-11-20)
[22] Lightricks/LTX-Video. web. https://github.com/lightricks/ltx-Video (2024-11-20)
[23] Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity. arxiv. https://arxiv.org/abs/2502.01776 (2025-02-03)
