# Retrieval Report

5 example queries against the arXiv cs.CL FAISS index (914 chunks from 50 papers, reference sections stripped). Distance is FAISS L2 over `all-MiniLM-L6-v2` embeddings -- lower is more similar. Regenerate with `python scripts/generate_retrieval_report.py`.

## Query 1: How can hidden states be used to detect hallucinations in large language models?

**[1] `2608.07525_12`** (distance=0.6574)

> of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632–642, 2015. [7] Yishuo Cai, Renjie Gu, Jiaxu Li, Xuancheng Huang, Junzhe Chen, Xiaotao Gu, and Minlie Huang. MHALO: Evaluating mllms as fine-grained hallucination detectors. In Findings of the Association for Computational Linguistics: ACL 2025, pages 9197–9222, 2025. [8] Jiawei Chen, Dingkang Yang, Tong Wu, Yue J...

**[2] `2608.07525_15`** (distance=0.6897)

> hallucinations in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19836–19845, 2025. [42] Yahan Tu, Rui Hu, and Jitao Sang. Ode: Open-set evaluation of hallucinations in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19836–19845, 2025. [43] Junyang Wang, Yuhang Wang,...

**[3] `2608.08024_1`** (distance=0.7800)

> At the same time, standard linear probes may be limited in how flexibly they adapt to the hallucination detection task when applied directly to frozen hidden states. This paper investigates whether hidden-state- based hallucination detection can be improved by augmenting probes with learnable prompt embed- dings (Lester et al., 2021). To this end, we in- troduce Prompt Embedding Probes, a white-bo...

## Query 2: What are the routing and load balancing strategies in Mixture-of-Experts architectures?

**[1] `2608.08650_0`** (distance=0.6088)

> LLM MoE Architecture Evolution The Evolution of Mixture-of-Experts Architectures in Large Language Models Routing, Topology, Load Balancing, and Expert Parallelism Jiguo Li1 jiguolee@gmail.com August 2026 Abstract Mixture-of-Experts models increase parameter capacity while keeping the computation activated by each token bounded, but their architectural evolution cannot be explained by a chronologi...

**[2] `2608.08650_11`** (distance=0.7775)

> Token-level dynamic computing budget budget control coupled with physical equalization Cross-layer global/local GMoE Reduce cross-layer expert redundancy Scheduling, caching and inter-layer interference need to be verified Latent/head structured Multi-Head LatentMoE Communication deterministic, decoupled from 𝑘 Insuﬀicient evidence of large-scale quality Another dimension that is often overlooked ...

**[3] `2608.08650_10`** (distance=0.8194)

> algorithm. LongCat-Flash combines ScMoE with token-dimension chunking and Single Batch Overlap; the oﬀicial LongCat-2.0 report further adopts fully parallel dense/MoE execution on each core of a dedicated accelerator[21]. These developments reduce exposed communication, but they do not automatically eliminate severe routing skew, heterogeneous expert runtimes, or GPU-memory pressure. Heterogeneous...

## Query 3: How do reasoning models allocate test-time compute across different questions?

**[1] `2608.07968_0`** (distance=0.7840)

> Thinking Hard, Not Smart: Reasoning Models Fail to Ration Test-Time Compute Across Questions Chenrui Fan*1, Yize Cheng*1, Ming Li1, Yongyuan Liang1, Tianyi Zhou2, Soheil Feizi1 1University of Maryland, College Park 2MBZUAI, UAE {cfan42, yzcheng, minglii, cheryunl, sfeizi}@umd.edu, tianyi.zhou@mbzuai.ac.ae § Project: https://github.com/Fcr09/thinking-hard-not-smart Abstract Reasoning language model...

**[2] `2608.07968_3`** (distance=0.8042)

> allocation and metareasoning. Rational metareasoning treats computation itself as a deci- sion, using its expected value to determine which reasoning operation is worth performing (Russell and Wefald, 1991; Sabbata et al., 2025). Recent work (Zhai et al., 2026) begins to allocate test-time compute across inputs using learned per-instance budget policies. ROI-Reasoning (Zhao et al., 2026) trains mo...

**[3] `2608.07968_2`** (distance=0.8138)

> rather than strategically. Models largely solve ques- tions in presentation order, spend progressively less on later questions, and respond little to stated point values. Their allocation is therefore gov- erned more by which question appears next than by which question is most worth attempting. • Budget pressure magnifies the failure. Aver- aged across all models, as the exam length N increases, ...

## Query 4: What techniques help with low-resource and multilingual machine translation?

**[1] `2608.07629_2`** (distance=0.6533)

> model trained on parallel data spanning 200 languages, with particular attention to low-resource African and Asian languages. Its SentencePiece vocabulary of 256,000 tokens provides subword coverage for all supported languages, including Unicode combining characters used in tonal orthographies. For languages outside this set, however, there is no established adaptation procedure. 2.2 Adapting to U...

**[2] `2608.07629_0`** (distance=0.8244)

> Embedding Initialization for Unseen Low-resource Languages in Multilingual NMT: A Case Study on Limbum–English Translation Samiratu Ntohsi , Neza David Tuyishimire , Anesu Kafesu , Marvin Ogore , Samuel Oluwajunwonlo Babalola , Oche Ankeli Ruzivo Research Lab, African Leadership University {sntohsi, dtuyishimire, mogore}@alueducation.com, {a.kafesu, s.babalola, o.ankeli}@alustudent.com Abstract Mu...

**[3] `2608.08283_2`** (distance=0.8816)

> research on methods for assessing the quality of these translations. Their translation capabilities are often attributed to incidental exposure to historical and bilingual materials during large-scale pretraining, as well as to models’ capacity to generalize from modern Chinese and related high-register texts (Briakou et al., 2023; Chang et al., 2023). Growing research documents LLM performance on...

## Query 5: How can gender bias be mitigated in machine translation systems?

**[1] `2608.08606_1`** (distance=0.4362)

> explicit controllability, interpretability, and compatibility with existing MT systems. 2 Related Work Gender bias in machine translation (MT) has been previously documented, par- ticularly in language pairs where the source language (e.g., English) lacks overt gender markers, while the target language (e.g., German, French, Spanish) re- quires grammatical gender agreement. Early studies [11, 16] ...

**[2] `2608.08606_0`** (distance=0.5784)

> Mitigating Gender Bias in English to Romanian Machine Translation Ioana Grigore[0009−0006−7370−8159] and Sergiu Nisioi[0000−0003−2247−4488] Human Language Technologies Research Center Faculty of Mathematics and Computer Science University of Bucharest ioanaagrigore28@gmail.com, sergiu.nisioi@unibuc.ro Abstract. Machine translation (MT) systems often fail to correctly translate gender, especially w...

**[3] `2608.08606_9`** (distance=0.5964)

> 0.846, a BLEU score of 96.92, a chrF++ score of 98.21, and a TER of 1.94. LoRA, however, underperformed in our experimental setting, suggesting that lightweight adapter tuning does not provide enough capacity for this task. A likely explanation is that this task requires more than light domain adapta- tion. The model must learn to use new source-side gender tags and realize them through correct Ro...

## Known limitation

Query 1's rank-1 and rank-2 results (`2608.07525_12`, `2608.07525_15`) are citation-list fragments, not body text -- only rank-3 is a genuine passage. `extract_text.py` only strips a References/Bibliography heading found past 30% into the document (`_MIN_REFERENCES_POSITION`), to avoid cutting real body text when an early match is a table-of-contents entry or cross-reference. In this one paper the heading legitimately sits at 26% because a large appendix follows the reference list, so the guard skips it and the reference list stays in the corpus. Affects 1/50 papers in this corpus; the other 46 papers with a detected heading were stripped correctly (see the commit that introduced `strip_references()`).
