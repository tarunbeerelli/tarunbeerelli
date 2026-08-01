I work at the intersection of optimization, learning, and intelligent systems. My projects span Riemannian meta-learning with second-order gradients and custom Triton kernels, multi-turn RL against verifiable execution feedback (GRPO), and quantized GPU inference serving at sub-millisecond P50 — research depth to deployed systems. Interested in world models and the mathematics of how systems learn to reason about structure.

---

### Current

**RLEF-Code** — `RL` `GRPO` `LLM post-training`
Multi-turn GRPO teaching Qwen2.5-Coder-7B to repair its own code against a live execution sandbox; attention hosts error repair, and capacity does not buy it.
`PyTorch` `vLLM` `PEFT/LoRA` `bf16` `1×H200` `W&B`

**RMAML** — `MAML` `Riemannian optimization` `GPU kernels`
Few-shot meta-learning constrained to the Stiefel manifold, with a custom Cayley retraction via the Woodbury identity — 13.4× over `geoopt`.
`PyTorch` `Triton` `geoopt` `Ray Tune` `MLflow`

**Vision Serving Platform** — `MLOps` `inference optimization` `Kubernetes`
ViT-tiny and CLIP served through NVIDIA Triton on GKE, with an ONNX→TensorRT INT8/FP16 quantization pipeline at 0.78ms P50 and 1,266 RPS.
`TensorRT` `ONNX` `gRPC` `GKE` `Terraform` `Prometheus` `Locust`

### Earlier

**M5 Forecasting** — `ML` `time-series` `gradient boosting`
28-day demand forecasting across 3,049 Walmart SKUs; per-state LightGBM over a 60M-row feature matrix. Top 4% on Kaggle.
`LightGBM` `scikit-learn` `pandas` `Flask` `GCP`

**Email Smart Compose** — `DL` `NLP` `seq2seq`
Real-time sentence completion — Bi-GRU encoder–decoder with a custom beam search decoder over an 85K-sentence Enron corpus.
`TensorFlow/Keras` `GloVe` `spaCy` `Flask` `GCP`

---

`Python` `C++` `MATLAB` `PyTorch` `TensorFlow` `Triton` `CUDA` `Docker` `Terraform` `GCP`
