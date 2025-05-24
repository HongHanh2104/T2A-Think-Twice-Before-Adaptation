# T2A: Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation

## Introduction

This repository contains the code for the paper "Think Twice before Adaptation: Improving Adaptability of DeepFake Detection via Online Test-Time Adaptation".

> Deepfake (DF) detectors face significant challenges when deployed in real-world environments, particularly when encountering test samples deviated from training data through either postprocessing manipulations or distribution shifts. We demonstrate postprocessing techniques can completely obscure generation artifacts presented in DF samples, leading to performance degradation of DF detectors. To address these challenges, we propose Think Twice before Adaptation (\texttt{T$^2$A}), a novel online test-time adaptation method that enhances the adaptability of detectors during inference without requiring access to source training data or labels. Our key idea is to enable the model to explore alternative options through an Uncertainty-aware Negative Learning objective rather than solely relying on its initial predictions as commonly seen in entropy minimization (EM)-based approaches. We also introduce an Uncertain Sample Prioritization strategy and Gradients Masking technique to improve the adaptation by focusing on important samples and model parameters. Our theoretical analysis demonstrates that the proposed negative learning objective exhibits complementary behavior to EM, facilitating better adaptation capability. Empirically, our method achieves state-of-the-art results compared to existing test-time adaptation (TTA) approaches and significantly enhances the resilience and generalization of DF detectors during inference.



## Installation

For easy installation, we recommend using the environment provided in [DeepFakeBench](https://github.com/SCLBD/DeepfakeBench).

```bash
git clone https://github.com/SCLBD/DeepfakeBench.git
cd DeepfakeBench
conda create -n DeepfakeBench python=3.7.2
conda activate DeepfakeBench
sh install.sh

```

## Usage

```python
from adapters import create_adapter
import yaml
config = yaml.load(open("configs/T2A.yaml", "r"), Loader=yaml.SafeLoader)

# Create adapter instance
adapter = create_adapter(
    adaptation_method=config["adaptation_method"],
    model=your_model,
    device=device,
    **config["config"]
)

# Adapt and get predictions
predictions = adapter.adapt_and_predict(data)
```


