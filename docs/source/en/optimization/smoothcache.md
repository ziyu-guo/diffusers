# SmoothCache

[SmoothCache](https://huggingface.co/papers/2411.10510) is a training-free acceleration technique designed to enhance the performance of Diffusion Transformer (DiT) pipelines. 
It reduces the need for computationally expensive operations via caching schemes based on layer-wise error analysis. 
SmoothCache offers fine-grained control over individual components. For instance, you can selectively enable caching for cross-attention or self-attention blocks based on 
their specific error distribution. This makes SmoothCache flexible across different pipelines and modalities. 
It also provides seamless, plug-and-play integration with [Diffusers DiT Pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/dit/pipeline_dit.py).


## Quick Start

### Install
```bash
pip install dit-smoothcache
```

### Usage - Inference

We implemented zero-intrusion helper class to control the caching behavior of the pipeline.

Generally, only a few additional lines needs to be added to the sampler scripts to enable/disable SmoothCache:

#### Inference usage with HF Diffuser DiTPipeline:
```python
import json
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

# Import SmoothCacheHelper
from SmoothCache import DiffuserCacheHelper  

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Initialize the DiffuserCacheHelper with the model
with open("smoothcache_schedules/50-N-3-threshold-0.35.json", "r") as f:
    schedule = json.load(f)
cache_helper = DiffuserCacheHelper(pipe.transformer, schedule=schedule)

# Enable the caching helper
cache_helper.enable()

words = ["Labrador retriever"]
class_ids = pipe.get_label_ids(words)
generator = torch.manual_seed(33)
image = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator).images[0]

# Restore the original forward method and disable the helper
# disable() should be paired up with enable() 
cache_helper.disable()
```

We have provided a few sample cache schedules in the SmoothCache repo. Below is an example on how 
to generate your own schedule for your specific task.

#### Caching schedule generation for HF Diffuser DiTPipeline:

```python
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from SmoothCache import DiffuserCalibrationHelper

def main():
    # Load pipeline
    pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    schedule_file="smoothcache_schedules/diffuser_schedule.json"
    num_inference_steps = 50

    # Initialize calibration helper
    calibration_helper = DiffuserCalibrationHelper(
        model=pipe.transformer,
        calibration_lookahead=3,
        calibration_threshold=0.15,
        schedule_length=num_inference_steps, 
        log_file=schedule_file
    )

    # Enable calibration
    calibration_helper.enable()

    # Run pipeline normally
    words = ["Labrador retriever", "combination lock", "cassette player"]

    
    class_ids = pipe.get_label_ids(words)

    generator = torch.manual_seed(33)
    images = pipe(
        class_labels=class_ids,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images  # Normal pipeline call

    # Disable calibration and generate schedule
    calibration_helper.disable()
```

The `calibration_lookahead` parameter specifies how many future diffusion steps the calibration process examines to identify stable regions for caching, 
while the `calibration_threshold` sets the maximum allowed error between layer representations to qualify a caching opportunity. 
Together, they balance performance gain from caching with the output quality.

## Visualization

![Mosaic Image](https://github.com/Roblox/SmoothCache/raw/main/assets/TeaserFigureFlat.png)
**Accelerating Diffusion Transformer inference across multiple modalities with 50 DDIM Steps on DiT-XL-256x256, 100 DPM-Solver++(3M) SDE steps for a 10s audio sample (spectrogram shown) on Stable Audio Open, 30 Rectified Flow steps on Open-Sora 480p 2s videos**


## Benchmark

### Image Generation with DiT-XL/2-256x256 Using DDIM Sampling

*Note: L2C is not training free.*

| Schedule       | Steps | FID (↓)     | sFID (↓)    | IS (↑)       | TMACs   | Latency (s) |
|----------------|-------|------------:|------------:|------------:|--------:|------------:|
| **L2C**        | 50    | 2.27 ± 0.04 | 4.23 ± 0.02 | 245.8 ± 0.7 | 278.71  | 6.85        |
|                |       |             |             |             |         |             |
| No Cache       | 50    | 2.28 ± 0.03 | 4.30 ± 0.02 | 241.6 ± 1.1 | 365.59  | 8.34        |
| FORA (n=2)     | 50    | 2.65 ± 0.04 | 4.69 ± 0.03 | 238.5 ± 1.1 | 190.26  | 5.17        |
| FORA (n=3)     | 50    | 3.31 ± 0.05 | 5.71 ± 0.06 | 230.1 ± 1.3 | 131.81  | 4.12        |
| Ours (a=0.08)  | 50    | 2.28 ± 0.03 | 4.29 ± 0.02 | 241.8 ± 1.1 | 336.37  | 7.62        |
| Ours (a=0.18)  | 50    | 2.65 ± 0.04 | 4.65 ± 0.03 | 238.7 ± 1.1 | 175.65  | 4.85        |
| Ours (a=0.22)  | 50    | 3.14 ± 0.05 | 5.19 ± 0.04 | 231.7 ± 1.0 | 131.81  | 4.11        |
|                |       |             |             |             |         |             |
| No Cache       | 30    | 2.66 ± 0.04 | 4.42 ± 0.03 | 234.6 ± 1.0 | 219.36  | 4.88       |
| FORA (r=2)     | 30    | 3.79 ± 0.04 | 5.72 ± 0.05 | 222.2 ± 1.2 | 117.08  | 3.13       |
| Ours (a=0.08)  | 30    | 3.72 ± 0.04 | 5.51 ± 0.05 | 222.9 ± 1.0 | 117.08  | 3.13       |
|                |       |             |             |             |         |             |
| No Cache       | 70    | 2.17 ± 0.02 | 4.33 ± 0.02 | 242.3 ± 1.6 | 511.83  | 11.47        |
| FORA (r=2)     | 70    | 2.36 ± 0.02 | 4.46 ± 0.03 | 242.2 ± 1.3 | 263.43  | 7.15        |
| Ours (a=0.08)  | 70    | 2.37 ± 0.02 | 4.29 ± 0.03 | 242.6 ± 1.5 | 248.8  | 6.9        |

