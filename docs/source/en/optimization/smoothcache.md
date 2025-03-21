# SmoothCache

[SmoothCache]([https://github.com/Roblox/SmoothCache](https://huggingface.co/papers/2411.10510)) is a training-free acceleration technique designed to enhance the performance of Diffusion Transformer (DiT) pipelines. 
It reduces the need for computationally expensive operations via caching schemes based on layer-wise error analysis. 
SmoothCache applies to different models and modalities. 
It also offers plug-and-play integration with [Diffusers DiT Pipeline]([https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/dit/pipeline_dit.py)), 



## Quick Start

### Install
```bash
pip install dit-smoothcache
```

### Usage - Inference

Inspired by [DeepCache](https://raw.githubusercontent.com/horseee/DeepCache), implemented zero-intrusion helper class to control the caching behavior of the pipeline.

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

### 256x256 Image Generation Task

![Mosaic Image](https://github.com/Roblox/SmoothCache/blob/main/assets/dit-mosaic.png)


## Benchmark

### Image Generation with DiT-XL/2-256x256 Using DDIM Sampling

*Note: L2C is not training free.*

| Schedule       | Steps | FID (↓)     | sFID (↓)    | IS (↑)       | TMACs   | Latency (s) |
|----------------|-------|------------:|------------:|------------:|--------:|------------:|
| **L2C**        | 50    | 2.27 ± 0.04 | 4.23 ± 0.02 | 245.8 ± 0.7 | 278.71  | 6.85        |
| No Cache       | 50    | 2.50 ± 0.04 | 4.35 ± 0.03 | 244.6 ± 0.9 | 282.57  | 8.08        |
| FORA (r=2)     | 50    | 2.47 ± 0.04 | 4.29 ± 0.03 | 246.1 ± 1.0 | 281.52  | 7.88        |
| Ours (a=0.08)  | 50    | 2.31 ± 0.03 | 4.25 ± 0.02 | 246.3 ± 1.1 | 281.69  | 7.75        |
|                |       |             |             |             |         |             |
| No Cache       | 70    | 2.44 ± 0.04 | 4.39 ± 0.03 | 243.7 ± 1.2 | 392.65  | 10.84       |
| FORA (r=2)     | 70    | 2.43 ± 0.05 | 4.36 ± 0.03 | 243.9 ± 1.1 | 391.57  | 10.47       |
| Ours (a=0.08)  | 70    | 2.34 ± 0.05 | 4.28 ± 0.03 | 245.0 ± 1.2 | 391.89  | 10.24       |
|                |       |             |             |             |         |             |
| No Cache       | 30    | 2.70 ± 0.04 | 4.47 ± 0.02 | 242.6 ± 0.7 | 169.53  | 5.12        |
| FORA (r=2)     | 30    | 2.64 ± 0.04 | 4.42 ± 0.03 | 243.2 ± 0.8 | 168.81  | 4.91        |
| Ours (a=0.08)  | 30    | 2.51 ± 0.04 | 4.38 ± 0.02 | 244.4 ± 1.0 | 168.95  | 4.83        |

