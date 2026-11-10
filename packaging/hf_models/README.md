---
license: apache-2.0
base_model:
- huawei-bayerlab/marigold-v2-0
- Qwen/Qwen-Image-Edit-2509
pipeline_tag: depth-estimation
tags:
- marigold
- depth-estimation
---

# Vathography App — model files

The depth model files the Vathography App downloads on its first run: the [Marigold V2](https://huggingface.co/huawei-bayerlab/marigold-v2-0) depth model (`depth/Log-stage2`) in ComfyUI's file layout. They are byte-identical to the files in [Comfy-Org/marigold-v2-0](https://huggingface.co/Comfy-Org/marigold-v2-0) (revision `c304e4370a1bc677a98c906c3683ace49a6bdadd`), kept here so that the app's download stays the same.

The app takes its backbone and its see-through model from [Vathography/marigold-v2-see-through-comfyui](https://huggingface.co/Vathography/marigold-v2-see-through-comfyui). The app itself: https://vathography.com

## Files

| File | What it is | sha256 |
|---|---|---|
| `marigold_v2_depth_log_stage2.safetensors` | the depth model (LoRA) | `ec1f5649eef5ddb487b4de2222f6b62837f094c02d652843bd4fceb481037ca0` |
| `marigold_v2_depth_log_stage2_vae.safetensors` | its VAE (fine-tuned decoder, Qwen-Image-Edit-2509 encoder) | `0c1a95f7410678a93b59ba42b3a069f50b54ff9f6e35525699eb8fcd63ce332c` |
| `marigold_v2_depth_conditioning.safetensors` | its conditioning | `9f5c00a2df993136c541cdbc37bcfb81ab1ba81adaec25bfb9cbfff81edd7c9c` |

The same files work in ComfyUI (`models/loras/`, `models/vae/`, `models/embeddings/`).

## Credits and license

**Marigold V2** — Igor Pavlovic, Thiemo Wandel, Anton Obukhov, Luca Bartolomei, Andrey Davydov, Fabio Tosi, Matteo Poggi, Sabine Süsstrunk, Dengxin Dai (EPFL, HUAWEI Bayer Lab, University of Bologna). [Paper](https://arxiv.org/abs/2609.08084) · [code](https://github.com/huawei-bayerlab/marigold-v2) · [weights](https://huggingface.co/huawei-bayerlab/marigold-v2-0).
**Qwen-Image-Edit-2509** — the Qwen team. [Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) · [report](https://arxiv.org/abs/2508.02324).
**ComfyUI** file layout and these files — Comfy-Org.

Apache License 2.0 (`LICENSE`), as the original works. Free to use, including commercially. Not affiliated with or endorsed by the authors above. Details in `NOTICE`.

```bibtex
@article{pavlovic2026marigoldv2,
    author = {Pavlovic, Igor and Wandel, Thiemo and Obukhov, Anton and Bartolomei, Luca and Davydov, Andrey and Tosi, Fabio and Poggi, Matteo and S{\"u}sstrunk, Sabine and Dai, Dengxin},
    title = {Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation},
    journal = {ACM Trans. Graph.}, volume = {45}, number = {6}, articleno = {204}, year = {2026},
    doi = {10.1145/3842528}
}
@misc{wu2025qwenimagetechnicalreport,
    title = {Qwen-Image Technical Report},
    author = {Chenfei Wu and others},
    year = {2025}, eprint = {2508.02324}, archivePrefix = {arXiv}
}
```
