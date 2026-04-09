<h1 align="center">M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation</h1>
<h4 align="center" style="line-height:1.4; margin-top:0.6rem">
CVPR 2026 Highlight
</h4>



<h4 align="center" style="line-height:1.4; margin-top:0.6rem">
  <a href="https://github.com/Graphic-Kiliani">Yiheng Zhang</a><sup>1*</sup>,
  <a href="https://www.caizhuojiang.com/">Zhuojiang Cai</a><sup>2*</sup>,
  <a href="https://openreview.net/profile?id=~Mingdao_Wang1">Mingdao Wang</a><sup>1*</sup>,
  <a href="https://openreview.net/profile?id=~Meitong_Guo1">Meitong Guo</a><sup>1</sup>,
  <a href="https://github.com/tingyunaiai9">Tianxiao Li</a><sup>1</sup>,
  <a href="https://xplorestaging.ieee.org/author/37088600614">Li Lin</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=KhFGpFIAAAAJ&hl=en">Yuwang Wang</a><sup>1†</sup>
</h4>

<p align="center" style="margin:0.2rem 0 0.6rem 0;">
  <sup>1</sup> Tsinghua University &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>2</sup> Beihang University &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>3</sup> Migu Beijing Research Institute
</p>

<p align="center" style="font-size:0.95em; color:#666; margin-top:0;">
  * Equal contribution &nbsp;&nbsp;|&nbsp;&nbsp; † Corresponding author
</p>

<p align="center">
  <a href="https://graphic-kiliani.github.io/M3DLayout/">
    <img src="https://img.shields.io/badge/Project%20Page-blue.svg" alt="Project Page" height="22">
  </a>
  <a href="https://arxiv.org/abs/2509.23728">
      <img src="https://img.shields.io/badge/arXiv-b31b1b.svg?logo=arXiv&logoColor=white" alt="arXiv height="22">
  </a>
  <a href="https://huggingface.co/datasets/Metaverse-AI-Lab/M3DLayout">
    <img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face" height="22">
  </a>  
</p>



<p align="center">
    <img width="90%" alt="pipeline", src="./assets/Teaser.png">
</p>
</h4>


## TODO
We have fully released our **Dataset** (Scene & Layout & Rendering), **Inference & Training** and **Data Curation** (Object Retrieval & Rendering & Description Generation), come and try it!!!
- [x] Release Object Retrieval code of M3DLayout
- [x] Release rendering code of layouts and scenes
- [x] Release inference code of M3DLayout
- [x] Release M3DLayout dataset
- [x] Provide training instruction for M3DLayout
- [x] Provide Data curation instruction for M3DLayout's three data resources (3D-Front, Matterport3D, Infinigen)

<p align="center">
    <img width="100%" alt="demo", src="./assets/demo.gif">
</p>

## Installation
```bash
conda create -n m3dlayout python=3.10 -y
conda activate m3dlayout
conda install cuda -c nvidia/label/cuda-12.1.0 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --use-pep517 --no-build-isolation
```

## Download Dataset
Please download our processed dataset from [Baidu Netdisk](https://pan.baidu.com/s/1I4GRh3VfWw1nyIMVHRyF5A?pwd=jdek) or [Huggingface](https://huggingface.co/datasets/Metaverse-AI-Lab/M3DLayout).

The dataset is separated into 3 parts:

| Type | Description | Content | Size |
|---|---|---|---|
| `scene_dataset` | Origin scene with separated object geometries and textures | • Infinigen (8392 rooms with normal object density + 7607 rooms with relatively low object density = 15999 rooms): scene.blend, scene with segemented objects and textures. <br>• Matterport3D (90 houses): Postprocessed by segmenting each house into separated ply objects, you can import each subdir as a whole to Blender to receive a complete scene.<br>• 3D-Front (5173 rooms): Since we have not applied any additional processing, you can download the original 3D-FRONT dataset directly from [3D-Front official link](https://huggingface.co/datasets/huanngzh/3D-Front) | 3T before uncompress |
| `rendering_dataset` | Rendered Images from scene | • Infinigen (15864 rooms): Floor masks, Oblique-view scene renderings, Top-down scene renderings, Text descriptions, Detailed per-scene JSON <br>• Matterport3D (90 houses + 4 json files + 1 README.md): Floor masks for each room, Top-down layout renderings for each room, Multi-Level Detailed per-scene JSON   <br>• 3D-Front (5173 rooms): Floor masks, Top-down scene renderings | 250GB before uncompress |
| `layout_dataset` | Layout extracted from scene | **`<data_source>_train.json`**, **`<data_source>_test.json`**, and **`<data_source>_val.json`** for Infinigen & Matterport3D & 3D-Front. Including object count, category, location, bbox size, rotation, multi-level detailed description, etc. | 31MB before uncompress |

To be simple, 

If you want to do Scene Generation/Understanding/Reconstruction, Embodied AI and so on, you can directly download the **`scene_dataset`**. Moreover, you can extract point cloud or do further Detection, Segmentation or Editing tasks since all objects in the scene are clearly separated.

If you want to do some image/text to layout/scene or some 2D tasks, you can download **`rendering_dataset`**.

If you want to utilize the intermediate scene layout for your downstream research, you can download **`layout_dataset`**.

We have provided abundant functions in `render.py`,  `util.py` and `visualization_mlayout.py` from [Object-Retrieval-Layout2Scene](https://github.com/Graphic-Kiliani/Object-Retrieval-Layout2Scene/tree/432d4c22dbd2d16e09d6c81629f124e523f0dc6a) to postprocess (visualize/filter/rendering etc. ) the infinigen scene data.



## Download Weights
Please download the model weights from [Google Drive](https://drive.google.com/drive/folders/19GMTUk92fC6FB_dHu0aUw5Whc4j2KEWS?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1F4JtU0yxA6Zi7tYb5dcK5g?pwd=63w9), and place them in the `./weights` directory.

## Inference
To run the Gradio demo for 3d layout generation from arbitrary text:
```bash
python gradio_demo.py
```

For batch inference from prompt text files (default: 3 files, 1500 prompts total):
```bash
python example.py  # Default: Autoregressive
```

Useful options:
```bash
# Enable viz3dl GIF rendering and saving
python example.py --render-gif

# Use diffusion model
python example.py --model diffusion

# Customize output directory
python example.py --output-dir outputs/my_batch_run
```

Notes:
- One input txt file produces one output json file.
- The 1500 prompts are the test prompt set used in the original M3DLayout paper.

## Training

Place layout dataset JSON files under `data/m3dlayout/` (as referenced by `config/m3dlayout_*.yaml`).

Train autoregressive model:
```bash
python scripts/train_autoregressive.py \
  --config_file config/m3dlayout_autoregressive.yaml \
  --experiment_tag m3dlayout_autoregressive
```

Train diffusion model:
```bash
python scripts/train_diffusion.py \
  --config_file config/m3dlayout_diffusion.yaml \
  --experiment_tag m3dlayout_diffusion
```

## Data Curation
We provide complete [processing pipelines](https://github.com/Graphic-Kiliani/M3DLayout-code/tree/main/data_curation) for the 3D-Front, Matterport3D, and Infinigen datasets. We hope these tools will help the community scale up the dataset with additional computing and storage resources.





## Citation
If you find our work helpful, please consider citing:
```bibtex
@article{zhang2025m3dlayout,
      title={M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation}, 
      author={Yiheng, Zhang and Zhuojiang, Cai and Mingdao, Wang and Meitong, Guo and Tianxiao, Li and Li, Lin and Yuwang, Wang},
      journal={arXiv preprint arXiv:2509.23728},
      year={2025},
      url={https://arxiv.org/abs/2509.23728}, 
}
```

## Acknowledgements
Our code borrows from [ATISS](https://github.com/nv-tlabs/ATISS) and [DiffuScene](https://github.com/tangjiapeng/DiffuScene). We thank them for their excellent work. Please follow their licenses when using this part of the code.