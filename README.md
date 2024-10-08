# RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation
<span class="author-block">
<a target="_blank" href="https://jianghanxiao.github.io/">Hanxiao Jiang</a><sup>1,2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://binghao-huang.github.io/">Binghao Huang</a><sup>1</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://warshallrho.github.io/">Ruihai Wu</a><sup>4</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://sg.linkedin.com/in/zhuoran-li-david">Zhuoran Li</a><sup>5</sup>,
</span>


<span class="author-block">
<a target="_blank" href="https://www.gargshubham.com/">Shubham Garg</a><sup>3</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://www.amazon.science/author/hooshang-nayyeri">Hooshang
    Nayyeri</a><sup>3</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://shenlong.web.illinois.edu/">Shenlong Wang</a><sup>2</sup>,
</span>
<span class="author-block">
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
</span>

<span class="author-block"><sup>1</sup>Columbia University,</span>
<span class="author-block"><sup>2</sup>University of Illinois Urbana-Champaign,</span>
<span class="author-block"><sup>3</sup>Amazon,</span>
<span class="author-block"><sup>4</sup>Peking University,</span>
<span class="author-block"><sup>5</sup>National University of Singapore</span>

### [Website](https://jianghanxiao.github.io/roboexp-web/) | [Paper](https://jianghanxiao.github.io/roboexp-web/roboexp.pdf) | [Colab](https://colab.research.google.com/drive/1xcteSyorfbiSzwHtZ77X4v6AlKcUyynK?usp=sharing) | [Video](https://www.youtube.com/watch?v=qaSbggX_tXU)


### Overview
This repository contains the official implementation of the **RoboEXP** system for the **interactive exploration** task.
<video autoplay muted loop height="100%" width="100%" controls src="https://github.com/Jianghanxiao/proj-RoboExp/assets/32015651/14cf19c6-5da0-4b84-a3a5-e4383396e3b9"></video>

### Try it in Colab! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xcteSyorfbiSzwHtZ77X4v6AlKcUyynK?usp=sharing)
In this [notebook](https://colab.research.google.com/drive/1xcteSyorfbiSzwHtZ77X4v6AlKcUyynK?usp=sharing), we replay an example exploration process using the saved observations and cached decisions. 

### Setup
```
# Here we use cuda-11.7
export CUDA_HOME=/usr/local/cuda-11.7/
# create conda environment
conda create -n roboexp python=3.8
conda activate roboexp

# Install the packages
bash scripts/install_env.sh

# download pretrained models
bash scripts/download_pretrained_models.sh
```

### Interactive Exploration
To run our RoboEXP system, you also need to create the file `my_apikey` and copy your OpenAI API key into the file.

To calibrate the wrist camera, we provide the hand-eye calibration code in `calibrate_wrist.py`, and you can download our calibration board.
```
# Download the calibration board
gdown 1KWYncDGjtGthePC3wzCu9zBW0nZI0RBM
# Download the example calibration result
gdown 1b2yp45eJVyXnOg11OImTER3zMRUcaZRN
```

Run our RoboEXP system
```
# Set `visualize=True` to enable visualizations
python interactive_explore.py 
```

### Code Structure Explanation
`roboexp`  is the main library for our RoboEXP system, comprising the following key components:
* `roboexp/env`: This component provides the basic control API for the robot, camera, and calibration setup. It facilitates obtaining observations from the camera and controlling the robot.
* `roboexp/perception`: This component offers an API for the perception module, enabling image processing to derive 2D semantic information from images.
* `roboexp/memory`: This component provides an API for the memory module, allowing instance merging in 3D and the construction of our Action-Conditioned Scene Graph (ACSG).
* `roboexp/decision`: This component offers an API for the decision module, facilitating decision-making based on observations.
* `roboexp/act`: This component provides an API for the action module, enabling action planning based on decisions and our memory.

### Citation
If you find this repo useful for your research, please consider citing the paper
```
@article{jiang2024roboexp,
    title={RoboEXP: Action-Conditioned Scene Graph via Interactive Exploration for Robotic Manipulation},
    author={Jiang, Hanxiao and Huang, Binghao and Wu, Ruihai and Li, Zhuoran and Garg, Shubham and Nayyeri, Hooshang and Wang, Shenlong and Li, Yunzhu},
    journal={CoRL},
    year={2024}
}
```
