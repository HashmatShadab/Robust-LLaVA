# **Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models**

[Hashmat Shadab Malik](https://github.com/HashmatShadab),
[ Fahad Shamshad](https://scholar.google.com.pk/citations?user=d7QL4wkAAAAJ&hl=en),
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[ Karthik Nandakumar](https://scholar.google.com/citations?user=2qx0RnEAAAAJ&hl=en),
[Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en),
and [Salman Khan](https://salman-h-khan.github.io)

#### **Mohamed bin Zayed University of AI (MBZUAI)**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.08486)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://hashmatshadab.github.io/Robust-LLaVA/)


[//]: # ([![Video]&#40;https://img.shields.io/badge/Video-Presentation-F9D371&#41;]&#40;https://drive.google.com/file/d/1ZdUV83RvyL4mqCyxlqqD468VbRRAGdDY/view?usp=sharing&#41;)

[//]: # ([![slides]&#40;https://img.shields.io/badge/Poster-PDF-87CEEB&#41;]&#40;https://drive.google.com/file/d/1fvR4KUFCAEFO7wZqr-f8isk5FYMQvsT9/view?usp=sharing&#41;)

[//]: # ([![slides]&#40;https://img.shields.io/badge/Presentation-Slides-B762C1&#41;]&#40;https://drive.google.com/file/d/1osaG-OsgUlODRfRqDPK6f79bOLmRcC42/view?usp=sharing&#41;)


<hr />

# :fire: Updates

* **(Jan 24, 2025)**
    * Training and evaluation codes are released.
    * Robust-LLaVA-H and Robust-LLaVA-G released: Excited to release the new integration of LLaVA with large-scale
      robust image encoders, ViT-H and ViT-G, respectively. :fire::fire:

<hr />
<div align="center">
    <img src="./assets/fig1.png" alt="Robust-LLaVA Diagram" width="600">

<p align="justify">
<b>Robust accuracy of Robust-LLaVA<sup>4</sup> on downstream vision-language tasks
with adversarial examples crafted at &epsilon; = 4/255:</b> The original CLIP integrated into LLaVA exhibits 
<i>minimal robustness</i>. Our proposed <b>Robust-LLaVA<sup>4</sup></b> <b>outperforms</b> state-of-the-art 
robust CLIP models, such as <span><b><a href="https://arxiv.org/abs/2402.12336" target="_blank"
style="color: #007bff; text-decoration: underline;">FARE<sup>4</sup></a></b></span> 
and <span><b><a href="https://arxiv.org/abs/2409.07353" target="_blank"
style="color: #007bff; text-decoration: underline;">Sim-CLIP<sup>4</sup></a></b></span> 
in <b>robust accuracy across all tasks and diverse datasets</b>, while <i>maintaining high clean accuracy</i>.
</p>
</div>

<div style="text-align: justify;">

> **Abstract:** Multi-modal Large Language Models (MLLMs) have demonstrated impressive capabilities in vision-language
> tasks, but their reliance on visual processing introduces critical security vulnerabilities. Their vision encoders
> remain susceptible to adversarial perturbations that can induce hallucinations, manipulate responses, or bypass safety
> mechanisms while maintaining coherent language generation. Current approaches attempt to address this by adversarially
> fine-tuning CLIP vision encoders on ImageNet-scale data, but exhibit inherent limitations in both robustness and
> generalization due to the restricted scale and diversity of adversarial training.
> In this work, we present an alternative approach by leveraging vision encoders adversarially pre-trained on
> billion-scale image-text pairs.
> Our analysis reveals two principal contributions:
(1) the extensive scale and diversity of adversarial pre-training enables these encoders to demonstrate
> superior robustness against diverse adversarial threats, ranging from imperceptible perturbations to advanced
> jailbreaking attempts , without requiring additional adversarial training, and (2) end-to-end MLLM optimization
> with these robust encoders facilitates enhanced adaptation of language components to robust visual features,
> substantially outperforming existing plug-and-play methodologies on complex reasoning tasks.
> Through systematic evaluation across visual question-answering, image captioning, and jail-break attacks, we
> demonstrate
> that MLLMs trained with these robust encoders achieve superior adversarial robustness while maintaining favorable
> clean
> performance. Our framework achieves 2× and 1.5× average robustness gains in captioning and VQA tasks, respectively,
> and
> delivers over 10% improvement against advanced jailbreaking attacks compared to state-of-the-art methods.

</div>


## Quantitative Evaluation 📊

We provide instructions to reproduce Robust-LLaVA results on ___. Please follow the instructions at eval/README.md.

<div align="center">
    <img src="./assets/Table1.png" alt="Robust-LLaVA Diagram" width="800">

<p align="justify">
<b>On <b>untargeted attacks</b>, results across </b><b>six datasets</b>, covering <i>image
                        captioning</i> and <i>visual question answering</i> tasks,
                        both <span><b>Robust-LLaVA<sup>4</sup><sub>G</sub></b></span> and
                        <span><b>Robust-LLaVA<sup>4</sup><sub>H</sub></b></span> maintain
                        <i>reasonable clean performance</i> while achieving <b>substantial robustness improvements</b>
                        over <span><b><a href="https://arxiv.org/abs/2402.12336" target="_blank"
                                         style="color: #007bff; text-decoration: underline;">FARE<sup>4</sup></a></b></span>
                        and <span><b><a href="https://arxiv.org/abs/2409.07353" target="_blank"
                                        style="color: #007bff; text-decoration: underline;">Sim-CLIP<sup>4</sup></a></b></span>
                        against
                        adversarial attacks, striking the <i>right balance</i> between <b>clean</b> and <b>adversarial
                        generalization</b>.
</p>
</div>

<div align="center">
    <img src="./assets/Table2.png" alt="Robust-LLaVA Diagram" width="800">

<p align="justify">
Both <span><a href="https://arxiv.org/abs/2402.12336" target="_blank"
                                         style="color: #007bff; text-decoration: underline;">FARE<sup>4</sup></a></span>
                        and <span><b><a href="https://arxiv.org/abs/2409.07353" target="_blank"
                                        style="color: #007bff; text-decoration: underline;">Sim-CLIP<sup>4</sup></a></b></span>
                        show <i>robustness</i>
                        against
                        <b>targeted attacks</b>, but <i>break</i> in a few cases at high perturbation budgets (<span><b>ε = 8/255</b></span>).
                        In contrast, <span><b>Robust-LLaVA<sup>4</sup><sub>G</sub></b></span> and
                        <span><b>Robust-LLaVA<sup>4</sup><sub>H</sub></b></span>
                        remain <b>fully robust</b> to these attacks even at high perturbation budgets.
                        This indicates a <i>strong resistance</i> to generating the attacker's targeted output.
                        The robustness of <span><b>Robust-LLaVA<sup>4</sup><sub>G</sub></b></span> stands out further as
                        it continues to generate
                        <i>high-quality captions</i> for adversarial examples, maintaining a <b>strong CIDEr score</b>.
</p>
</div>

<div align="center">
    <img src="./assets/Table3_4.png" alt="Robust-LLaVA Diagram" width="800">
</div>


<p align="justify">
<b>Comparison of various <span><b>vision encoders</b></span> integrated with <b>LLaVA</b> against <b>white-box</b> (<i><a href="https://arxiv.org/abs/2306.13213" target="_blank" style="color: #007bff; text-decoration: underline;">VisualAdv</a></i>)
and <b>black-box</b> (<i><a href="https://arxiv.org/abs/2403.09792" target="_blank" style="color: #007bff; text-decoration: underline;">HADES</a></i>) 
jailbreak attacks.</b> The <b>white-box results</b> (Table 3) show that <i>LLaVA with the original CLIP encoder</i> is the 
<b>most vulnerable</b>, producing the highest number of toxic outputs. In contrast, our 
<span><b>Robust-LLaVA<sup>4</sup><sub>G</sub></b></span> and 
<span><b>Robust-LLaVA<sup>4</sup><sub>H</sub></b></span> models <b>significantly reduce toxic content generation</b>. The <b>black-box results</b> (Table 4) highlight the effectiveness of different models against 
<i>HADES attacks</i>, with the <i>original CLIP encoder</i> exhibiting the 
<b>highest Attack Success Rate (ASR)</b>. In contrast, our 
<span><b>Robust-LLaVA</b></span> models achieve the <b>lowest ASR</b>, 
demonstrating <i>superior resilience</i> across multiple adversarial scenarios.
</p>

---

## Training :train:
We provide scripts for pretraining and finetuning of VideoGPT+. Please follow the instructions at [scripts/README.md](scripts/README.md).

---


## Qualitative Analysis :mag:

### Untargetted Attack on Image Captioning Task

<div style="width:auto; height:145px; overflow:hidden;">
  <img src="./assets/IC_U_1.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:140px; overflow:hidden;">
  <img src="./assets/IC_U_2.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:142px; overflow:hidden;">
  <img src="./assets/IC_U_3.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:134px; overflow:hidden;">
  <img src="./assets/IC_U_4.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:132px; overflow:hidden;">
  <img src="./assets/IC_U_6.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:162px; overflow:hidden;">
  <img src="./assets/IC_U_7.png" style="width:auto; height:auto;">
</div>


### Targetted Attack on Image Captioning Task

<div style="width:auto; height:145px; overflow:hidden;">
  <img src="./assets/IC_T_1.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:140px; overflow:hidden;">
  <img src="./assets/IC_T_2.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:142px; overflow:hidden;">
  <img src="./assets/IC_T_3.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:134px; overflow:hidden;">
  <img src="./assets/IC_T_4.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:132px; overflow:hidden;">
  <img src="./assets/IC_T_5.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:162px; overflow:hidden;">
  <img src="./assets/IC_T_6.png" style="width:auto; height:auto;">
</div>

### Untargetted Attack on Visual Question Answering(VQA) Task

<div style="width:auto; height:145px; overflow:hidden;">
  <img src="./assets/VQA_U_1.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:140px; overflow:hidden;">
  <img src="./assets/VQA_U_2.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:142px; overflow:hidden;">
  <img src="./assets/VQA_U_3.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:134px; overflow:hidden;">
  <img src="./assets/VQA_U_4.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:132px; overflow:hidden;">
  <img src="./assets/VQA_U_5.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:162px; overflow:hidden;">
  <img src="./assets/VQA_U_6.png" style="width:auto; height:auto;">
</div>


### Robustness to Common Corruptions on Image Captioning Task

<div style="width:auto; height:145px; overflow:hidden;">
  <img src="./assets/CC_1.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:140px; overflow:hidden;">
  <img src="./assets/CC_2.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:142px; overflow:hidden;">
  <img src="./assets/CC_3.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:134px; overflow:hidden;">
  <img src="./assets/CC_4.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:142px; overflow:hidden;">
  <img src="./assets/CC_5.png" style="width:auto; height:auto;">
</div>

<div style="width:auto; height:134px; overflow:hidden;">
  <img src="./assets/CC_6.png" style="width:auto; height:auto;">
</div>




## Contents

1) [Installation](#Installation)
2) [Available Models](#Available-Models)
3) [Training](#Training)
4) [Robustness against White-Box Attacks](#Robustness-against-White-Box-Attacks)
5) [Robustness against Transfer-Based Black-Box Attacks](#Robustness-against-Transfer-Based-Black-Box-Attacks)
6) [BibTeX](#bibtex)
7) [Contact](#contact)
8) [References](#references)

<hr>
<hr>



<a name="Installation"/>

## 💿 Installation

You can follow the instrcutions mention in the [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install)
codebase to install the required dependencies or follow the below steps:

1. Clone the repository:

```bash
git clone https://github.com/HashmatShadab/Robust-LLaVA 
cd Robust-LLaVA
```

2. Install the required dependencies:

```python
conda create -n llava_v python=3.10 -y
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia 
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

pip install open-clip-torch==2.19.0
pip install pycocoevalcap==1.2
pip install inflection==0.5.1
pip install torchattacks

```

<a name="Available-Models"/>

## 🏁 Available Models

### Models available:


<a name="Training"/>

## 🚀 Training

```python

```

Follwing arguments can be passed for `--model_name`: `unet, unetr, swin_unetr, seg_resnet, umamba_bot, umamba_enc`

To run training across all models and datasets, run the following scripts:

```python

```

The logs and trained models will be saved in the `Results` folder with the following structure:
`Results/{model_name}/data_{dataset_name}/natural/`


<a name="Robustness-against-White-Box-Attacks"/>

## 🛡️ Robustness against White-Box Attacks

### 1. White box Attacks

```python

```


To run the above attacks across all models and datasets, run the following scripts:

```python
# Pixel and Frequency-based attacks on Volumetric Segmentation models trained on BTCV dataset
bash
scripts / btcv / attacks.sh

```




<a name="Robustness-against-Transfer-Based-Black-Box-Attacks"/>

## 🛡️ Robustness against Transfer-Based Black-Box Attacks

After generating adversarial examples using a surrogate model, the transferability of adversarial examples can be
reported by evaluating them on unseen target models trained on the same dataset.
To evaluate any target model on the adversarial examples, run the following script:

```python
# Transferability on BTCV adversarial examples

```






<a name="bibtex"/>

## 📜 BibTeX

```bibtex

```

<hr />

<a name="contact"/>

## 📧 Contact

Should you have any question, please create an issue on this repository or contact at hashmat.malik@mbzuai.ac.ae

<hr />

<a name="references"/>

## 📚 References

+ Our code base is build upon [LLaVA](https://github.com/haotian-liu/LLaVA) and [RobustVLM](https://github.com/chs20/RobustVLM).  We thank them for open-sourcing their codebase.


