### StyleProtect: Safeguarding Artistic Identity in Fine-tuned Diffusion Models

Qiuyu Tang, Joshua Krinsky, Aparna Bharati

[[Paper](https://openaccess.thecvf.com/content/CVPR2026W/APAI/html/Tang_StyleProtect_Safeguarding_Artistic_Identity_in_Finetuned_Diffusion_Models_CVPRW_2026_paper.html)]

**TL;DR:** StyleProtect adds imperceptible perturbations to artworks so DreamBooth-style finetuning cannot copy the artist's style, by updating only style-sensitive cross-attention layers.

This repo is official code space for StyleProtect, including main method (StyleProtect), evaluation in /eval, and apply post-process for robustness check in /robustness.

#### Environment set-up
``` 
conda env create -f environment.yml
```

#### Dataset
The refined WikiArt and Anita Dataset are found in [Google Drive](https://drive.google.com/drive/folders/1EMlAoOAEKi_bqLabiUUXnpo2G1w-n2Jn?usp=sharing).

#### Run protection
First, change the path in train.sh file.
```
sh train.sh
```

#### Evaluation
Download the CSD [checkpoint](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view) and put checkpoint.pth in /eval/model/.
```
sh eval.sh
```

#### Citation
If helpful, please consider citing us as follows:

```bibtex
@InProceedings{Tang_2026_CVPR,
    author    = {Tang, Qiuyu and Krinsky, Joshua and Bharati, Aparna},
    title     = {StyleProtect: Safeguarding Artistic Identity in Finetuned Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2026},
    pages     = {10759-10769}
}
```

