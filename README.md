### StyleProtect: Safeguarding Artistic Identity in Fine-tuned Diffusion Models

This repo is official code space for StyleProtect, including main method (StyleProect), evaluation in /eval, and apply post-process for robustness check in /robustness.

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

