# A Multimodal approach to hate speech detection on memes
The project is part of the course EE559 Deep Leaning in EPFL.

## Introduction
TODO: Add here

## Dataset
Original Dataset can be download at this website: https://hatefulmemeschallenge.com/

Here we provided all the data (original and inpaited images) and also the generated results for image captioning in following link:https://epflch-my.sharepoint.com/:f:/g/personal/michael_hauri_epfl_ch/EiAFUxrHm5lJjNhAfHo5Z5YBrhjJAkHeVUvkCwqL0AY9gw

Download the image data "img", and query data "query236" under /data.

Your data folder should look like this:
```txt
.
├── data
│   ├── img
│   ├── query236
│     ├── dev_seen_.jsonl
│     ├── test_seen_.jsonl 
|     ├── train_.jsonl
|...
```
## Model
We developed a new approach that incorporates image captioning with various types of questions alongside meme images and meme text. We propose a multimodal approach that combines both VisualBERT and BERT. VisualBERT is utilized to understand the relationship between the meme image and the meme
text, while BERT is employed to handle the image captioning with different questions. The outputs from these two submodels are then fused and processed through a multilayer perceptron (MLP) for binary classification, forming our final model.

TODO: add model architecture image here

Our model and the dataset class is defined in the file [model.py](model/model.py)

## Set up environment
```
conda create --name <my-env> python=3.10
conda activate <my-env>
pip install -r requirements.txt
```

## Train
```
python train_script.py --epochs 5 --data-folder-path /your/data/path --save-folder-path you/save/folder/path --fusion-method weight_ensemble --query query_8
```
## Evaluation
### Fusion Methods Comparison

In this experiment, we evaluate the performance of three fusion methods: concatenation, weighted ensemble, and linear weighted ensemble. 

#### Fusion Methods Comparison Table

| Fusion Methods         | AUC        | Accuracy  |
|------------------------|------------|-----------|
| Concatenate            | 63.0 ± 0.0 | 63.1 ± 0.0|
| Weight Ensemble        | 64.2 ± 0.0 | 64.2 ± 0.0|
| Linear Weight Ensemble | 62.0 ± 0.0 | 62.2 ± 0.0|
| VisualBert Only        | 60.1 ± 0.0 | 60.2 ± 0.0|

### General Description vs. Specific Questions

We compared our best model of part [Fusion Methods Comparison](#fusion-methods-comparison), where we used hateful-related questions to a detailed description of the image. Results are shown in the table below, and confirm that informative image captions provide significant improvement (4.4 ± 0.0). This indicates that when the model is provided with information about race, religion, and gender identity, it learns better to classify whether content is hateful or not. This suggests that "hateful" speech is mostly related to these features.

#### Caption Comparison Table

| Image Caption         | AUC        | Accuracy  |
|-----------------------|------------|-----------|
| General description   | 59.8 ± 0.0 | 60.0 ± 0.0|
| Specific questions    | 64.2 ± 0.0 | 64.2 ± 0.0|


We also provide our best model [hatefulmemcladdifier_weight_ensemble_vit_smooth-totem-17_202405230523.zip](https://epflch-my.sharepoint.com/:f:/g/personal/michael_hauri_epfl_ch/EiAFUxrHm5lJjNhAfHo5Z5YBrhjJAkHeVUvkCwqL0AY9gw)


## Image Captioning
We also provide the code for generating the image captioning for differnt questions, which is in [vqa.py](vqa.py)

## Inpainted image
The code for generating the inpainted image could be found in [inpaiting.ipynb](OCR/inpainting.ipynb)

