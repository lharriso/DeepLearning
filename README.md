# A Multimodal approach to hate speech detection on memes
The project is part of the course EE559 Deep Leaning in EPFL.

## Introduction
Hate speech is becoming an increasingly serious problem in society and has a severe impact on individuals' mental and emotional health. No field is safe from hate speech, including memes, which usually are used to convey a joke to make an individual laugh but can sadly also be used to communicate hate speech which is a serious issue we face today. Our project attempts to adress this problem by learning to classify whether a meme is considered as hate speech or not. We incorporate the image captioning of meme images with various questions. We leverage BERT and Visualbert as well as other machine learning techniques to analyze text and image data simultaneously in order to create a hate speech classifier.


## Model
We developed a new approach that incorporates image captioning with various types of questions alongside meme images and meme text. We propose a multimodal approach that combines both VisualBERT and BERT. VisualBERT is utilized to understand the relationship between the meme image and the meme
text, while BERT is employed to handle the image captioning with different questions. The outputs from these two submodels are then fused and processed through a multilayer perceptron (MLP) for binary classification, forming our final model.

![model architecture](/image/model_architecture.png)

Our model and the dataset class is defined in the file [model.py](model/model.py)

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
## Inference
Download our best model [best_model.zip](https://epflch-my.sharepoint.com/:f:/g/personal/michael_hauri_epfl_ch/EiAFUxrHm5lJjNhAfHo5Z5YBrhjJAkHeVUvkCwqL0AY9gw) and unzip it.
```
python inference_run.py --data-folder-path /your/data/path --checkpoint-path /best/model/path
```
## Evaluation
### Fusion Methods Comparison

In this experiment, we evaluate the performance of three fusion methods: concatenation, weighted ensemble, and linear weighted ensemble. 

#### Fusion Methods Comparison Table

| Fusion Methods         | AUC        | Accuracy  |
|------------------------|------------|-----------|
| Concatenate            | 63.0  | 63.1 |
| Weight Ensemble        | 64.2  | 64.2 |
| Linear Weight Ensemble | 62.0  | 62.2 |
| VisualBert Only        | 60.1  | 60.2 |

### General Description vs. Specific Questions

We compared our best model of part [Fusion Methods Comparison](#fusion-methods-comparison), where we used hateful-related questions to a detailed description of the image. Results are shown in the table below, and confirm that informative image captions provide significant improvement (4.4). This indicates that when the model is provided with information about race, religion, and gender identity, it learns better to classify whether content is hateful or not. This suggests that "hateful" speech is mostly related to these features.

#### Caption Comparison Table

| Image Caption         | AUC        | Accuracy  |
|-----------------------|------------|-----------|
| General description   | 59.8  | 60.0 |
| Specific questions    | 64.2  | 64.2 |


## Image Captioning
We also provide the code for generating the image captioning for differnt questions, which is in [vqa.py](vqa.py)

## Inpainted image
The code for generating the inpainted image could be found in [inpaiting.ipynb](OCR/inpainting.ipynb)

