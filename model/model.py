import torch, torchvision
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from copy import deepcopy

#can comment following if not using detectron2 for visual embeddings
# from model.visual_embedding.visual_embeding_detectron2 import VisualEmbedder
####

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, VisualBertModel, TrainingArguments, Trainer, VisualBertConfig
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import roc_auc_score
from datasets import load_metric
acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
precision_metric = load_metric('precision')
recall_metric = load_metric('recall')

from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset

class HatefulMemesData(Dataset):
    def __init__(self, df,img_dir, tokenizer, sequence_length,query,caption_sequence_length=512, visual_embed_model='vit', print_text=False, visual_embeder_detecron2=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):         
        self.device=device
        self.query=query
        self.sequence_length = sequence_length
        self.caption_sequence_length= caption_sequence_length
        self.tokenizer = tokenizer
        self.print_text = print_text
        self.dataset = pd.read_json(path_or_buf=df, lines=True).to_dict(orient='records')
        self.img_dir = img_dir
        self.visual_embed_model = visual_embed_model
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.feature_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(self.device)
        if self.visual_embed_model=='detectron2' and visual_embeder_detecron2 is not None:
            self.visualembedder=visual_embeder_detecron2

    def __len__(self):
        return len(self.dataset)


    def tokenize_data(self, example):
   
        idx = example['id']
        # idx = [idx] if isinstance(idx, str) else idx
        
        encoded_dict = self.tokenizer(example['text'], padding='max_length', max_length=self.sequence_length, truncation=True, return_tensors='pt')
        tokens = encoded_dict['input_ids']
        token_type_ids = encoded_dict['token_type_ids']
        attn_mask = encoded_dict['attention_mask']
        
        captioning_encode_dict=self.tokenizer(example[self.query], padding='max_length', max_length=self.caption_sequence_length,truncation=True, return_tensors='pt')
        caption_token=captioning_encode_dict['input_ids']
        caption_token_type_ids=captioning_encode_dict['token_type_ids']
        caption_attn_mask=captioning_encode_dict['attention_mask']

        targets = torch.tensor(example['label']).type(torch.int64)

        ## Get Visual Embeddings
        try:
            if self.visual_embed_model=='vit':
                #TODO: make it work
                img = example['img'].split('/')[-1]
                img = Image.open(os.path.join(self.img_dir , img))
                img = np.array(img)
                img = img[...,:3]
                inputs = self.feature_extractor(images=img, return_tensors="pt")
                outputs = self.feature_model(**inputs.to(self.device))
                visual_embeds = outputs.last_hidden_state
                visual_embeds = visual_embeds.cpu() #
            elif self.visual_embed_model=='detectron2':
                visual_embeds = self.visualembedder.visual_embeds_detectron2([cv2.imread(os.path.join(self.img_dir, example['img'].split('/')[-1]))])[0]

        except:
            # print("Error with Id: ", idx)
            if self.visual_embed_model=='vit':
                visual_embeds = np.zeros(shape=(197, 768), dtype=float)
            elif self.visual_embed_model=='detectron2':
                visual_embeds = np.zeros(shape=(100, 1024), dtype=float)

        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.int64)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.int64)

        inputs={"input_ids": tokens.squeeze(),
            "attention_mask": attn_mask.squeeze(),
            "token_type_ids": token_type_ids.squeeze(),
            "visual_embeds": visual_embeds.squeeze(),
            "visual_token_type_ids": visual_token_type_ids.squeeze(),
            "visual_attention_mask": visual_attention_mask.squeeze(),
            "label": targets.squeeze(),
            "caption_input_ids": caption_token.squeeze(),
            "caption_attention_mask": caption_attn_mask.squeeze(),
            "caption_token_type_ids": caption_token_type_ids.squeeze()
        }
        
        return inputs
  
    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])
        
        if self.print_text:
            for k in inputs.keys():
                print(k, inputs[k].shape, inputs[k].dtype)

        return inputs


# TODO: Add your fusion model here
class HateMemeClassifier(torch.nn.Module):
    def __init__(self,fusion_method, visual_embedder='vit',wandb_run=None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        fusion_method: 'concatenate' or 'weight_ensemble' or 'linear_weight_ensemble'
        visual_embedder: 'vit' or 'detectron2'
        """
        super(HateMemeClassifier, self).__init__()
        self.fusion_method = fusion_method # 'concatenate' or 'weight_ensemble' or 'linear_weight_ensemble'
        self.wandb_run=wandb_run

        configuration = VisualBertConfig.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre',
                                                hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        self.visualbert = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre', config=configuration)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')
        for param in self.visualbert.parameters():
            param.requires_grad = False
        for param in self.bertmodel.parameters():
            param.requires_grad = False

        if visual_embedder=='vit':
            self.embed_cls = nn.Linear(768, 1024)
        elif visual_embedder=='detectron2':
            self.embed_cls = nn.Linear(1024, 1024)
        self.num_labels = 2
        self.dropout = nn.Dropout(0.1)
        
    
        if self.fusion_method=='weight_ensemble':
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Initial value of alpha
            self.cls= nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(768, 348),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(348, 192),
                nn.ReLU(),
                nn.Linear(192, self.num_labels)
            )
        # TODO: Calculate the weights for the loss function and weight balanced loss
        # nSamples = [5178, 2849]
        # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        # self.loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(normedWeights))
        self.loss_fct = CrossEntropyLoss()
        
    
    def forward(self, input_ids, attention_mask, token_type_ids, visual_embeds, visual_attention_mask,
                visual_token_type_ids, labels,caption_input_ids, caption_attention_mask, caption_token_type_ids):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        visual_embeds_cls = self.embed_cls(visual_embeds)
        with torch.no_grad():
            outputs = self.visualbert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    visual_embeds=visual_embeds_cls,
                    visual_attention_mask=visual_attention_mask,
                    visual_token_type_ids=visual_token_type_ids,
                )
        
        visualbert_embedding = outputs[1] # output is a context vector of 768 dimensions

        
        with torch.no_grad():
            caption_outputs = self.bertmodel(caption_input_ids, attention_mask=caption_attention_mask, token_type_ids=caption_token_type_ids)
                
        # Get the embeddings of the [CLS] token
        caption_embeddings = caption_outputs.last_hidden_state[:,0,:] # output is a context vector of 768 dimensions
        
        if self.fusion_method=='weight_ensemble':
            # funsion model: weight ensenble of the two embeddings: alpha*visualbert_embedding + (1-alpha)*caption_embeddings 
            fused_embedding = self.alpha * self.dropout(visualbert_embedding) + (1-self.alpha) * self.dropout(caption_embeddings)
            self.wandb_run.log({"alpha": self.alpha.data.cpu().numpy()},commit=False)
        
        logits = self.cls(fused_embedding)
        ##
        
        reshaped_logits = logits.view(-1, self.num_labels)
        loss = self.loss_fct(reshaped_logits, labels.view(-1))
      
        return loss, reshaped_logits



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    auc_score = roc_auc_score(labels, predictions)
    return {"accuracy": acc['accuracy'], "auroc": auc_score,'f1':f1['f1'],'precision':precision['precision'],'recall':recall['recall']} 
