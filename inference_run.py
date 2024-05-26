import torch
from transformers import Trainer, AutoTokenizer, TrainingArguments
from model.model import HatefulMemesData, HateMemeClassifier, compute_metrics
from safetensors.torch import load_file
import os

from copy import deepcopy



import os
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, VisualBertModel, VisualBertConfig,BertModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from datasets import load_metric
import argparse

class HateMemeClassifier(torch.nn.Module):
    def __init__(self,fusion_method, visual_embedder='vit'):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        fusion_method: 'concatenate' or 'weight_ensemble' or 'linear_weight_ensemble' or 'visualbert'
        visual_embedder: 'vit'
        """
        super(HateMemeClassifier, self).__init__()
        self.fusion_method = fusion_method # 'concatenate' or 'weight_ensemble' or 'linear_weight_ensemble' or 'visualbert'
        configuration = VisualBertConfig.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre',
                                                hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        self.visualbert = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre', config=configuration)
        if self.fusion_method != 'visualbert':
            self.bertmodel = BertModel.from_pretrained('bert-base-uncased')

        if visual_embedder=='vit':
            self.embed_cls = nn.Linear(768, 1024)
        self.num_labels = 2
        self.dropout = nn.Dropout(0.3)
    
        if self.fusion_method=='weight_ensemble':
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Initial value of alpha
            
        if self.fusion_method=='linear_weight_ensemble':
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Initial value of alpha
            self.cls_visual = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.BatchNorm1d(768),
            )
            self.cls_text = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.BatchNorm1d(768),
            )

        self.cls=nn.Linear(768, self.num_labels)

        # Calculate the weights for the loss function and weight balanced loss
        nSamples = [5450,3050]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.loss_fct = CrossEntropyLoss(weight=torch.FloatTensor(normedWeights))

        # self.loss_fct = CrossEntropyLoss()
        
    
    def forward(self, input_ids, attention_mask, token_type_ids, visual_embeds, visual_attention_mask,
                visual_token_type_ids, labels,caption_input_ids, caption_attention_mask, caption_token_type_ids):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        visual_embeds_cls = self.embed_cls(visual_embeds)
        
        outputs = self.visualbert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_embeds=visual_embeds_cls,
                visual_attention_mask=visual_attention_mask,
                visual_token_type_ids=visual_token_type_ids,
            )
        
        visualbert_embedding = outputs[1] # output is a context vector of 768 dimensions

        if self.fusion_method != "visualbert":
        
            caption_outputs = self.bertmodel(caption_input_ids, attention_mask=caption_attention_mask, token_type_ids=caption_token_type_ids)
                    
            # Get the embeddings of the [CLS] token
            caption_embeddings = caption_outputs.last_hidden_state[:,0,:] # output is a context vector of 768 dimensions
        
        if self.fusion_method=='weight_ensemble':
            # funsion model: weight ensenble of the two embeddings: alpha*visualbert_embedding + (1-alpha)*caption_embeddings 
            fused_embedding = self.alpha * self.dropout(visualbert_embedding) + (1-self.alpha) * self.dropout(caption_embeddings)
            # self.wandb_run.log({"alpha": self.alpha.data.cpu().numpy()},commit=False)
        
            logits = self.cls(fused_embedding)
        if self.fusion_method=='linear_weight_ensemble':
            # funsion model: weight ensenble of the two embeddings: alpha*visualbert_embedding + (1-alpha)*caption_embeddings 
            fused_embedding = self.alpha * self.cls_visual(visualbert_embedding) + (1-self.alpha) * self.cls_text(caption_embeddings)
            # self.wandb_run.log({"alpha": self.alpha.data.cpu().numpy()},commit=False)
        
            logits = self.cls(fused_embedding)
        
        if self.fusion_method=='visualbert':
            logits = self.cls(self.dropout(visualbert_embedding))
            
        if self.fusion_method=='concatenate':
            # funsion model: concatenate the two embeddings
            fused_embedding = torch.cat((visualbert_embedding, caption_embeddings), dim=1)
            logits = self.cls(fused_embedding)
        
        
        reshaped_logits = logits.view(-1, self.num_labels)
        loss = self.loss_fct(reshaped_logits, labels.view(-1))
      
        return loss, reshaped_logits

def parse_args():
    parser = argparse.ArgumentParser(description='Train Hateful meme model')

    parser.add_argument(
        "--data-folder-path",
        type=str,
        help="data folder path for hatefulmemes dataset. ex. ../data/hateful_memes",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="to save checkpoint",)
    
    return parser.parse_args()
    
def main():
    args=parse_args()
    # Prepare directories
    data_folder_path=args.data_folder_path
    checkpoint_path=args.checkpoint_path
    query = 'query_8' 

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Define paths for evaluation data
    validation_data_path = os.path.join(data_folder_path, 'query236/dev_seen_.jsonl')
    img_inpainted_dir = os.path.join(data_folder_path, 'img')
    visual_embed_model = 'vit'  
    fusion_method = 'weight_ensemble'
    seq_len = 50

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HateMemeClassifier(fusion_method=fusion_method, visual_embedder=visual_embed_model)
    model = model.to(device)

    # Load the model weights from the safetensors file
    state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(state_dict)

    # Prepare evaluation dataset
    eval_dataset = HatefulMemesData(
        validation_data_path, 
        img_inpainted_dir, 
        tokenizer, 
        sequence_length=seq_len, 
        query=query, 
        visual_embed_model=visual_embed_model, 
        device=device
    )

    # Define training arguments (only evaluation relevant parameters)
    train_args = TrainingArguments(
        output_dir="./results",  
        per_device_eval_batch_size=24,
	    report_to='none'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Evaluate the model
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()
