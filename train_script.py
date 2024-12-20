from model.model import HatefulMemesData
from model.model import HateMemeClassifier
from model.model import compute_metrics
import argparse
import torch
import os
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from transformers import BertTokenizer, VisualBertForPreTraining, AutoTokenizer
import time
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='Train Hateful meme model')
    parser.add_argument(
        "--epochs",
        type=int,
        help="number of training epochs",
    )
    parser.add_argument(
        "--data-folder-path",
        type=str,
        help="data folder path for hatefulmemes dataset. ex. ../data/hateful_memes",
    )

    parser.add_argument(
        "--save-folder-path",
        type=str,
        help="to save checkpoint",
    )

    parser.add_argument(
        "--visual-embed-model",
        type=str,
        help="to save checkpoint",
        default='vit'
    )

    parser.add_argument(
        "--fusion-method",
        type=str,
        help="fusion method: 'concatenate' or 'weight_ensemble' or 'linear_weight_ensemble' or 'visualbert' (visualbert only)",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="which query to run, ex query_1 (general description) or query_8 (short description + race + religion + gender)",
    )

    parser.add_argument(
        "--wandb-key",
        type=str,
        help="wandb key path",
        default='None'
    )
    
    return parser.parse_args()
    



def main():
    args = parse_args()
    data_dir=args.data_folder_path
    save_dir=args.save_folder_path
    fusion_method=args.fusion_method
    wandb_key_path=args.wandb_key
    query=args.query

    if wandb_key_path != 'None':
        with open(wandb_key_path, "r") as f:
            wandb_key = f.read().strip()
            
        wandb.login(key=wandb_key)
        wandb_run=wandb.init(
            project="HatefulMemes",
        )
    
    else:
        wandb_run=None

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_data_path=os.path.join(data_dir, 'query236/train_.jsonl')
    validation_data_path=os.path.join(data_dir, 'query236/dev_seen_.jsonl')
    img_inpainted_dir=os.path.join(data_dir, 'img')
    visual_embed_model=args.visual_embed_model
    output_dir=os.path.join(save_dir, f'hatefulmemcladdifier_{fusion_method}_{visual_embed_model}_{time.strftime("%Y%m%d%H%M")}')

    batch_size = 24
    seq_len = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HateMemeClassifier(fusion_method=fusion_method, visual_embedder=visual_embed_model,wandb_run=wandb_run)
    model = model.to(device)

    args = TrainingArguments(
        output_dir = output_dir,
        seed = 110, 
        evaluation_strategy = "steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs= args.epochs,
        weight_decay=0.05,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="auroc",
        eval_steps = 50,
        save_steps = 100,
        logging_steps=10,
        fp16 = False,
        gradient_accumulation_steps = 2
    )


    trainer = Trainer(
        model,
        args,
        train_dataset = HatefulMemesData(train_data_path, img_inpainted_dir, tokenizer, sequence_length=seq_len,query=query, visual_embed_model=visual_embed_model,device=device),
        eval_dataset =  HatefulMemesData(validation_data_path, img_inpainted_dir,tokenizer, sequence_length=seq_len,query=query, visual_embed_model=visual_embed_model,device=device),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()
