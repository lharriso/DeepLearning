import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

file_path = "data/hateful_memes/train.jsonl"
train_df = pd.read_json(path_or_buf=file_path, lines=True)
print("number of training data points:",train_df.shape[0])

file_path = "data/hateful_memes/test_seen.jsonl"
test_seen_df = pd.read_json(path_or_buf=file_path, lines=True)
print("number of test seen data points:",test_seen_df.shape[0])

file_path = "data/hateful_memes/test_unseen.jsonl"
test_unseen_df = pd.read_json(path_or_buf=file_path, lines=True)
print("number of test unseen data points:",test_unseen_df.shape[0])

file_path = "data/hateful_memes/dev_seen.jsonl"
dev_seen_df = pd.read_json(path_or_buf=file_path, lines=True)
print("number of dev seen data points:",dev_seen_df.shape[0])

file_path = "data/hateful_memes/dev_unseen.jsonl"
dev_unseen_df = pd.read_json(path_or_buf=file_path, lines=True)
print("number of dev unseen data points:",dev_unseen_df.shape[0])



torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', trust_remote_code=True)

query = '<ImageHere>Please describe the image in detail'

df_list = [train_df,test_seen_df,dev_seen_df]
df_list_name = [ "train_df_wQuery","test_seen_df_wQuery","dev_seen_df_wQuery"]
for k,df in enumerate(df_list):
  df.insert(4, 'query', None)
  for i,img_name in enumerate(df['img']):
    if i % 10 == 0:
      print(i)
    img_path = 'data/hateful_memes/img_inpainted/'+img_name[4:]
    with torch.cuda.amp.autocast():
      response, _ = model.chat(tokenizer, query=query, image=img_path, history=[], do_sample=False)
      #save img in
      df['query'][i] = response

  df.to_json('data/hateful_memes/'+df_list_name[k]+'_'+'.jsonl', orient='records', lines=True)
