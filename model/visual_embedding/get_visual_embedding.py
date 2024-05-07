import os
import pandas as pd
import cv2
from visual_embeding_detectron2 import VisualEmbedder
from tqdm import tqdm


train_data_path='../../data/hateful_memes/train_df_wQuery_.jsonl'
validation_data_path='../../data/hateful_memes/dev_seen_df_wQuery_.jsonl'
test_data_path='../../data/hateful_memes/test_seen_df_wQuery_.jsonl'
img_inpainted_dir='../../data/hateful_memes/img_inpainted'
cfg_path="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
MIN_BOXES=10 
MAX_BOXES=100

# ##Train Data
# data_path=train_data_path
# img_dir=img_inpainted_dir

# visualembedder=VisualEmbedder(cfg_path=cfg_path, min_boxes=MIN_BOXES, max_boxes=MAX_BOXES)

# img_data = pd.read_json(path_or_buf=data_path, lines=True).to_dict(orient='records')

# for i, item in enumerate(img_data):
#     image_list = [cv2.imread(os.path.join(img_dir, item['img'].split('/')[-1]))]
#     visual_embeds = visualembedder.visual_embeds_detectron2(image_list)
#     img_data[i]['visual_embedding'] = visual_embeds

# # Save the new img_data as json
# img_data_df = pd.DataFrame(img_data)
# img_data_df.to_json(data_path.replace('.jsonl', '_visual_embedded_imginpainted.jsonl'), orient='records', lines=True)


# ##Validation Data
# data_path=validation_data_path
# img_dir=img_inpainted_dir

# visualembedder=VisualEmbedder(cfg_path=cfg_path, min_boxes=MIN_BOXES, max_boxes=MAX_BOXES)

# img_data = pd.read_json(path_or_buf=data_path, lines=True).to_dict(orient='records')

# for i, item in enumerate(img_data):
#     image_list = [cv2.imread(os.path.join(img_dir, item['img'].split('/')[-1]))]
#     visual_embeds = visualembedder.visual_embeds_detectron2(image_list)
#     img_data[i]['visual_embedding'] = visual_embeds

# # Save the new img_data as json
# img_data_df = pd.DataFrame(img_data)
# img_data_df.to_json(data_path.replace('.jsonl', '_visual_embedded_imginpainted.jsonl'), orient='records', lines=True)

##Test Data
data_path=test_data_path
img_dir=img_inpainted_dir
save_interval = 100  # Save data every 100 images

visualembedder=VisualEmbedder(cfg_path=cfg_path, min_boxes=MIN_BOXES, max_boxes=MAX_BOXES)

img_data = pd.read_json(path_or_buf=data_path, lines=True).to_dict(orient='records')

for i, item in enumerate(tqdm(img_data)):
    image_list = [cv2.imread(os.path.join(img_dir, item['img'].split('/')[-1]))]
    visual_embeds = visualembedder.visual_embeds_detectron2(image_list)
    
    img_data[i]['visual_embedding'] = visual_embeds[0]
    # Save the data every 'save_interval' images
    if (i + 1) % save_interval == 0 or i == len(img_data) - 1:
        partial_data_df = pd.DataFrame(img_data[:i+1])
        # Save in append mode if not the first chunk
        if i >= save_interval:
            partial_data_df.to_json(data_path.replace('.jsonl', '_visual_embedded_imginpainted.jsonl'), orient='records', lines=True, mode='a', header=False)
        else:
            partial_data_df.to_json(data_path.replace('.jsonl', '_visual_embedded_imginpainted.jsonl'), orient='records', lines=True)

        print(f"Data saved up to index {i}")


