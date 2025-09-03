
import torch
from PIL import Image

from lavis.models import load_model_and_preprocess
import time

from ingest_data import IngestToDB

# #### Load a dataset image
from lavis.datasets.builders import load_dataset
# coco_dataset = load_dataset("coco_caption",vis_path="/mnt/storage/bharati/data/lavis/datasets/coco/images")
coco_dataset = load_dataset("coco_caption",vis_path="/mnt/storage/RSystemsBenchmarking/data/datasets/coco/images")


# setup device to use
MODEL_NAME = 'albef'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess(name=f"{MODEL_NAME}_feature_extractor", model_type="base", is_eval=True, device=device)


# Setup DB
IngestToDB.connection_pool = IngestToDB.connect_to_db()




# Process and Store Data
time_start = time.perf_counter()

db = IngestToDB()
for split in coco_dataset:
    for data in coco_dataset[split]:
        # Process and Store Data
        time_start_img = time.perf_counter()
        raw_image = data["image"]
        caption = data["text_input"]
        img_path = data["image_path"]

    
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        text_input = txt_processors["eval"](caption)
        sample = {"image": image, "text_input": [text_input]}

        # Multimodal features
        features_multimodal = model.extract_features(sample)
        data_toingest = {
            'table_name':f'image_embeddings_{MODEL_NAME}_image_text',
            'image_name':data['image_id'],
            'image_path':img_path,
            'embedding':features_multimodal.multimodal_embeds.reshape(-1).cpu().numpy().tolist(),
            'other_metadata':{
                'caption' : caption
            }
        }
        db.save_embedding_to_db(data_toingest)


        # features_image = model.extract_features(sample, mode="image")
        # data_toingest = {
        #     'table_name':f'image_embeddings_{MODEL_NAME}_image',
        #     'image_name':data['image_id'],
        #     'image_path':img_path,
        #     'embedding':features_image.image_embeds.reshape(-1).cpu().numpy().tolist(),
        #     'other_metadata':{
        #         'caption' : caption
        #     }
        # }
        # db.save_embedding_to_db(data_toingest)


        features_text = model.extract_features(sample, mode="text")
        data_toingest = {
            'table_name':f'image_embeddings_{MODEL_NAME}_text',
            'image_name':data['image_id'],
            'image_path':img_path,
            'embedding':features_text.text_embeds.reshape(-1).cpu().numpy().tolist(),
            'other_metadata':{
                'caption' : caption
            }
        }
        db.save_embedding_to_db(data_toingest)

        time_end_img = time.perf_counter()        

        print(f"[STATS]:[{data['image_id']}]IngestionTime={time_end_img-time_start_img}",flush=True)

        
time_end = time.perf_counter()        

print(f"[STATS]:IngestionTime={time_end-time_start}",flush=True)

