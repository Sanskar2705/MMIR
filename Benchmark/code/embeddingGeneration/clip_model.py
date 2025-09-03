import os
import json
import torch
from tqdm import tqdm
import clip
import pandas as pd
from PIL import Image
from contextlib import asynccontextmanager
import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

import argparse





def load_clip_model_openai(model_variant: str):
    """
    Load OpenAI CLIP model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading OpenAI CLIP model on {device}")
    model, transform = clip.load(model_variant, device=device)
    model.eval()
    return model, transform, device


def generate_caption_embeddings(json_file, model, device, output_file):
    """
    Generate CLIP embeddings for the first 4 captions of each image.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data)} image-caption pairs to process.")

    results = []
    for item in tqdm(data, desc="Embedding captions"):
        image_name = item["image"]
        # captions = item["caption"][:4]  # Take only the first 4 captions
        captions = item["caption"]

        for caption in captions:
            try:
                text_token = clip.tokenize([caption], truncate=True).to(device)
                with torch.no_grad():
                    features = model.encode_text(text_token)
                    features = features / features.norm(dim=-1, keepdim=True)

                embedding = features.cpu().numpy().flatten().tolist()
                results.append({
                    "image_name": image_name,
                    "caption": caption.strip(),
                    "embedding": embedding
                })

            except Exception as e:
                print(f"Error processing caption: {caption} -> {e}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} caption embeddings to {output_file}")
    if results:
        print(f"Embedding vector size: {len(results[0]['embedding'])}")


def generate_image_embeddings(image_folder, json_file, model, transform, device, output_file):
    """
    Generate image embeddings using CLIP and save them as JSON. 
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data)} images to process.")

    embeddings = []
    for item in tqdm(data, desc="Embedding images"):
        image_name = item['image']
        image_filename = os.path.basename(image_name)
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_name}")
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model.encode_image(img_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            embedding = features.cpu().numpy().flatten().tolist()
            embeddings.append({
                "image_name": image_name,
                "embedding": embedding
            })

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    with open(output_file, "w") as f:
        json.dump(embeddings, f, indent=2)

    print(f"\nSaved {len(embeddings)} embeddings to {output_file}")
    if embeddings:
        print(f"Embedding vector size: {len(embeddings[0]['embedding'])}")

def main():

    parser = argparse.ArgumentParser(description="This code will generate clip embeddings")

    # Load config
    config = load_config()
    selected_dataset = config["selected_config"]["dataset"]

    # Optional argument
    parser.add_argument("--modality", "-m", help="Modality of embeddings to be generated", default="text")
    parser.add_argument("--dataset", "-d", help="Dataset of which embeddings will be generated. Default value from selected_config in config file.", default=selected_dataset)

    args = parser.parse_args()
    modality = args.modality
    dataset_name = args.dataset


    
    json_file = config["paths"]["dataset"][dataset_name]["ingest_annotations_path"]
    output_file = os.path.join(config["paths"]["project_root"],"data","embeddings",f"clip_{dataset_name}_{modality}_embeddings.json")
    image_folder = config["paths"]["dataset"][dataset_name]["base_image_path"]
    model_variant = config["models"]["clip"]


    model, transform, device = load_clip_model_openai(model_variant)

    if modality == "text":
        generate_caption_embeddings(json_file, model, device, output_file)
    elif modality == "image":
        generate_image_embeddings(image_folder, json_file, model, transform, device, output_file)
    else:
        print("Invalid Modality")



if __name__ == "__main__":
    main()
