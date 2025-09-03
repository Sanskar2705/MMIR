import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from contextlib import asynccontextmanager
import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config
import argparse


def load_minilm_model(model_variant: str):
    """
    Load the MiniLM model from sentence-transformers.
    """
    print("Loading MiniLM model...")
    model = SentenceTransformer(model_variant)  # You can change this to another MiniLM variant if needed
    return model

def generate_caption_embeddings(json_file, model, output_file):
    """
    Generate MiniLM embeddings for the first 4 captions of each image.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"Found {len(data)} image-caption pairs to process.")

    results = []
    for item in tqdm(data, desc="Embedding captions"):
        image_name = item["image"]
        # captions = item["caption"][:4]  # Take only the first 4 captions
        captions = item["caption"]  # Take only the first 4 captions

        for caption in captions:
            try:
                embedding = model.encode(caption, normalize_embeddings=True).tolist()
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
    output_file = os.path.join(config["paths"]["project_root"],"data","embeddings",f"minilm_{dataset_name}_{modality}_embeddings.json")
    model_variant = config["models"]["minilm"]
    # image_folder = config["paths"]["dataset"][selected_dataset]["base_image_path"]

    # json_file = "/mnt/storage/RSystemsBenchmarking/data/datasets/coco/annotations/coco_karpathy_test.json"
    # output_file = "coco_minilm_caption_embeddings.json"

    model = load_minilm_model(model_variant)

    if modality == "text":
        generate_caption_embeddings(json_file, model, output_file)
    else:
        print("Invalid Modality")

if __name__ == "__main__":
    main()
