#!/usr/bin/env python3

import json
import os
import requests
import pysolr
import time
from typing import Dict, List

from contextlib import asynccontextmanager
import sys
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config
import argparse

# Remove proxy settings
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

class SolrCoreFixer:

    def __init__(self,
                 solr_url: str = "http://localhost:8983/solr",
                 core_name: str = "Coco_caption_test",  # updated core name
                 data_file: str = "/mnt/storage/sanskar/samy/Benchmark/Benchmark/code/embedding/coco_clip_caption_embeddings.json",  # updated path
                 batch_size: int = 1000):
        self.solr_url = solr_url
        self.core_name = core_name
        self.core_url = f"{solr_url}/{core_name}"
        self.data_file = data_file
        self.batch_size = batch_size
        self.embedding_vector_dim = 0

    def delete_and_recreate_core(self) -> bool:
        try:
            print(f"Deleting core: {self.core_name}")
            requests.get(f"{self.solr_url}/admin/cores", params={
                "action": "UNLOAD",
                "core": self.core_name,
                "deleteIndex": "true",
                "deleteDataDir": "true",
                "deleteInstanceDir": "true"
            })
            time.sleep(5)
            print(f"Recreating core: {self.core_name}")
            create_response = requests.get(f"{self.solr_url}/admin/cores", params={
                "action": "CREATE",
                "name": self.core_name,
                "configSet": "_default"
            })
            time.sleep(5)
            return create_response.status_code == 200
        except Exception as e:
            print(f"Error recreating core:")
            return False

    def setup_fresh_schema(self) -> bool:
        try:
            types_resp = requests.get(f"{self.core_url}/schema/fieldtypes", params={"wt": "json"}).json()
            field_types = {ft['name'] for ft in types_resp.get('fieldTypes', [])}

            if ("knn_vector_"+f"{self.embedding_vector_dim}") not in field_types:
                field_type_payload = {
                    "add-field-type": {
                        "name": "knn_vector_"+f"{self.embedding_vector_dim}",
                        "class": "solr.DenseVectorField",
                        "vectorDimension": self.embedding_vector_dim,
                        "similarityFunction": "cosine"
                    }
                }
                resp = requests.post(f"{self.core_url}/schema", json=field_type_payload, params={"wt": "json"})
                if resp.status_code != 200:
                    # print(f"Failed to add vector field type: {resp.text}")
                    print(f"Failed to add vector field type: ")

                    return False

            fields_resp = requests.get(f"{self.core_url}/schema/fields", params={"wt": "json"}).json()
            existing_fields = {f['name'] for f in fields_resp.get('fields', [])}

            field_defs = [
                {"name": "image_name", "type": "string", "stored": True, "indexed": True},
                {"name": "caption", "type": "string", "stored": True, "indexed": True},
                {"name": "embedding_vector", "type": "knn_vector_"+f"{self.embedding_vector_dim}", "stored": True, "indexed": True}
            ]

            for field in field_defs:
                if field['name'] not in existing_fields:
                    resp = requests.post(f"{self.core_url}/schema", json={"add-field": field}, params={"wt": "json"})
                    if resp.status_code != 200:
                        # print(f"Error adding field {field['name']}: {resp.text}")
                        print(f"Error adding field {field['name']}: ")
                        return False

            time.sleep(3)
            return True
        except Exception as e:
            print(f"Error setting up schema: ")
            return False

    def load_data(self) -> List[Dict]:
        with open(self.data_file, 'r') as f:
            return json.load(f)

    def prepare_document(self, record: Dict) -> Dict:
        return {
            "image_path": record["image_name"],
            "caption": record["caption"],
            "embedding_vector": record["embedding"]
        }

    def index_data(self, data: List[Dict]) -> None:
        solr = pysolr.Solr(self.core_url, always_commit=True, timeout=60)
        total = len(data)
        for i in range(0, total, self.batch_size):
            batch = data[i:i + self.batch_size]
            docs = [self.prepare_document(rec) for rec in batch]
            try:
                solr.add(docs)
                print(f"Indexed batch {i // self.batch_size + 1} ({len(docs)} docs)")
            except Exception as e:
                print(f"Failed to index batch {i // self.batch_size + 1}: ")
            time.sleep(0.1)

    def get_vector_embedding_dimension(self,data):
        if len(data) <= 0:
            print("Error : get_vector_embedding_dimension : data not found")
            return 0
        
        data_ele = data[0]
        if "embedding" in data_ele:
            return len(data_ele["embedding"])
        else:
            print("Error : get_vector_embedding_dimension : embedding not found")
            return 0

    def run(self) -> None:

        try:
            data = self.load_data()
            self.embedding_vector_dim = self.get_vector_embedding_dimension(data)
        except Exception as e:
            print(f"Failed to load data: ")
            return
        
        print(f"Embedding dimension is : {self.embedding_vector_dim}")
        # _ = input()

        print("Recreating Solr core and indexing data")
        if not self.delete_and_recreate_core():
            print("Failed to recreate core.")
            return
        if not self.setup_fresh_schema():
            print("Failed to set up schema.")
            return
        
        self.index_data(data)
        print("Indexing complete.")

def main():

    parser = argparse.ArgumentParser(description="This code will generate clip embeddings")

    # Load config
    config = load_config()
    selected_dataset = config["selected_config"]["dataset"]
    selected_model = config["selected_config"]["model"]
    solr_connection_protocol = config["vector_store"]["solr"]["protocol"]
    solr_connection_host = config["vector_store"]["solr"]["host"]
    solr_connection_port = config["vector_store"]["solr"]["port"]
    solr_connection_url_path = config["vector_store"]["solr"]["url_path"]

    solr_connection_url = solr_connection_protocol+"://"+solr_connection_host+":"+solr_connection_port+solr_connection_url_path

    # Optional argument
    parser.add_argument("--modality", "-m", help="Modality of data/embeddings", default="text")
    parser.add_argument("--dataset", "-d", help="Dataset Name. Default value from selected_config in config file.", default=selected_dataset)
    parser.add_argument("--model", "-M", help="Model with which embeddings(if any) were generated. Default value from selected_config in config file.", default=selected_model)


    args = parser.parse_args()
    modality = args.modality
    # modality = "text"
    dataset_name = args.dataset
    model_name = args.model


    print("This script will delete and recreate the Solr core to fix schema and index vector data.")
    confirm = input("This will DELETE existing data. Continue? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Aborted.")
        return
    
    data_file_path = os.path.join(config["paths"]["project_root"],"data","embeddings",f"{model_name}_{dataset_name}_{modality}_embeddings.json")
    core_name = "_".join([model_name,dataset_name,modality])
    
    fixer = SolrCoreFixer(solr_url = solr_connection_url,data_file = data_file_path, core_name=core_name)
    fixer.run()

if __name__ == "__main__":
    main()
