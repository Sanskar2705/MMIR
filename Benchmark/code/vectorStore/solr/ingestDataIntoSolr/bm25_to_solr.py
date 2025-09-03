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
                 core_name: str = "Coco_bm25",
                 data_path: str = "/mnt/storage/RSystemsBenchmarking/data/datasets/coco/annotations/coco_karpathy_test.json",
                 batch_size: int = 1000):
        self.solr_url = solr_url
        self.core_name = core_name
        self.core_url = f"{solr_url}/{core_name}"
        self.data_path = data_path
        self.batch_size = batch_size

    def delete_core(self) -> None:
        try:
            print(f"Deleting existing core: {self.core_name}")
            requests.get(f"{self.solr_url}/admin/cores?action=UNLOAD&core={self.core_name}&deleteIndex=true&deleteDataDir=true&deleteInstanceDir=true")
        except Exception as e:
            print(f"Warning: Failed to delete core {self.core_name}: {e}")

    def create_core(self) -> None:
        print(f"Creating core: {self.core_name}")
        res = requests.get(f"{self.solr_url}/admin/cores?action=CREATE&name={self.core_name}&configSet=_default")
        if res.status_code != 200:
            raise RuntimeError(f"Core creation failed: {res.text}")

    def update_schema(self) -> None:
        print(f"Updating schema for core: {self.core_name}")
        headers = {"Content-Type": "application/json"}
        schema_url = f"{self.core_url}/schema"
        fields = [
            {"name": "image_path", "type": "string", "stored": True},
            {"name": "caption", "type": "text_general", "stored": True},
        ]

        for field in fields:
            res = requests.post(schema_url, headers=headers, json={"add-field": field})
            if res.status_code != 200 and "already exists" not in res.text:
                print(f"Failed to add field {field['name']}: {res.text}")

    def read_data(self) -> List[Dict]:
        print(f"Reading data from: {self.data_path}")
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def prepare_documents(self, record: Dict) -> List[Dict]:
        image_path = record["image"]
        captions = record["caption"][:4]  # Take top 4 captions
        return [{"image_path": image_path, "caption": c.strip()} for c in captions]

    def index_data(self, data: List[Dict]) -> None:
        solr = pysolr.Solr(self.core_url, always_commit=True, timeout=60)
        total = len(data)
        for i in range(0, total, self.batch_size):
            batch = data[i:i + self.batch_size]
            docs = []
            for rec in batch:
                docs.extend(self.prepare_documents(rec))  # One doc per caption
            try:
                solr.add(docs)
                print(f"Indexed batch {i // self.batch_size + 1} ({len(docs)} docs)")
            except Exception as e:
                print(f"Failed to index batch {i // self.batch_size + 1}: {e}")
            time.sleep(0.1)

    def recreate_core_and_index(self) -> None:
        self.delete_core()
        self.create_core()
        self.update_schema()
        data = self.read_data()
        self.index_data(data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This code will generate clip embeddings")

    # Load config
    config = load_config()
    selected_dataset = config["selected_config"]["dataset"]
    solr_connection_protocol = config["vector_store"]["solr"]["protocol"]
    solr_connection_host = config["vector_store"]["solr"]["host"]
    solr_connection_port = config["vector_store"]["solr"]["port"]
    solr_connection_url_path = config["vector_store"]["solr"]["url_path"]

    solr_connection_url = solr_connection_protocol+"://"+solr_connection_host+":"+solr_connection_port+solr_connection_url_path

    # Optional argument
    # parser.add_argument("--modality", "-m", help="Modality of data/embeddings", default="text")
    parser.add_argument("--dataset", "-d", help="Dataset Name. Default value from selected_config in config file.", default=selected_dataset)
    # parser.add_argument("--model", "-M", help="Model with which embeddings(if any) were generated. Default value from selected_config in config file.", default="bm25")


    args = parser.parse_args()
    # modality = args.modality
    modality = "text"
    dataset_name = args.dataset
    # model_name = args.model
    model_name = "bm25"



    if modality == "text" and model_name =="bm25":
        data_file_path = config["paths"]["dataset"][dataset_name]["ingest_annotations_path"]
        core_name = "_".join([model_name,dataset_name,modality])

        fixer = SolrCoreFixer(solr_url = solr_connection_url,data_path = data_file_path, core_name=core_name)
        fixer.recreate_core_and_index()
    else:
        print("Invalid Modality or Model")
