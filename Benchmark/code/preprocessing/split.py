

import sys, json, argparse
from pathlib import Path

# Allow: from Benchmark.config.config_utils import load_config
sys.path.append("/mnt/storage/RSystemsBenchmarking/gitProject")
from Benchmark.config.config_utils import load_config

def main():

    parser = argparse.ArgumentParser(description="This code will split the dataset (annotations) into ingest and test split")

    # Load config
    config = load_config()
    selected_dataset = config["selected_config"]["dataset"]

    # Optional argument
    parser.add_argument("--dataset", "-d", help="Dataset to be split", default=selected_dataset)

    args = parser.parse_args()
    dataset_name = args.dataset

    # Input file
    input_file = Path(config["paths"]["dataset"][dataset_name]["annotations_path"])
    
    dataset_root_path = input_file.parent.parent

    # Output files
    preprocessed_folder = dataset_root_path / "processed_mapping"
    test_output = input_file.parent / (dataset_name+"_ingest.json")
    query_output = input_file.parent / (dataset_name+"_query.json")

    preprocessed_folder.mkdir(parents=True, exist_ok=True)

    # Load original dataset
    with open(input_file, "r") as f:
        data = json.load(f)

    # Prepare test and query data
    test_data = []
    query_data = []

    for item in data:
        if len(item["caption"]) >= 5:
            test_data.append({
                "image": item["image"],
                "caption": item["caption"][:4]
            })
            query_data.append({
                "image": item["image"],
                "caption": item["caption"][4]
            })

    # Save to output files
    with open(test_output, "w") as f:
        json.dump(test_data, f, indent=2)

    with open(query_output, "w") as f:
        json.dump(query_data, f, indent=2)

    print(f" Created:\n- {test_output} with {len(test_data)} items\n- {query_output} with {len(query_data)} items")

if __name__ == "__main__":
    main()
