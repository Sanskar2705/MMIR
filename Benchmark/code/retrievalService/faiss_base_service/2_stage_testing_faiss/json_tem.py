import json
import argparse
from pathlib import Path

def convert_json(input_path: Path, output_path: Path):
    with open(input_path, "r") as f:
        data = json.load(f)

    detailed_results = data.get("detailed_results", [])

    output = {"queries": []}

    for entry in detailed_results:
        query_obj = {
            "query": entry.get("query"),
            "ground_truth": entry.get("ground_truth"),
            "stage1_hits": []
        }

        top_k = entry.get("top_k_retrieved", [])

        for rank, image_path in enumerate(top_k, start=1):
            query_obj["stage1_hits"].append({
                "image_path": image_path,
                "score": None,   # score not available in original JSON
                "rank": rank
            })

        output["queries"].append(query_obj)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Converted JSON written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert detailed_results JSON into query-stage1 format"
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path"
    )

    args = parser.parse_args()

    output_path = (
        args.output
        if args.output is not None
        else args.input_json.with_name(args.input_json.stem + "_converted.json")
    )

    convert_json(args.input_json, output_path)


if __name__ == "__main__":
    main()
