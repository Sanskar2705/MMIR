#!/bin/bash

# List of all cores
# cores=(
#   "clip_coco_image"
#   "clip_coco_text"
#   "clip_flickr_image"
#   "clip_flickr_text"
#   "flava_coco_text"
#   "flava_flickr_text"
#   "flava_coco_image"
#   "flava_flickr_image"
#   "uniir_coco_text"
#   "uniir_flickr_text"
#   "uniir_coco_image"
#   "uniir_flickr_image"
#   "uniir_coco_joint-image-text"
#   "uniir_flickr_joint-image-text"
# )

cores=(
  "minilm_flickr_text"
  "minilm_coco_text")

# Loop over each core
for core in "${cores[@]}"; do
  echo "=== Processing core: $core ==="

  # 1. Check if knn_vector_384 is present
  echo ">>> Checking field type knn_vector_384 for $core ..."
  curl --silent --noproxy localhost "http://localhost:8983/solr/$core/schema/fieldtypes/knn_vector_384" | jq .

  # 2. Replace embedding_vector field
  echo ">>> Replacing embedding_vector field with knn_vector_384 in $core ..."
  curl --noproxy localhost -X POST -H 'Content-type:application/json' \
    "http://localhost:8983/solr/$core/schema" \
    -d '{"replace-field":{"name":"embedding_vector","type":"knn_vector_384","indexed":true,"stored":true}}'

  echo -e "\n"
done
