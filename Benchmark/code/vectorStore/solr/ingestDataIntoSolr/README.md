### NOTES

1. Between indexing data with different embedding dimensions, there might be an error "Failed to index" a core in solr, in such a scenario  : 
Run the following on terminal, replace CORE_NAME with the name of the core trying to add data to

```
curl -X POST http://localhost:8983/solr/CORE_NAME/schema \
  -H 'Content-Type: application/json' \ 
  -d '{"delete-field": {"name": "embedding_vector"}}' --noproxy localhost 
```


Then re-run the ingestion code

2. To view schema of a core , set the given CORE_NAME
```
curl http://localhost:8983/solr/CORE_NAME/schema --noproxy localhost
```