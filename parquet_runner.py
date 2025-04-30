import pyarrow.parquet as pq
pfile = pq.read_table("/workspace/LLM/train-00000-of-00001-6ef3991c06080e14.parquet")
print("Column names: {}".format(pfile.column_names))
print("Schema: {}".format(pfile.schema))