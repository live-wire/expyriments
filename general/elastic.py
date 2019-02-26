# Trying out the low level elasticsearch client
# pip install elasticsearch

from datetime import datetime
from elasticsearch import Elasticsearch


es = Elasticsearch()
a = es.indices.create(index='try-index', ignore=400)
b = es.index(index="try-index", doc_type="test-type", id=42, body={"any": "data", "timestamp": datetime.now(), 'lol': 'search me pls'})
print(b)
