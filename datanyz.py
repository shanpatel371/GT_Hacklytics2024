from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import json

encoder = SentenceTransformer('all_MiniLM-L6-v2')
client = QdrantClient(path="/Users/shan/Documents/Documents - Rudra’s MacBook Pro/Gt_Hack2024/dataset.json")

with open('/Users/shan/Documents/Documents - Rudra’s MacBook Pro/Gt_Hack2024/dataset.json', 'r') as json_file:
    data = json.load(json_file)

documents = [{'hotel_name': item['hotel_name']} for item in data]

qudrant = QdrantClient(":memory:")
 
qudrant.recreate_collection(
    collection_name="hotels",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), #Vector size is defined by used model
        distance = models.DistanceType.COSINE
    )
)

qudrant.upload_records(
    collection_name="hotels",
    records=[
        models.Record(
            id=i,
            vector=encoder.encode(document['hotel_name']).tolist(),
            payload=document
        ) for i, document in enumerate(documents)
    ]
)

hits = qudrant.search(
    collection_name="hotels",
    query_vector=encoder.encode("hotel_name").tolist(),
    limit=3
)

for hit in hits:
    print(hit.payload, "score:", hit.score)