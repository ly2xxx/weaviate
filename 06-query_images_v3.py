from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate
import json

model = SentenceTransformer('clip-ViT-B-32')

#V3
client = weaviate.Client(
  url="http://localhost:8080", 
)

Image.open("resources/eiffel-tower-day.jpg").show(title="Query for...")

embedding=model.encode(Image.open("resources/eiffel-tower-day.jpg"))

response = (
    client.query
    .get("MyImages", ["name", "path"])
    .with_near_vector({"vector":embedding})
    .with_limit(1)
    .do()
)
print(json.dumps(response, indent=4))

first_obj=response["data"]["Get"]["MyImages"][0]
Image.open(first_obj["path"]).show(title="Reponse")
