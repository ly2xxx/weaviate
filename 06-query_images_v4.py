from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate
import json

model = SentenceTransformer('clip-ViT-B-32')

#V4
client = weaviate.connect_to_local()

Image.open("resources/eiffel-tower-day.jpg").show(title="Query for...")

embedding=model.encode(Image.open("resources/eiffel-tower-day.jpg"))

# Convert the numpy.ndarray to a list
embedding_list = embedding.tolist()

# Get the collection
images = client.collections.get("MyImages")

response = images.query.near_vector(
    near_vector=embedding_list,
    return_properties=["name", "path"],
    limit=1
)

first_obj = response.objects[0]
print(first_obj)
path_value = first_obj["properties"]["path"]
Image.open(path_value).show(title="Response")
# print(json.dumps(response, indent=4))

# first_obj=response["data"]["Get"]["MyImages"][0]
# Image.open(first_obj["path"]).show(title="Reponse")

client.close()