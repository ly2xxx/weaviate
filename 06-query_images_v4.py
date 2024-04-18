from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate

model = SentenceTransformer('clip-ViT-B-32')

#V4
client = weaviate.connect_to_local()

query_picture = "resources/eiffel-tower-night.jpg"
Image.open(query_picture).show(title="Query for...")

embedding=model.encode(Image.open(query_picture))

# Convert the numpy.ndarray to a list
embedding_list = embedding.tolist()

# Get the collection
images = client.collections.get("MyImagesV4")

response = images.query.near_vector(
    near_vector=embedding_list,
    return_properties=["name", "path"],
    limit=1
)

first_obj = response.objects[0]
print(first_obj)
path_value = first_obj.properties["path"]
Image.open(path_value).show(title="Response")

client.close()