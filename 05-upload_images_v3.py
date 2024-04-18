from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate
import json

images=['cat.jpg','eiffel-tower-night.jpg','two_dogs_in_snow.jpg']

model = SentenceTransformer('clip-ViT-B-32')

#V3
client = weaviate.Client(
  url="http://localhost:8080", 
)

schema = {
    "class": "MyImages",
}

client.schema.delete_class("MyImages")


client.schema.create_class(schema)

with client.batch as batch:
  batch.batch_size=10    
  for imgurl in images:
      path=f"resources/{imgurl}"
      data_obj = {
        "name": imgurl,
        "path": path
      }
      img_emb = model.encode(Image.open(path))
      print(img_emb)
      batch.add_data_object(data_obj,"MyImages", vector=img_emb)

# clean up
#client.schema.delete_class("MyImages")

