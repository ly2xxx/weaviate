import weaviate
from sentence_transformers import SentenceTransformer, util
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')

#V4
client = weaviate.connect_to_local()

class_schema = {
    "class": "MyImages",
    "properties": [
        {
            "name": "name",
            "dataType": ["text"]
        },
        {
            "name": "path",
            "dataType": ["text"]
        },
        {
            "name": "vector",
            "dataType": ["vector"]
        }
    ]
}

client.schema.create_class(class_schema)

images = ['cat.jpg', 'eiffel-tower-night.jpg', 'two_dogs_in_snow.jpg'] 

with client.batch() as batch:
    for img in images:
        img_path = f"resources/{img}"
        img_emb = model.encode(Image.open(img_path))
        
        data_object = {
            "name": img,
            "path": img_path,
            "vector": img_emb.tolist()
        }
        
        batch.add_data_object(data_object, "MyImages")
