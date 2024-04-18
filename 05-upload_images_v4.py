import weaviate
from weaviate.classes.config import Property, DataType
from sentence_transformers import SentenceTransformer, util
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')

#V4
client = weaviate.connect_to_local()
#https://weaviate.io/developers/weaviate/manage-data/collections
# clean up
COLLECTION_NAME = "MyImagesV4"
client.collections.delete(COLLECTION_NAME)

client.collections.create(
    COLLECTION_NAME, 
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="path", data_type=DataType.TEXT),
    ]
)

images = ['cat.jpg', 'eiffel-tower-day.jpg', 'two_dogs_in_snow.jpg'] 

images_collection = client.collections.get(COLLECTION_NAME)

for imgurl in images:
    img_path = f"resources/{imgurl}"
    img_emb = model.encode(Image.open(img_path))
    print(img_emb)

    uuid = images_collection.data.insert(
        properties={
            "name": imgurl,
            "path": img_path,
        },
        vector=img_emb.tolist()
    )

    print(uuid)    

client.close()
