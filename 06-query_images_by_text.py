from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate

model = SentenceTransformer('clip-ViT-B-32')

#V4
client = weaviate.connect_to_local()

# Encode text prompt with CLIP
embedding= model.encode("Cat on a desk") 

# import clip

# model, preprocess = clip.load("ViT-B/32") 

# text = "Paris at night"
# text_embedding = model.encode_text(clip.tokenize(text)) 

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