from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate

#run locally hosted model
model = SentenceTransformer('./models/clip-ViT-B-32/')
#replace with model = SentenceTransformer('clip-ViT-B-32') to run inference from huggingface

#V4
client = weaviate.connect_to_local()

# Encode text prompt with CLIP
embedding= model.encode("Girl eating ice cream") 

# import clip

# model, preprocess = clip.load("ViT-B/32") 

# text = "Paris at night"
# text_embedding = model.encode_text(clip.tokenize(text)) 

# Convert the numpy.ndarray to a list
embedding_list = embedding.tolist()

# Get the collection
images = client.collections.get("MyImagesLocal")

response = images.query.near_vector(
    near_vector=embedding_list,
    return_properties=["name", "path"],
    limit=3
)

def display_picture(picture_obj):
    print(picture_obj)
    path_value = picture_obj.properties["path"]
    Image.open(path_value).show(title="Response")

display_picture(response.objects[0])
display_picture(response.objects[1])
display_picture(response.objects[2])

client.close()