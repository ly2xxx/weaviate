import streamlit as st
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.config import Property, DataType

st.title("Add Images to Weaviate")

# Initialize SentenceTransformer model
model = SentenceTransformer('clip-ViT-B-32')

# Connect to the local Weaviate instance
client = weaviate.connect_to_local()

# Clean up existing collection (if any)
COLLECTION_NAME = "MyImagesLocal"
client.collections.delete(COLLECTION_NAME)

# Create a new collection with 'name' and 'path' properties
client.collections.create(
    COLLECTION_NAME,
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="path", data_type=DataType.TEXT),
    ]
)

# Get the collection
images_collection = client.collections.get(COLLECTION_NAME)

# Input: Image folder path
img_folder = st.sidebar.text_input("Image folder path:")

# Button to add images
if st.sidebar.button("Add images"):
    img_files = Path(img_folder).glob("*")
    with images_collection.batch.dynamic() as batch:
        for img_path in img_files:
            img_name = os.path.basename(img_path)
            img_emb = model.encode(img_path)

            # Add the image object to Weaviate
            batch.add_object(properties={
                "name": img_name,
                "path": str(img_path)
            }, vector=img_emb.tolist())

    st.success("Images added to Weaviate!")

# Close the Weaviate client
client.close()
