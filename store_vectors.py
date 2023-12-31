# import thu vien
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

from PIL import Image
import numpy as np


def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs,
                          outputs=vgg16_model.get_layer("fc1").output)
    return extract_model


def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_vector(model, image_path):
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    vector = model.predict(img_tensor)[0]
    # Chuan hoa vector = chia chia L2 norm (tu google search)
    vector = vector / np.linalg.norm(vector)
    return vector


data_folder = "dataset"
model = get_extract_model()

vectors = []
paths = []

for image_path in os.listdir(data_folder)[:50]:
    image_path_full = os.path.join(data_folder, image_path)
    image_vector = extract_vector(model, image_path_full)
    vectors.append(image_vector)
    paths.append(image_path_full)
