from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import os 
from fastapi.responses import JSONResponse
import io
from torchvision import transforms
import torch
import torchvision

import streamlit as st 

weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT

loaded_model = torchvision.models.efficientnet_b0(weights = weight)

from torch import nn
class_names = ['nude', 'safe', 'sexy']


loaded_model.classifier = nn.Linear(in_features=1280 ,  out_features = len(class_names))

loaded_model.load_state_dict(torch.load(f='nudex.pth', weights_only=True,  map_location=torch.device('cpu')))
loaded_model.eval()

transforms = weight.transforms()


st.title('Nudity Detection Model Prediction ')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    # Load image using PIL
    image = Image.open(uploaded_file)

    # Display the image in Streamlit
    #st.image(image, caption='Uploaded Image', use_column_width=True)

    Image = transforms(image.convert('RGB')).unsqueeze(dim=0)

    with torch.inference_mode():
        y_logit = loaded_model(Image)
        y_label = y_logit.argmax(dim=1)


    if st.button('Predict the class of the image '):
        st.success(f'Predicted Class of the Image : {class_names[y_label]}')
