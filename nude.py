from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import os 
from fastapi.responses import JSONResponse
import io
from torchvision import transforms
import torch
import torchvision

weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT

loaded_model = torchvision.models.efficientnet_b0(weights = weight)

from torch import nn
class_names = ['Not-Safe', 'safe', 'sexy']


loaded_model.classifier = nn.Linear(in_features=1280 ,  out_features = len(class_names))

loaded_model.load_state_dict(torch.load(f='nudex.pth', weights_only=True,  map_location=torch.device('cpu')))
loaded_model.eval()

transforms = weight.transforms()


app = FastAPI(title='Single eye Image classification')

@app.get('/')
def get():
    return {'Message ' : 'Single eye Image classification'}


@app.post("/predict-image")
async def upload_image(file: UploadFile = File(...)):

    try : 
            
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
                return JSONResponse(
                    content={"error": "Only JPEG and PNG images are allowed."},
                    status_code=400
                )
            
            # Read file content

        contents = await file.read()
            
            # Load image using PIL for further processing (optional)
        image = Image.open(io.BytesIO(contents))
        image = image.convert('RGB')

        images = transforms(image).unsqueeze(dim=0)


        with torch.inference_mode():
            y_logit = loaded_model(images)
            y_label = y_logit.argmax(dim=1)



        return {'Predicted model' : class_names[y_label]}

        
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"},
            status_code=500
        )
