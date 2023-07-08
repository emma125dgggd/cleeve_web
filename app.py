from fastapi import FastAPI, HTTPException, File
from fastapi import UploadFile
import base64
import uvicorn
import numpy as np
import json
from lib import *

app = FastAPI()

@app.post("/process-slide")
async def process_slide_api(image: UploadFile = File(...)):
    try:
        file_content = await image.read()
        encoded_image = base64.b64encode(file_content).decode("utf-8")

        # Create a JSON object with the base64-encoded image data
        json_obj = {"image_data": encoded_image,
                    "filename" : image.filename
                    }
        json_data = json.dumps(json_obj)
        count, image = process_slide(json_data)

        # Return the response as JSON
        return count, image

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      

