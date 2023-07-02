from fastapi import FastAPI, HTTPException
from fastapi import UploadFile
import uvicorn
import numpy as np
import json
import lib

app = FastAPI()

@app.post("/process-slide")
async def process_slide_api(json_obj: dict):
    try:
        response = process_slide(json_obj)

        # Return the response as JSON
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  
