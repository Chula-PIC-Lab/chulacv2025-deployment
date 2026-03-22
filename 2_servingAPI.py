## Download model
## wget https://piclab.ai/classes/cv2023/chestxray.onnx

## Start API
## mamba activate cv2026
## uvicorn servingAPI:app --host 0.0.0.0 --port 8500

import cv2
import onnxruntime as rt
import numpy as np
from fastapi import FastAPI, File
####
sessOptions = rt.SessionOptions()
sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 
chestxrayModel = rt.InferenceSession('chestxray.onnx', sessOptions, providers=[ 'CUDAExecutionProvider', 'CPUExecutionProvider'])
####
app = FastAPI()

def decodeByte2Numpy(inputImage):
    outputImage = np.frombuffer(inputImage, np.uint8)
    outputImage = cv2.imdecode(outputImage, cv2.IMREAD_COLOR)
    return outputImage

@app.post('/chestxray')
def chest_xray_api(image: bytes = File(...)):
    try:
        inputImage = decodeByte2Numpy(image)
        inputImage = cv2.resize(cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB), (224,224))
        inputTensor = ((inputImage / 255) - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        inputTensor = inputTensor.transpose(2,0,1)[np.newaxis].astype(np.float32)

        outputTensor = chestxrayModel.run([], {'input': inputTensor})[0]

        outputDict = {'label': int(np.argmax(outputTensor)) }
        return outputDict    
    except Exception as e:
        print(e)
        return {'status':'INVALID_IMAGE_FILE'}