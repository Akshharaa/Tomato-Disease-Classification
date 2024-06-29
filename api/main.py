from fastapi import FastAPI, File, UploadFile
from fastapi.responses import UJSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_model/m1.keras")

CLASS_NAMES = ['Bacterial_spot', 'Early_blight', 'Late_blight','Leaf_Mold','Septoria_leaf_spot','Spider_mites_Two_spotted_spider_mite','Target_Spot','YellowLeaf__Curl_Virus','mosaic_virus','healthy']

@app.get('/hello')
async def hello():
    return "Welcome"

#predict method ---post is appropriate method

def read_file_image(data) ->np.ndarray:
    image= np.array(Image.open(BytesIO(data)))
    print(image.shape)
    return image
    
@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    #read file first to get the bytes then change into tensor or numpy
    image= read_file_image(await file.read())
    #Adding dimension
    img_batch=np.expand_dims(image,0)
    
    prediction= MODEL.predict(img_batch)
    
    prediction_list = prediction.tolist()  # Convert numpy array to list
    
    predicted_class=CLASS_NAMES[np.argmax(prediction[0])]
    confidence=np.max(prediction[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        "prediction": prediction_list}
    



if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)