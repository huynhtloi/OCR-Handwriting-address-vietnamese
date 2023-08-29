import os
import base64
import logging
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from preprocess import bounding_box, preprocessing_img
from model import model, predict_results

app = Flask(__name__)
api = Api(app)
CORS(app, resources={r"/*": {"origins": "*"}}) 

class ImageUpload(Resource):
    def post(self):
        try:
            data = request.get_json()
            single = bool(request.args.get('single'))
            if "image" in data:
              base64_img = data["image"]

              # Tách phần mã base64 của ảnh
              img_data = base64_img.split(",")[1]

              # Giải mã base64 thành ảnh
              decoded_image = base64.b64decode(img_data)
              
              image_path = os.path.join("uploads", "uploaded_image.png")

              with open(image_path, "wb") as f:
                  f.write(decoded_image)

              # Đọc tấm hình ảnh
              image = cv2.imread(image_path)
              
              if not single:                
                test_img = bounding_box(image)
                test_img = np.array(test_img)
                
                act_model = model()
                act_model.load_weights(os.path.join('best_weights_epochs.hdf5'))
                
                prediction = act_model.predict(test_img)
                
                results = predict_results(prediction)

                return {"message": "Image uploaded and decoded successfully", "predictions": results}, 201
              else:
                test_img = preprocessing_img(image)
                test_img = np.array([test_img])
                
                act_model = model()
                act_model.load_weights(os.path.join('best_weights_epochs.hdf5'))
                
                prediction = act_model.predict(test_img)
                
                results = predict_results(prediction)
                
                return {"message": "Image uploaded and decoded successfully", "predictions": results}, 201
            else:
                return {"error": "Image data not found in JSON payload"}, 400
        except Exception as e:
            logging.error(e)
            return {"error": str(e)}, 500

api.add_resource(ImageUpload, '/ocr')

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
