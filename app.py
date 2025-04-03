from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("models/hand_scan_model.h5")

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Clean" if prediction[0][0] < 0.5 else "Dirty"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = "static/" + file.filename
        file.save(file_path)

        result = predict_image(file_path)
        return render_template("index.html", result=result, file_path=file_path)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
