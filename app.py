import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']

    # Save the uploaded image temporarily
    temp_path = 'temp.jpg'
    image_file.save(temp_path)

    # Preprocess the image
    img = image.load_img(temp_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)

    # Load the model
    model = tf.keras.models.load_model('model.hdf5')

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class = int(predicted_class)

    # Return the predicted class to the client
    if predictions == 0:
        prediction_result = "Get that mole checkout"
    else:
        prediction_result = "you are safe"

    # Delete the temporary image file
    os.remove(temp_path)

    return render_template('result.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
