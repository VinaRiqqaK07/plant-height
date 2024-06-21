# Final Code
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from roboflow import Roboflow
import json
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="loaded more than 1 DLL from .libs:")
app = Flask(__name__)
CORS(app, resources={r"/process_image": {"origins": "http://127.0.0.1:5500"}})

def calculate_pixel_height(bbox):
    return bbox['height']

def establish_conversion_factor(reference_height, reference_pixel_height):
    return reference_height / reference_pixel_height

def process_object_detection_results(json_path, image_path, output_path):
    # Load the JSON results
    with open(json_path, 'r') as file:
        detection_results = json.load(file)

    # Extract bounding box information for the reference object and the plant
    reference_bbox = next(item for item in detection_results['predictions'] if item['class'] == 'ref_obj')
    plant_bbox = next(item for item in detection_results['predictions'] if item['class'] == 'plant')

    # Calculate pixel height of the reference object
    reference_pixel_height = calculate_pixel_height(reference_bbox)

    # Known height of the reference object in inches
    reference_height_inches = 6

    # Establish pixel-to-inch conversion factor
    conversion_factor = establish_conversion_factor(reference_height_inches, reference_pixel_height)

    # Calculate the actual height of the plant in inches
    plant_pixel_height = calculate_pixel_height(plant_bbox)
    plant_height_inches = plant_pixel_height * conversion_factor

    # Load the original image
    image = cv2.imread(image_path)

    # Draw bounding boxes and labels on the image
    xr = reference_bbox['x'] - (reference_bbox['width']/2);
    yr = reference_bbox['y'] - (reference_bbox['height']/2)
    reference_box = [

        round(xr),
        round(yr),
        round(xr + reference_bbox['width']),
        round(yr + reference_bbox['height'])
    ]

    xp = plant_bbox['x'] - (plant_bbox['width']/2);
    yp = plant_bbox['y'] - (plant_bbox['height']/2);
    plant_box = [
        round(xp),
        round(yp),
        round(xp + plant_bbox['width']),
        round(yp + plant_bbox['height'])
    ]

    # Draw reference object bounding box in green
    cv2.rectangle(image, (reference_box[0], reference_box[1]), (reference_box[2], reference_box[3]), (0, 255, 0), 2)

    # Draw plant bounding box in red
    cv2.rectangle(image, (plant_box[0], plant_box[1]), (plant_box[2], plant_box[3]), (0, 0, 255), 2)

    # Display labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_y_offset = 20
    cv2.putText(image, f"Reference Object Height: {reference_height_inches} inches", (reference_box[0], reference_box[1] - label_y_offset),
                font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(image, f"Plant Height: {plant_height_inches:.2f} inches", (plant_box[0], plant_box[1] - label_y_offset),
                font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Save the annotated image
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to {output_path}")
    return plant_height_inches, output_path

def predict_object(image_path):
    rf = Roboflow(api_key="T6ZndIelM8vVkOMaCBVn")
    project = rf.workspace("tim-4ijf0").project("plantmo")
    model = project.version(1).model
    
    # infer on a local image
    #print(model.predict("50.jpg", confidence=80, overlap=50).json())
    predictions_json = model.predict(image_path, confidence=40, overlap=30).json()
    
    output_json_path = 'detection_results.json'
    
    # Save the predictions to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(predictions_json, json_file, indent=2)
    
    print(f"Predictions saved to {output_json_path}")
    return output_json_path

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image = request.files['image']
    image_path = os.path.join('uploads', image.filename)
    output_image_path = os.path.join('uploads', f"annotated_{image.filename}")
    image.save(image_path)
    
    json_path = predict_object(image_path)
    plant_height_inches, annotated_image_path = process_object_detection_results(json_path, image_path, output_image_path)
    response = jsonify({
        "plant_height_inches": plant_height_inches,
        "annotated_image_url": f"/uploads/{os.path.basename(annotated_image_path)}"
    })
    print("Response: ", response.get_json())
    return response, 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host='0.0.0.0')