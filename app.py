from flask import Flask, render_template, request
import joblib
import numpy as np
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image  # Import PIL for image handling

app = Flask(__name__)

# Set device for PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pretrained dementia model (Random Forest)
dementia_model_filename = r"C:\Users\DELL\Desktop\neuroimaging\rf_dementia_model.pkl"
rf_dementia_model = joblib.load(dementia_model_filename)  # Load the dementia model

# Load the brain tumor model (CNN)
brain_tumor_model_path = r'C:/Users/DELL/Desktop/neuroimaging/brain_tumor_model.pth'

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply convolution and pooling
        x = x.view(-1, 32 * 32 * 32)  # Flatten the output
        x = torch.relu(self.fc1(x))  # First fully connected layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid output for binary classification
        return x

# Initialize the brain tumor model and load the weights
model = CNNModel()
model.load_state_dict(torch.load(brain_tumor_model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Load the schizophrenia model (Random Forest)
schizophrenia_model_filename = r"C:\Users\DELL\Desktop\neuroimaging\rf_schizophrenia_model.pkl"
rf_schizophrenia_model = joblib.load(schizophrenia_model_filename)  # Load schizophrenia model

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for dementia prediction
@app.route('/predict_dementia', methods=['POST'])
def predict_dementia():
    # Retrieve input data from the form
    age = request.form['age']
    educ = request.form['educ']
    ses = request.form['ses']
    mmse = request.form['mmse']
    cdr = request.form['cdr']
    etiv = request.form['etiv']
    nwbv = request.form['nwbv']
    asf = request.form['asf']

    # Prepare input data for the dementia model
    input_data = np.array([[float(age), float(educ), float(ses), float(mmse), float(cdr), float(etiv), float(nwbv), float(asf)]])

    # Predict using the pre-trained dementia model
    prediction = rf_dementia_model.predict(input_data)

    # Map prediction back to the label (0 = Non-demented, 1 = Demented)
    dementia_prediction = "Demented" if prediction[0] == 1 else "Non-demented"

    # Render the home page with the prediction result
    return render_template('index.html', dementia_prediction=dementia_prediction)

# Route for brain tumor prediction
@app.route('/predict_brain_tumor', methods=['POST'])
def predict_brain_tumor():
    try:
        # Handle image upload and preprocessing
        file = request.files['image']
        if not file:
            raise ValueError("No file uploaded")

        # Save the uploaded image temporarily
        image_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)  # Ensure temp directory exists
        file.save(image_path)

        # Load the image and preprocess it for the CNN model
        transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Resize to match model input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        image = Image.open(image_path).convert('RGB')  # Open the image in RGB mode
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device

        # Predict using the brain tumor model
        with torch.no_grad():
            output = model(image)
            prediction = torch.round(output).item()  # Convert to binary prediction

        # Map prediction back to the label (0 = No tumor, 1 = Tumor)
        tumor_prediction = "Tumor" if prediction == 1 else "No Tumor"

        # Return the result to the template
        return render_template('index.html', prediction=tumor_prediction, img_path=image_path)

    except Exception as e:
        return render_template('index.html', prediction=None, img_path=None, error=str(e))
    # Route for schizophrenia prediction
@app.route('/predict_schizophrenia', methods=['POST'])
def predict_schizophrenia():
    # Retrieve input data from the form for schizophrenia prediction
    age = float(request.form['age'])
    fatigue = float(request.form['fatigue'])
    slowing = float(request.form['slowing'])
    pain = float(request.form['pain'])
    hygiene = float(request.form['hygiene'])  
    movement = float(request.form['movement'])

    # Prepare input data for schizophrenia model
    input_features = np.array([[age, fatigue, slowing, pain, hygiene, movement]])

    # Predict using the pre-trained schizophrenia model
    schizophrenia_prediction = rf_schizophrenia_model.predict(input_features)

    # Mapping the numerical prediction to the corresponding proneness level
    proneness_map = {
        0: "Elevated Proneness",
        1: "Moderate Proneness",
        2: "High Proneness"
    }

    # Determine the proneness level based on the model's prediction
    schizophrenia_prediction_text = proneness_map.get(schizophrenia_prediction[0], "Unknown Proneness")

    # Render the home page with the prediction result
    return render_template('index.html', schizophrenia_prediction=schizophrenia_prediction_text)

# Route for About Us page
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

# Route for Services page
@app.route('/services')
def services():
    return render_template('services.html')

# Route for Reviews page
@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

if __name__ == "__main__":  # Corrected line
    app.run(debug=True, port=5500)

