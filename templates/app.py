from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Print the device being used

# Define the CNN model class
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 128)
        self.fc2 = torch.nn.Linear(128, 1)  # For binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Set the upload folder
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the model
model = CNNModel()
model.load_state_dict(torch.load('brain_tumor_model.pth', map_location=device))  # Load the model's state dict
model.to(device)  # Move the model to the device
model.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize the image
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    # Save the uploaded file
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Preprocess the image
    img = Image.open(img_path).convert('RGB')  # Convert to RGB if it's grayscale
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make a prediction
    with torch.no_grad():  # Disable gradient calculation
        prediction = model(img_tensor)
    predicted_class = torch.round(prediction).item()  # Get the predicted class (0 or 1)

    result = "Tumor" if predicted_class == 1 else "No Tumor"  # Assuming class 1 is Tumor

    return render_template('index.html', prediction=result, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True, port=5500)
