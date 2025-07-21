'''from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import face_recognition
import warnings
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
SEQUENCE_LENGTH = 10
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

warnings.filterwarnings("ignore")

# ----------------- Model Definition -----------------

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# ----------------- Dataset Definition -----------------

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        for i, frame in enumerate(self.frame_extract(video_path)):
            if i < first_frame or (i - first_frame) % a != 0:
                continue

            faces = face_recognition.face_locations(frame)
            if faces:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            else:
                h, w, _ = frame.shape
                min_dim = min(h, w)
                start_h = (h - min_dim) // 2
                start_w = (w - min_dim) // 2
                frame = frame[start_h:start_h+min_dim, start_w:start_w+min_dim]

            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break

        if len(frames) < self.count:
            return None

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# ----------------- Prediction Logic -----------------

def predict(model, img):
    if img is None:
        return None
    sm = nn.Softmax()
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

# ----------------- Routes -----------------

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("No file part in the request.")
        return render_template('result.html', filename='None', prediction='No file uploaded.', confidence='')

    file = request.files['file']
    if file.filename == '':
        print("No file selected.")
        return render_template('result.html', filename='None', prediction='No file selected.', confidence='')

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    print(f"File saved at: {video_path}")

    # Check if file is saved
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found after saving!")
        return render_template('result.html', filename=file.filename, prediction='Error in file saving', confidence='')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # Preparing dataset
    dataset = validation_dataset([video_path], sequence_length=SEQUENCE_LENGTH, transform=transform)

    # Load the model
    model = Model(num_classes=2)
    try:
        model.load_state_dict(torch.load('deepfake_models/deepfake_model.pth', map_location=torch.device('cpu')))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return render_template('result.html', filename=file.filename, prediction='Error loading model', confidence='')

    model.eval()

    video_tensor = dataset[0]
    if video_tensor is None:
        print("Error: Video tensor is None.")
        return render_template('result.html', filename=file.filename, prediction='Insufficient frames.', confidence='')

    prediction = predict(model, video_tensor)

    if prediction is None:
        print("Prediction returned None.")
        return render_template('result.html', filename=file.filename, prediction='No prediction made', confidence='')

    label = "REAL" if prediction[0] == 1 else "FAKE"
    confidence = f"{prediction[1]:.2f}"

    print(f"Prediction: {label}, Confidence: {confidence}%")

    return render_template('result.html', filename=file.filename, prediction=label, confidence=confidence)

# ----------------- Run App -----------------

if __name__ == '__main__':
    app.run(debug=True)

'''
from flask import Flask, render_template, request
import os
import torch
from torch import nn
from torchvision import models, transforms
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset
import face_recognition
import warnings
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Constants (same as test.py)
SEQUENCE_LENGTH = 10
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MODEL_PATH = 'deepfake_models/deepfake_model.pth'  # adjust path as needed

# Model Definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Dataset Definition
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=SEQUENCE_LENGTH, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        for i, frame in enumerate(self.frame_extract(video_path)):
            if i < first_frame or (i - first_frame) % a != 0:
                continue

            if frame is None:
                continue

            faces = face_recognition.face_locations(frame)
            if faces:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            else:
                h, w, _ = frame.shape
                min_dim = min(h, w)
                start_h = (h - min_dim) // 2
                start_w = (w - min_dim) // 2
                frame = frame[start_h:start_h + min_dim, start_w:start_w + min_dim]

            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break

        if len(frames) < self.count:
            return None

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# Prediction function
def predict_vid(model, video_tensor):
    if video_tensor is None:
        return None
    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        fmap, logits = model(video_tensor)
        probs = sm(logits)
        conf, pred = torch.max(probs, 1)
        confidence = conf.item() * 100
        prediction = pred.item()
    return prediction, confidence

# Routes
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', filename='None', prediction='No file uploaded', confidence='')

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', filename='None', prediction='No file selected', confidence='')

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    dataset = ValidationDataset([video_path], sequence_length=SEQUENCE_LENGTH, transform=transform)

    model = Model(num_classes=2)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    except Exception as e:
        return render_template('result.html', filename=filename, prediction=f'Error loading model: {str(e)}', confidence='')

    model.eval()

    video_tensor = dataset[0]
    if video_tensor is None:
        return render_template('result.html', filename=filename, prediction='Insufficient frames in video.', confidence='')

    result = predict_vid(model, video_tensor)
    if result is None:
        return render_template('result.html', filename=filename, prediction='Prediction failed.', confidence='')

    label = "REAL" if result[0] == 1 else "FAKE"
    confidence = f"{result[1]:.2f}"

    return render_template('result.html', filename=filename, prediction=label, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)


