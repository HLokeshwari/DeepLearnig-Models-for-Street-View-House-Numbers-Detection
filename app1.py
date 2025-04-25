import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from models import MulticlassClassifier, CNNModel, get_alexnet, get_vgg16, get_rnn, get_lstm, get_autoencoder, \
    get_resnet, compare_optimizers
from preprocessing import load_data, preprocess_image
import pytesseract
import easyocr
from PIL import Image
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Load dataset
train_loader, test_loader = load_data()

# Initialize and load trained models
models_dict = {
    "multiclass": MulticlassClassifier(num_classes=10).to(device),
    "cnn": CNNModel(num_classes=10).to(device),
    "alexnet": get_alexnet(num_classes=10).to(device),
    "vgg16": get_vgg16(num_classes=10).to(device),
    "rnn": get_rnn(num_classes=10).to(device),
    "lstm": get_lstm(num_classes=10).to(device),
    "autoencoder": get_autoencoder().to(device),
    "resnet": get_resnet(num_classes=10).to(device)
}

# Load model weights if they exist
for model_name, model in models_dict.items():
    model_path = f"models/{model_name}_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model: {model_name}")
    else:
        print(f"Warning: Model file '{model_path}' not found!")

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize EasyOCR reader
easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def preprocess_for_ocr(image):
    """Preprocess the image for OCR with multiple attempts."""
    # Convert to grayscale
    gray = np.array(image.convert("L"))
    # Resize to make digits larger (6x)
    gray = cv2.resize(gray, (gray.shape[1] * 6, gray.shape[0] * 6), interpolation=cv2.INTER_CUBIC)

    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Attempt 1: Gaussian blur + adaptive thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 5)

    # Morphological operations to clean up
    kernel = np.ones((7, 7), np.uint8)
    thresh1 = cv2.dilate(thresh1, kernel, iterations=2)
    thresh1 = cv2.erode(thresh1, kernel, iterations=1)

    # Attempt 2: Otsu thresholding with Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh2 = cv2.dilate(thresh2, kernel, iterations=1)

    # Convert thresholded images to base64 for display
    def to_base64(img):
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')

    thresh1_base64 = to_base64(thresh1)
    thresh2_base64 = to_base64(thresh2)

    return thresh1, thresh2, thresh1_base64, thresh2_base64

def extract_number(image):
    """Extract number from image using OCR with multiple attempts."""
    thresh1, thresh2, thresh1_base64, thresh2_base64 = preprocess_for_ocr(image)

    # Attempt 1: OCR on adaptive thresholded image (Tesseract)
    text1 = pytesseract.image_to_string(thresh1, config='--psm 10 digits -c tessedit_char_whitelist=0123456789')
    digits1 = "".join(filter(str.isdigit, text1)).strip()

    # Attempt 2: OCR on Otsu thresholded image (Tesseract)
    text2 = pytesseract.image_to_string(thresh2, config='--psm 6 digits -c tessedit_char_whitelist=0123456789')
    digits2 = "".join(filter(str.isdigit, text2)).strip()

    # Attempt 3: OCR with tighter bounding box (Tesseract)
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits3 = ""
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        digit_roi = thresh1[y:y+h, x:x+w]
        text3 = pytesseract.image_to_string(digit_roi, config='--psm 10 digits -c tessedit_char_whitelist=0123456789')
        digits3 = "".join(filter(str.isdigit, text3)).strip()

    # Attempt 4: OCR with --psm 8 (single word) (Tesseract)
    text4 = pytesseract.image_to_string(thresh1, config='--psm 8 digits -c tessedit_char_whitelist=0123456789')
    digits4 = "".join(filter(str.isdigit, text4)).strip()

    # Attempt 5: Use EasyOCR as a fallback
    result = easyocr_reader.readtext(np.array(image), allowlist='0123456789')
    digits5 = ""
    for detection in result:
        text = detection[1]
        digits5 = "".join(filter(str.isdigit, text)).strip()
        if digits5:
            break

    # Select the first successful OCR result
    digits = digits1 or digits2 or digits3 or digits4 or digits5
    if digits:
        return int(digits[0]), thresh1_base64, thresh2_base64
    return None, thresh1_base64, thresh2_base64

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Image Upload & Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file!"})
    model_choice = request.form.get('model', '').lower()
    if model_choice not in models_dict:
        return jsonify({"error": "Invalid model choice!"})

    img_path = os.path.join("uploads", file.filename)
    file.seek(0)
    file.save(img_path)
    image = Image.open(img_path).convert("RGB")

    try:
        file.seek(0)
        input_tensor, _ = preprocess_image(img_path)
        input_tensor = input_tensor.to(device)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})

    model = models_dict[model_choice]
    with torch.no_grad():
        if model_choice == "autoencoder":
            output = model(input_tensor)
            recon_image = transforms.ToPILImage()(output.squeeze(0).cpu())
            result = "Image reconstructed"
            confidence = None
            predicted_number = None
        else:
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            predicted_number = prediction.item()
            confidence = confidence.item() * 100
            result = f"Predicted Number: {predicted_number}"

    ocr_number, thresh1_base64, thresh2_base64 = extract_number(image)
    if ocr_number is not None:
        ocr_result = f"OCR Extracted Number: {ocr_number}"
        if predicted_number is not None:
            match = ""
        else:
            match = "Comparison not possible: Autoencoder model used."
    else:
        ocr_result = "No digits detected"
        match = "Comparison not possible: Invalid OCR prediction result."

    # Apply fix: Use OCR result if model is not confident
    if confidence is not None and confidence < 20 and ocr_number is not None:
        final_number = ocr_number
        result = f"OCR Used Instead - Final Predicted Number: {final_number}"
    else:
        final_number = predicted_number

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    if model_choice != "autoencoder":
        prob_dict = {str(i): f"{probabilities[0][i].item() * 100:.2f}%" for i in range(10)}
    else:
        prob_dict = None

    response = {
        "model": model_choice,
        "prediction": result,
        "final_number": final_number,
        "confidence": f"{confidence:.2f}%" if confidence is not None else "N/A",
        "probabilities": prob_dict,
        "ocr_result": ocr_result,
        "match": match,
        "uploaded_image": img_base64,
        "thresh_image1": thresh1_base64,
        "thresh_image2": thresh2_base64,
    }

    if model_choice == "autoencoder":
        recon_buffered = BytesIO()
        recon_image.save(recon_buffered, format="PNG")
        recon_base64 = base64.b64encode(recon_buffered.getvalue()).decode('utf-8')
        response["reconstructed_image"] = recon_base64

    return jsonify(response)


# Optimizer Comparison Route
@app.route('/optimizer_comparison', methods=['POST'])
def optimizer_comparison():
    model_choice = request.form.get('model', '').lower()
    if model_choice not in models_dict or model_choice == "autoencoder":
        return jsonify({"error": "Invalid model choice or unsupported for optimizer comparison!"})
    model = models_dict[model_choice]
    criterion = nn.CrossEntropyLoss()
    results = compare_optimizers(model, train_loader, criterion, epochs=3)
    plt.figure(figsize=(10, 5))
    for opt, loss_hist in results.items():
        plt.plot(loss_hist, label=f"{opt} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Optimizer Comparison - {model_choice} Model")
    plt.legend()
    plt.grid(True)
    plot_path = "static/optimizer_comparison.png"
    plt.savefig(plot_path)
    plt.close()
    return jsonify({"message": "Optimizer comparison complete!", "plot": plot_path})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)