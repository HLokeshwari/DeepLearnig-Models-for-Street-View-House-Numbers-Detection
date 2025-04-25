SVHN Digit Classification with Multi-Model Deep Learning

The system addresses real-world digit classification challenges using the SVHN dataset, achieving a test accuracy of 71.40% with ResNet18.

📖 Project Overview

The Street View House Numbers (SVHN) dataset presents challenges like class imbalance and visual noise (e.g., uneven illumination, occlusion). This project develops a robust pipeline combining:

Multiple Deep Learning Models: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs), and transfer learning models (ResNet18, VGG16, AlexNet).

Preprocessing Pipeline: Data augmentation (flips, rotations, color jitter) and oversampling for minority digits (“1”, “7”).

Flask Web Application: Real-time digit recognition with OCR validation using EasyOCR and Tesseract.

Evaluation: Comprehensive metrics (accuracy, loss, confusion matrices) visualized with Matplotlib and Seaborn.

Key Achievements:

ResNet18 achieved 71.40% test accuracy.

OCR integration enhances reliability for low-confidence predictions.

Optimized training with early stopping, learning rate scheduling, and optimizer comparison (Adam vs. SGD).

Watch a short demo video: SVHN Project Demo

🛠️ Requirements
   Hardware:
    NVIDIA GPU (e.g., GTX 1080) for training (optional; CPU sufficient for inference).
    Minimum 16 GB RAM.
   50 GB storage for dataset and models.

Software:
   Python 3.8+
   PyTorch 1.9
   Flask
   EasyOCR, Tesseract
   NumPy, Pandas, Matplotlib, Seaborn
   OpenCV (cv2)
   tqdm

Install dependencies:

pip install -r requirements.txt

📂 Repository Structure

svhn-digit-classification/
├── Dataset/                  # SVHN dataset (train/, test/, labels.txt)
├── models/                   # Trained model weights
├── static/                   # Plots (confusion matrices, loss/accuracy curves)
├── uploads/                  # Uploaded images for Flask app
├── preprocessing.py          # Data loading and preprocessing
├── models.py                 # Model architectures (CNN, RNN, LSTM, etc.)
├── main.py                   # Training and evaluation script
├── app1.py                   # Flask web application
├── templates/                # HTML templates for Flask
│   └── index.html
├── requirements.txt          # Python dependencies
├── README.md                 # This file

🚀 Getting Started

Clone the Repository:

git clone https://github.com/lokeshwari14/svhn-digit-classification.git
cd svhn-digit-classification

Download the SVHN Dataset:

Get Format 2 from SVHN Dataset.
Place train/ and test/ folders in Dataset/.

Ensure labels.txt is included in each folder.

Install Tesseract:

Windows: Download and install from Tesseract OCR.
Update app1.py with the Tesseract path:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Set Up Environment:

pip install -r requirements.txt

Train Models:

python main.py
Trains all models (Multiclass, CNN, RNN, LSTM, ResNet18, VGG16, AlexNet, Autoencoder).
Saves weights in models/ and plots in static/.

Run the Flask App:
python app1.py


Access the web interface at http://localhost:5000.



Upload an image, select a model, and view predictions with OCR validation.

Plots:

Confusion matrices: static/<model>_confusion_matrix.png
Loss curves: static/<model>_loss.png
Accuracy curves: static/<model>_accuracy.png
Optimizer comparison: static/optimizer_comparison.png
Confusion matrix for ResNet18 model.

📬 Contact

For questions or collaboration, reach out to:
Lokeshwari Hukumathirao: hlokeshwari14@gmail.com



LinkedIn: Your LinkedIn Profile (replace with your actual profile)



License: MIT License
