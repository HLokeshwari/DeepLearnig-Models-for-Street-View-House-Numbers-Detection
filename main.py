import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from models import MulticlassClassifier, CNNModel, get_alexnet, get_vgg16, get_rnn, get_lstm, get_autoencoder, \
    get_resnet, compare_optimizers
from preprocessing import load_data

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Load dataset with optimized batch size
train_loader, test_loader = load_data(batch_size=32)
print(f"Total Batches in Train Loader: {len(train_loader)}")
print(f"Total Batches in Test Loader: {len(test_loader)}")

# Analyze class distribution
labels = train_loader.dataset.labels  # Directly use precomputed labels
class_counts = Counter(labels)
print("Class Distribution in Training Data:", class_counts)

# Compute class weights for imbalanced dataset
num_classes = 10
class_weights = torch.tensor([1.0 / (class_counts[i] + 1e-6) for i in range(num_classes)], dtype=torch.float).to(device)
print("Class Weights:", class_weights)

# Define loss functions
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss for classification
ae_criterion = nn.MSELoss()  # For autoencoder

# Initialize models
multi_model = MulticlassClassifier(num_classes=10).to(device)
cnn_model = CNNModel(num_classes=10).to(device)
alexnet_model = get_alexnet(num_classes=10).to(device)
vgg16_model = get_vgg16(num_classes=10).to(device)
rnn_model = get_rnn(num_classes=10).to(device)
lstm_model = get_lstm(num_classes=10).to(device)
autoencoder = get_autoencoder().to(device)
resnet_model = get_resnet(num_classes=10).to(device)

# Model dictionary
models = {
    "multiclass": multi_model,
    "cnn": cnn_model,
    "alexnet": alexnet_model,
    "vgg16": vgg16_model,
    "rnn": rnn_model,
    "lstm": lstm_model,
    "autoencoder": autoencoder,
    "resnet": resnet_model
}

# Load existing models if available
for model_name, model in models.items():
    model_path = f"models/{model_name}_model.pth"
    if os.path.exists(model_path):
        print(f"✅ Loading model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"⚠️ Warning: Model file '{model_path}' not found! Training from scratch.")

# Compare optimizers (example for CNN model)
print("\n🔍 Comparing optimizers on CNN model...")
results = compare_optimizers(cnn_model, train_loader, criterion, epochs=3)

# Save optimizer loss comparison plot
plt.figure(figsize=(10, 5))
for opt, loss_hist in results.items():
    plt.plot(loss_hist, label=f"{opt} Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Optimizer Comparison - Loss vs. Epochs")
plt.legend()
plt.grid(True)
plt.savefig("static/optimizer_comparison.png")
plt.close()
print("✅ Optimizer comparison plot saved: static/optimizer_comparison.png")


# Validation function
def evaluate_model(model, test_loader, criterion, is_autoencoder=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if is_autoencoder:
                loss = criterion(outputs, images)
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total if not is_autoencoder else None
    return avg_loss, accuracy, all_preds, all_labels


# Training function with early stopping and learning rate scheduling
def train_model(model, optimizer, model_name, num_epochs=20, is_autoencoder=False, patience=5):
    model.train()
    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = [] if not is_autoencoder else None
    loss_fn = ae_criterion if is_autoencoder else criterion

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"\n🚀 Training {model_name} model...")
    for epoch in range(num_epochs):
        # Training phase
        total_train_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, images if is_autoencoder else labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss, val_accuracy, _, _ = evaluate_model(model, test_loader, loss_fn, is_autoencoder)
        val_loss_history.append(val_loss)
        if not is_autoencoder:
            val_accuracy_history.append(val_accuracy)
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        else:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            model_path = f"models/{model_name}_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved: {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break

        model.train()

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Train Loss", linestyle='dashed')
    plt.plot(val_loss_history, label="Val Loss", linestyle='solid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"static/{model_name}_loss.png")
    plt.close()
    print(f"✅ Loss plot saved: static/{model_name}_loss.png")

    # Plot validation accuracy (for classification models)
    if not is_autoencoder:
        plt.figure(figsize=(10, 5))
        plt.plot(val_accuracy_history, label="Val Accuracy", linestyle='solid', color='green')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{model_name} Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"static/{model_name}_accuracy.png")
        plt.close()
        print(f"✅ Accuracy plot saved: static/{model_name}_accuracy.png")


# Debug predictions function (for classification models only)
def debug_predictions(model, test_loader, model_name):
    if model_name == "autoencoder":
        print(f"Skipping debug_predictions for {model_name} (not a classification model)")
        return

    model.eval()
    predictions = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            for i in range(len(labels)):
                predictions.append({
                    "true_label": labels[i].item(),
                    "predicted_label": predicted[i].item(),
                    "confidence": confidences[i].item() * 100
                })
            if len(predictions) >= 10:  # Limit to 10 examples
                break

    print(f"\nDebugging {model_name} Predictions (First 10 Examples):")
    for pred in predictions:
        print(
            f"True: {pred['true_label']}, Predicted: {pred['predicted_label']}, Confidence: {pred['confidence']:.2f}%")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"static/{model_name}_confusion_matrix.png")
    plt.close()
    print(f"✅ Confusion matrix saved: static/{model_name}_confusion_matrix.png")


# Evaluate all models and find the best one
def compare_all_models(models, test_loader, criterion, ae_criterion):
    print("\n🔍 Evaluating all models on test dataset...")
    model_metrics = {}

    for model_name, model in models.items():
        is_autoencoder = model_name == "autoencoder"
        loss_fn = ae_criterion if is_autoencoder else criterion
        test_loss, test_accuracy, _, _ = evaluate_model(model, test_loader, loss_fn, is_autoencoder)

        model_metrics[model_name] = {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy if not is_autoencoder else None
        }

        if is_autoencoder:
            print(f"{model_name} - Test Loss: {test_loss:.4f} (Accuracy not applicable)")
        else:
            print(f"{model_name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Find the best model (excluding autoencoder)
    classification_models = {k: v for k, v in model_metrics.items() if k != "autoencoder"}
    best_model = max(classification_models.items(), key=lambda x: x[1]["test_accuracy"] or -float("inf"))

    print("\n🏆 Best Model Summary:")
    print(f"Model: {best_model[0]}")
    print(f"Test Accuracy: {best_model[1]['test_accuracy']:.2f}%")
    print(f"Test Loss: {best_model[1]['test_loss']:.4f}")

    # Plot accuracy comparison for classification models
    plt.figure(figsize=(12, 6))
    model_names = [name for name in classification_models.keys()]
    accuracies = [metrics["test_accuracy"] for metrics in classification_models.values()]

    bars = plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel("Models")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')

    # Highlight the best model
    best_idx = model_names.index(best_model[0])
    bars[best_idx].set_color('orange')

    plt.tight_layout()
    comparison_plot_path = "static/model_accuracy_comparison.png"
    plt.savefig(comparison_plot_path)
    plt.close()
    print(f"✅ Model accuracy comparison plot saved: {comparison_plot_path}")

    return model_metrics, best_model


# Train models with optimized learning rates and early stopping
try:
    train_model(multi_model, optim.Adam(multi_model.parameters(), lr=0.001), "multiclass", num_epochs=5, patience=5)
    debug_predictions(multi_model, test_loader, "multiclass")

    train_model(cnn_model, optim.Adam(cnn_model.parameters(), lr=0.001), "cnn", num_epochs=5, patience=5)
    debug_predictions(cnn_model, test_loader, "cnn")

    train_model(alexnet_model, optim.Adam(alexnet_model.parameters(), lr=0.0001), "alexnet", num_epochs=5, patience=5)
    debug_predictions(alexnet_model, test_loader, "alexnet")

    train_model(vgg16_model, optim.Adam(vgg16_model.parameters(), lr=0.0001), "vgg16", num_epochs=5, patience=5)
    debug_predictions(vgg16_model, test_loader, "vgg16")

    train_model(rnn_model, optim.Adam(rnn_model.parameters(), lr=0.001), "rnn", num_epochs=5, patience=5)
    debug_predictions(rnn_model, test_loader, "rnn")

    train_model(lstm_model, optim.Adam(lstm_model.parameters(), lr=0.001), "lstm", num_epochs=5, patience=5)
    debug_predictions(lstm_model, test_loader, "lstm")

    train_model(autoencoder, optim.Adam(autoencoder.parameters(), lr=0.001), "autoencoder", num_epochs=5,
                is_autoencoder=True, patience=5)
    debug_predictions(autoencoder, test_loader, "autoencoder")  # Will be skipped

    train_model(resnet_model, optim.Adam(resnet_model.parameters(), lr=0.0001), "resnet", num_epochs=5, patience=5)
    debug_predictions(resnet_model, test_loader, "resnet")

    # Evaluate all models and find the best one
    model_metrics, best_model = compare_all_models(models, test_loader, criterion, ae_criterion)

    torch.autograd.set_detect_anomaly(True)

except Exception as e:
    print(f"❌ Training failed: {str(e)}")