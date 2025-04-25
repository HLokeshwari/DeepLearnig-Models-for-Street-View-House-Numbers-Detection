import torch
import matplotlib.pyplot as plt


# Save model checkpoint
def save_model(model, filename="best_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")


# Load model checkpoint
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    print(f"Model {filename} loaded successfully.")


# Plot training results
def plot_results(results):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results["optimizers"], results["val_losses"], marker='o', label="Loss")
    plt.title("Validation Loss")
    plt.xlabel("Optimizer")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(results["optimizers"], results["val_accuracies"], marker='o', label="Accuracy", color="g")
    plt.title("Validation Accuracy")
    plt.xlabel("Optimizer")
    plt.ylabel("Accuracy (%)")

    plt.legend()
    plt.show()
