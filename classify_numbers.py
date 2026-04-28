import numpy as np 
import matplotlib.pyplot as plt
import time 
import torch  
from sklearn.datasets import fetch_openml


# ----- Chosing gpu if available, otherwise using cpu -----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")


# ----- Loading the dataset -----
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32)
Y = mnist.target.astype(np.int64)   

print("X shape:", X.shape)  # Should be (70000, 784)
print("Y shape:", Y.shape)  # Should be (70000,)


# ---- Splitting the dataset into training and testing sets -----
X_train_raw = X[:60000]  # First 60,000 samples for training
Y_train = Y[:60000]

X_test = X[60000:]   # Last 10,000 samples for testing
Y_test = Y[60000:]

# ----- Normalising the dataset -----
mu = X_train_raw.mean(axis=0) 
std = X_train_raw.std(axis=0)    
std[std == 0] = 1  # To avoid division by zero for features with zero variance

X_norm = (X_train_raw - mu) / std
X_test_norm = (X_test - mu) / std

# ------ Convert to PyTorch tensors and move to the selected device -----
X_train_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)

print("data moved to device:", device)


# ----- Nearest Neighbor Classifier using Euclidean distance -----

def nn_classifier_torch(X_train_tensor, Y_train_tensor, X_test_tensor, chunk_size=1000):
    predictions = []
    start_time = time.time()

    for start in range(0, X_test_tensor.shape[0], chunk_size):
        end = min(start + chunk_size, X_test_tensor.shape[0])
        X_test_chunk = X_test_tensor[start:end]
        
        # Compute distances between the test chunk and the entire training set
        distances = torch.cdist(X_test_chunk, X_train_tensor, p=2)
        
        # Find the indices of the nearest neighbors
        nearest_indices = torch.argmin(distances, dim=1)
        
        pred_chunk = Y_train_tensor[nearest_indices]
        predictions.append(pred_chunk)

        print(f"Classified test image {start} to {end}")
    
    predictions = torch.cat(predictions)
    end_time = time.time()
    print(f"Classification time:", round(end_time - start_time, 2), "seconds")

    return predictions

# --- testing the classifier on the test set ---

Y_pred_tensor = nn_classifier_torch(X_train_tensor, Y_train_tensor, X_test_tensor, chunk_size=1000)

# --- confusion matrix ---

def confusion_matrix_torch(y_true_t, y_pred_t, num_classes=10):
    C = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    y_true = y_true_t.cpu()
    y_pred = y_pred_t.cpu()

    for true, pred in zip(y_true, y_pred):
        C[true, pred] += 1
    return C


def error_rate_torch(y_true_t, y_pred_t):
    return torch.mean((y_true_t != y_pred_t).float()).item()

C_test = confusion_matrix_torch(Y_test_tensor, Y_pred_tensor)
err_test = error_rate_torch(Y_test_tensor, Y_pred_tensor)


print("Confusion matrix:")
print(C_test.numpy())
print("Error rate:", err_test)

# ----- plotting som missclassifeied images -----


def plot_images(X, Y_true, Y_pred, indices, num_images=12, title="Images"):
    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        idx = indices[i]
        image = X[idx].reshape(28, 28)

        plt.subplot(3, 4, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"True: {Y_true[idx]}, Pred: {Y_pred[idx]}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



# Move labels and predictions to CPU/NumPy

Y_test_np = Y_test_tensor.cpu().numpy()
Y_pred_np = Y_pred_tensor.cpu().numpy()

# Find indices where prediction is wrong

misclassified_idx = np.where(Y_test_np != Y_pred_np)[0]

print("Number of misclassified images:", len(misclassified_idx))
print("First 10 misclassified indices:", misclassified_idx[:10])

num_images = 12

plot_images(X_test, Y_test_np, Y_pred_np, misclassified_idx, num_images=12, title="Misclassified MNIST test images")


# ----- Ploting som correctly classified images -----

correct_idx = np.where(Y_test_np == Y_pred_np)[0]

print("Number of correctly classified images:", len(correct_idx))
print("First 10 correctly classified indices:", correct_idx[:10])


num_images = 12

plot_images(X_test, Y_test_np, Y_pred_np, correct_idx, num_images=12, title="Correctly classified MNIST test images")