from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data 
Y = iris.target


#Splitting the dataset into training and testing sets.

X_train_raw = np.vstack([X[0:30], X[50:80], X[100:130]]) #consist of 30 samples (flower and 4 features )from each class
Y_train = np.hstack([Y[0:30], Y[50:80], Y[100:130]]) #labels (type of flower)

X_test_raw = np.vstack([X[30:50], X[80:100], X[130:150]])
Y_test = np.hstack([Y[30:50], Y[80:100], Y[130:150]])


# ------ Normalising the dataset ---------

mu = X_train_raw.mean(axis=0)  # mean of each column (feature)
std = X_train_raw.std(axis=0)    # std of each column (feature)

X_train = (X_train_raw - mu) / std
X_test  = (X_test_raw  - mu) / std  

def add_bias(X):
    N = X.shape[0]
    bias = np.ones((N, 1))  # Create a column of ones for
    return np.hstack([X, bias])


def one_hot(labels, num_classes = 3):
    N = len(labels)
    T = np.zeros((N, num_classes))
    for i, label in enumerate(labels): 
        T[i, label] = 1
    return T                


# --------- applying the functions to the training and testing data -----------
X_train_b = add_bias(X_train)
X_test_b = add_bias(X_test)

T_train = one_hot(Y_train)

print("X_train_b shape:", X_train_b.shape)  # Should be (90, 5)
print("T_train shape:  ", T_train.shape)    # Should be (90, 3)

# Clasifier

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def forward(X_b, W): 
    z = X_b @ W.T # matrix multiplication
    return sigmoid(z)


def compute_mse(g, T):
    error = g - T
    mse = 0.5 * np.mean(np.sum(error**2, axis=1))
    return mse  


def compute_gradient(X_b, g, T):
    error = g - T                 
    delta = error * g * (1 - g)  # element-wise multiplication
    return delta.T @ X_b

def train(X_b, T, alpha=0.01, epochs=1000):
    C = T.shape[1]        # 3 classes
    D = X_b.shape[1]      # 5 inputs

    W = np.zeros((C, D))  # start with all zeros
    mse_history = []

    for epoch in range(epochs):
        g    = forward(X_b, W)           
        mse  = compute_mse(g, T)        
        grad = compute_gradient(X_b, g, T)  
        W    = W - alpha * grad          # Equation 23 - take a step

        mse_history.append(mse)

    return W, mse_history


# -------- train the model and get the weights and the history of MSE --------

W, mse_history = train(X_train_b, T_train, alpha=0.005, epochs=3000)

# Plot MSE - this is how you check if training worked
plt.plot(mse_history)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training convergence')
plt.grid(True)
plt.show()

print("Starting MSE:", round(mse_history[0], 4))
print("Final MSE:   ", round(mse_history[-1], 4))


# --------- Confusion matrix -----------------


def predict(X_b, W):
    g = forward(X_b, W)  # get the predicted probabilities
    return np.argmax(g, axis=1)  # return the class with the highest probability

def confusion_matrix(y_true, y_pred, num_classes = 3):
    C = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        C[true, pred] += 1
    return C

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

pred_train = predict(X_train_b, W)
pred_test  = predict(X_test_b,  W)

print("TRAINING SET:")
print("Confusion matrix:")
print(confusion_matrix(Y_train, pred_train))
print("Error rate:", error_rate(Y_train, pred_train))

print("\n TEST SET: ")
print("Confusion matrix:")
print(confusion_matrix(Y_test, pred_test))
print("Error rate:", error_rate(Y_test, pred_test))


# ---------- last 30 of each ----------------- 

X_train2_raw = np.vstack([X[30:60], X[70:100], X[120:150]])
Y_train2 = np.hstack([Y[30:60], Y[70:100], Y[120:150]])

X_test2_raw = np.vstack([X[0:20], X[50:70], X[100:120]])
Y_test2 = np.hstack([Y[0:20], Y[50:70], Y[100:120]])

# ------ Normalising the dataset ---------

mu = X_train2_raw.mean(axis=0)  # mean of each column (feature)
std = X_train2_raw.std(axis=0)    # std of each column (feature)

X_train2 = (X_train2_raw - mu) / std
X_test2  = (X_test2_raw  - mu) / std  
# -----------------------------------

X_train2_b = add_bias(X_train2)
X_test2_b = add_bias(X_test2)   
T_train2 = one_hot(Y_train2)

W2, mse_history2 = train(X_train2_b, T_train2, alpha=0.005, epochs=3000)

pred_train2 = predict(X_train2_b, W2)
pred_test2  = predict(X_test2_b,  W2)   

print("CASE 2: TRAINING SET:")
print("Confusion matrix:")
print(confusion_matrix(Y_train2, pred_train2))
print("Error rate:", error_rate(Y_train2, pred_train2))

print("\nCASE 2: TRAINING SET:")
print("Confusion matrix:")
print(confusion_matrix(Y_test2, pred_test2))
print("Error rate:", error_rate(Y_test2, pred_test2))


# ---------Histogram ----------------

feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

class_names = ['Setosa', 'Versicolor', 'Virginica']
colors = ['blue', 'orange', 'green']

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 8))
axes = [ax0, ax1, ax2, ax3]

for features in range(4):
    for classes in range(3):
        axes[features].hist(X_train[Y_train == classes, features], 
                             alpha=0.5, 
                             label=class_names[classes], 
                             color=colors[classes],
                             bins=15 ) 
    axes[features].set_title(feature_names[features])
    axes[features].set_xlabel('Normalized Value')
    axes[features].set_ylabel('count')
    axes[features].legend()

plt.suptitle('Feature histograms for all 3 class', fontsize=14)
plt.tight_layout()
plt.show()

print('\n')


# ------- Removing the feature with the most overlap ----

X_train_3f = np.delete(X_train, 1 , axis=1)  # remove sepal width
X_test_3f  = np.delete(X_test, 1 , axis=1)  


X_train_3f_b = add_bias(X_train_3f)
X_test_3f_b  = add_bias(X_test_3f)  

T_train_3f = one_hot(Y_train)

print("T_train_3f shape:  ", T_train_3f.shape)
print("X_train_3f_b shape:", X_train_3f_b.shape)  # Should be (90, 4)

W_3f, mse_history_3f = train(X_train_3f_b, T_train_3f, alpha=0.005, epochs=3000)

plt.plot(mse_history_3f)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training convergence (3 features)')
plt.grid(True)
plt.show()

print("Starting MSE (3 features):", round(mse_history_3f[0], 4))
print("Final MSE (3 features):   ", round(mse_history_3f[-1], 4))

pred_train_3f = predict(X_train_3f_b, W_3f)
pred_test_3f  = predict(X_test_3f_b,  W_3f)

print("\n 3 FEATURES: Removed Sepal Width:")

print("\nTraining confusion matrix:")
print(confusion_matrix(Y_train, pred_train_3f))
print("Training error rate:", error_rate(Y_train, pred_train_3f))

print("\nTest confusion matrix:")
print(confusion_matrix(Y_test, pred_test_3f))
print("Test error rate:", error_rate(Y_test, pred_test_3f))

# ------- Traning with only 2 feature ----

X_train_2f = np.delete(X_train, [0, 1] , axis=1)  # remove sepal length and sepal width
X_test_2f  = np.delete(X_test, [0, 1] , axis=1)


X_train_2f_b = add_bias(X_train_2f)
X_test_2f_b  = add_bias(X_test_2f)

T_train_2f = one_hot(Y_train)

W_2f, mse_history_2f = train(X_train_2f_b, T_train_2f, alpha=0.005, epochs=3000)

plt.plot(mse_history_2f)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training convergence (2 features)')
plt.grid(True)
plt.show()

print("Starting MSE (2 features):", round(mse_history_2f[0], 4))
print("Final MSE (2 features):   ", round(mse_history_2f[-1], 4))

pred_train_2f = predict(X_train_2f_b, W_2f)
pred_test_2f  = predict(X_test_2f_b,  W_2f)

print("\n 2 FEATURES: Petal Length and Petal Width:")

print("\nTraining confusion matrix:")
print(confusion_matrix(Y_train, pred_train_2f))
print("Training error rate:", error_rate(Y_train, pred_train_2f))

print("\nTest confusion matrix:")
print(confusion_matrix(Y_test, pred_test_2f))
print("Test error rate:", error_rate(Y_test, pred_test_2f))

# ------- only 1 feature ----

X_train_1f = np.delete(X_train, [0, 1, 3] , axis=1)  # remove sepal length, sepal width, and petal length
X_test_1f  = np.delete(X_test, [0, 1, 3] , axis=1)



X_train_1f_b = add_bias(X_train_1f)
X_test_1f_b  = add_bias(X_test_1f)

T_train_1f = one_hot(Y_train)

W_1f, mse_history_1f = train(X_train_1f_b, T_train_1f, alpha=0.005, epochs=3000)

plt.plot(mse_history_1f)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training convergence (1 feature)')
plt.grid(True)
plt.show()

print("Starting MSE (1 feature):", round(mse_history_1f[0], 4))
print("Final MSE (1 feature):   ", round(mse_history_1f[-1], 4))

pred_train_1f = predict(X_train_1f_b, W_1f)
pred_test_1f  = predict(X_test_1f_b,  W_1f)

print("\n 1 FEATURE: Petal Length:")

print("\nTraining confusion matrix:")
print(confusion_matrix(Y_train, pred_train_1f))
print("Training error rate:", error_rate(Y_train, pred_train_1f))

print("\nTest confusion matrix:")
print(confusion_matrix(Y_test, pred_test_1f))
print("Test error rate:", error_rate(Y_test, pred_test_1f))