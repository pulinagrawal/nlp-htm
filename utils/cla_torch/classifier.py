from tabnanny import verbose
import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tqdm import tqdm  # For progress bar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier 
# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from image_encoder import ImageEncoder, topk_per_location, binary_topk_encoding

def encode_image(encoder, image, k=2, threshold=0.0, method='topk_values'):
    """
    Encodes an RGB image using the provided encoder.

    Args:
        encoder (ImageEncoder): The image encoder.
        image (torch.Tensor): The input image tensor of shape [3, H, W].
        k (int): Number of top activations to consider per location.
        threshold (float): Threshold for activation values (used in binary_topk_encoding).
        method (str): Encoding method to use ('topk_values' or 'binary_topk').

    Returns:
        np.ndarray: The encoded image flattened into a 1D feature vector.
    """
    # Split the image into R, G, B channels
    r_channel, g_channel, b_channel = image[0, :, :], image[1, :, :], image[2, :, :]

    # Initialize list to store encoded channels
    encoded_channels = []

    for channel in [r_channel, g_channel, b_channel]:
        input_image = channel.unsqueeze(0)  # Shape: [1, H, W]

        # Encode the channel
        output = encoder(input_image)
        topk_values, topk_indices = topk_per_location(output, k)
        num_filters = output.shape[0]
        input_shape = input_image.shape

        if method == 'topk_values':
            # Using topk_values as encoding
            mask = torch.zeros_like(output.permute(2, 1, 0))
            masked_encoding = mask.scatter_(dim=2, index=topk_indices, src=topk_values)
        else:
            # Using binary_topk_encoding with threshold
            binary_encoding = binary_topk_encoding(topk_indices, topk_values, num_filters, threshold)
            masked_encoding = binary_encoding

        # Flatten the encoding and append to the list
        flattened_encoding = masked_encoding.flatten().numpy()
        encoded_channels.append(flattened_encoding)

    # Concatenate the encoded channels to form a single feature vector
    encoded_image = np.concatenate(encoded_channels)

    return encoded_image

def main():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    # Parameters for the encoder
    n_filters = 25 
    filter_size = 3
    std_dev_min = 0.2
    std_dev_max = 0.4
    center_min = 1
    center_max = 1.5
    sparsity = 0.7
    k = 2
    threshold = 0.0  # Adjust as needed
    method = 'topk_values'  # Choose 'topk_values' or 'binary_topk'

    # Create the encoder
    encoder = ImageEncoder(
        n_filters=n_filters,
        filter_size=filter_size,
        std_dev_range=(std_dev_min, std_dev_max),
        center_range=(center_min, center_max),
        sparsity=sparsity
    )
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Prepare data
    X = []
    y = []

    print("Encoding images...")
    for image, label in tqdm(cifar_dataset):
        encoded_image = encode_image(encoder, image, k=k, threshold=threshold, method=method)
        X.append(encoded_image)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Optionally, standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classifiers to evaluate
    classifiers = {
        # 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, verbose=1),
        # 'SVM (Linear Kernel)': SVC(kernel='linear', random_state=42, verbose=1),
        # 'SVM (RBF Kernel)': SVC(kernel='rbf', random_state=42, verbose=1),
        # 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, verbose=1), 
        'MLP Classifier': MLPClassifier(hidden_layer_sizes=(2000,100), max_iter=200, random_state=42),
        'Naive Bayes': GaussianNB()
    }

    # Evaluate each classifier
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name}: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    main()