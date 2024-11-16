import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from matplotlib.widgets import Slider, Button, RadioButtons

from image_encoder import ImageEncoder, reconstruct_from_binary_encoding, topk_per_location

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def binary_topk_encoding(topk_indices, topk_values, num_filters, threshold):
    """
    Creates a binary representation of the top-k encoding output, setting indices to 1
    only if the corresponding activation values are greater than the threshold.

    Args:
        topk_indices (torch.Tensor): Indices of the top-k filters per spatial location, shape [H, W, k]
        topk_values (torch.Tensor): Activation values of the top-k filters per spatial location, shape [H, W, k]
        num_filters (int): Total number of filters
        threshold (float): Threshold value. Only activation values greater than this will be set in the encoding.

    Returns:
        binary_encoding (torch.Tensor): Binary tensor indicating presence of top-k filters at each location,
                                        shape [H, W, num_filters], dtype torch.float32
    """
    H, W, k = topk_indices.shape
    # Flatten H and W dimensions
    topk_indices_flat = topk_indices.view(-1, k)  # Shape: [H*W, k]
    topk_values_flat = topk_values.view(-1, k)    # Shape: [H*W, k]

    # Apply threshold to activation values
    mask = topk_values_flat > threshold  # Shape: [H*W, k]

    # Create binary encoding
    binary_encoding_flat = torch.zeros((H*W, num_filters), dtype=torch.float32)
    # Use scatter to set the top-k indices to 1 where activation values are above threshold
    indices_to_set = topk_indices_flat[mask]
    positions = torch.nonzero(mask)
    if indices_to_set.numel() > 0:
        binary_encoding_flat[positions[:, 0], indices_to_set] = 1

    # Reshape back to [H, W, num_filters]
    binary_encoding = binary_encoding_flat.view(H, W, num_filters)
    return binary_encoding

class RGBExtractor:
    def __init__(self):
        pass

    def extract_rgb(self, image):
        """
        Extracts and returns the R, G, B channels from an RGB image.

        Args:
            image (torch.Tensor): Input image of shape [3, height, width]

        Returns:
            tuple: R, G, B channels as separate tensors
        """
        r_channel = image[0, :, :]
        g_channel = image[1, :, :]
        b_channel = image[2, :, :]
        return r_channel, g_channel, b_channel

def main():
    # Load CIFAR-10 test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    cifar_dataset.index = 0

    # Initialize parameters
    n_filters = 50
    filter_size = 8
    std_dev_min = 0.2
    std_dev_max = 0.4
    center_min = 3
    center_max = 5
    sparsity = 0.7
    k = 5
    threshold = 0.0  # Initial threshold value

    num_images = 5  # Adjusted for better visualization

    # Get the first num_images images
    images = []
    labels = []
    for i in range(num_images):
        idx = (cifar_dataset.index + i) % len(cifar_dataset)
        image, label = cifar_dataset[idx]
        images.append(image)
        labels.append(label)
    cifar_dataset.index = (cifar_dataset.index + num_images) % len(cifar_dataset)

    # Separating the R G B Values
    r_val_input_images = []
    g_val_input_images = []
    b_val_input_images = []

    rgb_extractor = RGBExtractor()

    for image in images:
        r_channel, g_channel, b_channel = rgb_extractor.extract_rgb(image)
        r_input_image = r_channel.unsqueeze(0)
        g_input_image = g_channel.unsqueeze(0)
        b_input_image = b_channel.unsqueeze(0)
        r_val_input_images.append(r_input_image)
        g_val_input_images.append(g_input_image)
        b_val_input_images.append(b_input_image)

    # Adjust the figure and axes to accommodate the combined images
    fig, axes = plt.subplots(8, num_images, figsize=(num_images * 2, 10))
    plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.9, wspace=0.8, hspace=0.5)

    # Lists to store reconstructed images and displays
    reconstructed_images_r, reconstructed_images_g, reconstructed_images_b = [], [], []
    original_displays_r, original_displays_g, original_displays_b = [], [], []
    reconstructed_displays_r, reconstructed_displays_g, reconstructed_displays_b = [], [], []
    original_displays_rgb, reconstructed_displays_rgb = [], []

    for i in range(num_images):

        ax_original_r = axes[0, i]
        ax_reconstructed_r = axes[1, i]
        ax_original_g = axes[2, i]
        ax_reconstructed_g = axes[3, i]
        ax_original_b = axes[4, i]
        ax_reconstructed_b = axes[5, i]
        ax_original_rgb = axes[6, i]         # For original RGB image
        ax_reconstructed_rgb = axes[7, i]    # For reconstructed RGB image

        # Display original R, G, B channels
        ax_original_r.imshow(r_val_input_images[i].squeeze(0).numpy(), cmap='Reds')
        ax_original_r.set_title(f'Red - Label: {labels[i]}')
        ax_original_r.axis('off')
        ax_original_g.imshow(g_val_input_images[i].squeeze(0).numpy(), cmap='Greens')
        ax_original_g.set_title('Green')
        ax_original_g.axis('off')
        ax_original_b.imshow(b_val_input_images[i].squeeze(0).numpy(), cmap='Blues')
        ax_original_b.set_title('Blue')
        ax_original_b.axis('off')

        # Display original RGB image
        original_rgb_image = np.transpose(images[i].numpy(), (1, 2, 0))
        ax_original_rgb.imshow(original_rgb_image)
        ax_original_rgb.set_title('Original RGB')
        ax_original_rgb.axis('off')
        original_displays_rgb.append(ax_original_rgb)

        # Initialize reconstructed images with zeros
        reconstructed_image_r = np.zeros((32, 32))
        reconstructed_display_r = ax_reconstructed_r.imshow(reconstructed_image_r, cmap='Reds', vmin=0, vmax=1)
        ax_reconstructed_r.axis('off')
        reconstructed_image_g = np.zeros((32, 32))
        reconstructed_display_g = ax_reconstructed_g.imshow(reconstructed_image_g, cmap='Greens', vmin=0, vmax=1)
        ax_reconstructed_g.axis('off')
        reconstructed_image_b = np.zeros((32, 32))
        reconstructed_display_b = ax_reconstructed_b.imshow(reconstructed_image_b, cmap='Blues', vmin=0, vmax=1)
        ax_reconstructed_b.axis('off')

        # Initialize reconstructed RGB image
        reconstructed_image_rgb = np.zeros((32, 32, 3))
        reconstructed_display_rgb = ax_reconstructed_rgb.imshow(reconstructed_image_rgb)
        ax_reconstructed_rgb.axis('off')
        reconstructed_displays_rgb.append(reconstructed_display_rgb)

        # Append to lists
        reconstructed_images_r.append(reconstructed_image_r)
        original_displays_r.append(ax_original_r)
        reconstructed_displays_r.append(reconstructed_display_r)

        reconstructed_images_g.append(reconstructed_image_g)
        original_displays_g.append(ax_original_g)
        reconstructed_displays_g.append(reconstructed_display_g)

        reconstructed_images_b.append(reconstructed_image_b)
        original_displays_b.append(ax_original_b)
        reconstructed_displays_b.append(reconstructed_display_b)

    # Adjusted parameters for sliders and button positions
    axcolor = 'lightgoldenrodyellow'
    slider_height = 0.03
    slider_width = 0.35
    spacing = 0.05
    start_height = 0.25

    # First column of sliders
    ax_n_filters = plt.axes([0.1, start_height, slider_width, slider_height], facecolor=axcolor)
    ax_filter_size = plt.axes([0.1, start_height - spacing, slider_width, slider_height], facecolor=axcolor)
    ax_std_dev_min = plt.axes([0.1, start_height - 2 * spacing, slider_width, slider_height], facecolor=axcolor)
    ax_std_dev_max = plt.axes([0.1, start_height - 3 * spacing, slider_width, slider_height], facecolor=axcolor)
    ax_threshold = plt.axes([0.1, start_height - 4 * spacing, slider_width, slider_height], facecolor=axcolor)

    # Second column of sliders
    ax_center_min = plt.axes([0.55, start_height, slider_width, slider_height], facecolor=axcolor)
    ax_center_max = plt.axes([0.55, start_height - spacing, slider_width, slider_height], facecolor=axcolor)
    ax_sparsity = plt.axes([0.55, start_height - 2 * spacing, slider_width, slider_height], facecolor=axcolor)
    ax_k = plt.axes([0.55, start_height - 3 * spacing, slider_width, slider_height], facecolor=axcolor)

    # Radio buttons for method selection
    ax_method = plt.axes([0.55, start_height - 5 * spacing, slider_width, slider_height], facecolor=axcolor)
    radio_method = RadioButtons(ax_method, ('topk_values', 'binary_topk'))

    # Create sliders
    _n_filters = 1000   
    s_n_filters = Slider(
        ax_n_filters, 'n_filters', 1, _n_filters, valinit=n_filters, valstep=1)
    s_filter_size = Slider(
        ax_filter_size, 'filter_size', 1, 15, valinit=filter_size, valstep=1)
    s_std_dev_min = Slider(
        ax_std_dev_min, 'std_dev_min', 0.1, 1.0, valinit=std_dev_min)
    s_std_dev_max = Slider(
        ax_std_dev_max, 'std_dev_max', 0.1, 1.0, valinit=std_dev_max)
    s_threshold = Slider(
        ax_threshold, 'Threshold', 0.0, 1.0, valinit=threshold)

    s_center_min = Slider(
        ax_center_min, 'center_min', 0.0, 7.0, valinit=center_min)
    s_center_max = Slider(
        ax_center_max, 'center_max', 0.0, 7.0, valinit=center_max)
    s_sparsity = Slider(
        ax_sparsity, 'sparsity', 0.0, 1.0, valinit=sparsity)
    s_k = Slider(ax_k, 'k', 1, _n_filters, valinit=k, valstep=1)

    method = ['topk_values']  # Use list for mutability in nested function

    # Define a function to update the reconstruction
    def update(val):
        # Get the hyperparameters from the sliders
        n_filters = int(s_n_filters.val)
        filter_size = int(s_filter_size.val)
        std_dev_min = s_std_dev_min.val
        std_dev_max = s_std_dev_max.val
        center_min = s_center_min.val
        center_max = s_center_max.val
        sparsity = s_sparsity.val
        k = int(s_k.val)
        threshold = s_threshold.val

        # Ensure std_dev_min <= std_dev_max
        if std_dev_min > std_dev_max:
            std_dev_min, std_dev_max = std_dev_max, std_dev_min
            s_std_dev_min.set_val(std_dev_min)
            s_std_dev_max.set_val(std_dev_max)

        # Ensure center_min <= center_max
        if center_min > center_max:
            center_min, center_max = center_max, center_min
            s_center_min.set_val(center_min)
            s_center_max.set_val(center_max)

        # Create the encoder with new parameters
        encoder = ImageEncoder(
            n_filters=n_filters,
            filter_size=filter_size,
            std_dev_range=(std_dev_min, std_dev_max),
            center_range=(center_min, center_max),
            sparsity=sparsity
        )

        # For each image, reconstruct and update the display
        for i in range(num_images):

            input_image_r = r_val_input_images[i]
            input_image_g = g_val_input_images[i]
            input_image_b = b_val_input_images[i]

            # Function to process each channel
            def process_channel(input_image):
                output = encoder(input_image)
                topk_values, topk_indices = topk_per_location(output, k)
                num_filters = output.shape[0]
                input_shape = input_image.shape

                if method[0] == 'topk_values':
                    # Current implementation using topk_values
                    mask = torch.zeros_like(output.permute(2, 1, 0))
                    masked_encoding = mask.scatter_(dim=2, index=topk_indices, src=topk_values)
                else:
                    # Using binary_topk_encoding with threshold
                    binary_encoding = binary_topk_encoding(topk_indices, topk_values, num_filters, threshold)
                    masked_encoding = binary_encoding

                reconstructed_image = reconstruct_from_binary_encoding(
                    masked_encoding, encoder, input_shape)
                reconstructed_image = reconstructed_image[:, 2:-2, 2:-2]  # Remove padding
                reconstructed_image = reconstructed_image.detach().numpy().squeeze(0)

                # Normalize reconstructed image
                reconstructed_image -= reconstructed_image.min()
                if reconstructed_image.max() != 0:
                    reconstructed_image /= reconstructed_image.max()

                return reconstructed_image

            # Process R, G, B channels
            reconstructed_image_r = process_channel(input_image_r)
            reconstructed_image_g = process_channel(input_image_g)
            reconstructed_image_b = process_channel(input_image_b)

            # Update the displays for individual channels
            reconstructed_displays_r[i].set_data(reconstructed_image_r)
            reconstructed_displays_g[i].set_data(reconstructed_image_g)
            reconstructed_displays_b[i].set_data(reconstructed_image_b)

            # Combine reconstructed channels into an RGB image
            reconstructed_image_rgb = np.stack(
                [reconstructed_image_r, reconstructed_image_g, reconstructed_image_b], axis=2)

            # Normalize the combined RGB image
            reconstructed_image_rgb -= reconstructed_image_rgb.min()
            if reconstructed_image_rgb.max() != 0:
                reconstructed_image_rgb /= reconstructed_image_rgb.max()

            # Update the display for the reconstructed RGB image
            reconstructed_displays_rgb[i].set_data(reconstructed_image_rgb)

        fig.canvas.draw_idle()

    # Register the update function with each slider
    s_n_filters.on_changed(update)
    s_filter_size.on_changed(update)
    s_std_dev_min.on_changed(update)
    s_std_dev_max.on_changed(update)
    s_center_min.on_changed(update)
    s_center_max.on_changed(update)
    s_sparsity.on_changed(update)
    s_k.on_changed(update)
    s_threshold.on_changed(update)

    # Function to handle method selection
    def method_func(label):
        method[0] = label
        update(None)

    radio_method.on_clicked(method_func)

    # Initial update to display reconstructed images
    update(None)

    # Add a button to load next images
    ax_next = plt.axes([0.45, 0.05, 0.1, 0.04])
    b_next = Button(ax_next, 'Next Images')

    def next_images(event):
        # Load next num_images images
        images = []
        labels = []
        for i in range(num_images):
            idx = (cifar_dataset.index + i) % len(cifar_dataset)
            image, label = cifar_dataset[idx]
            images.append(image)
            labels.append(label)
        cifar_dataset.index = (cifar_dataset.index + num_images) % len(cifar_dataset)

        r_val_input_images.clear()
        g_val_input_images.clear()
        b_val_input_images.clear()

        for i in range(num_images):
            r_channel, g_channel, b_channel = rgb_extractor.extract_rgb(images[i])
            r_input_image = r_channel.unsqueeze(0)
            g_input_image = g_channel.unsqueeze(0)
            b_input_image = b_channel.unsqueeze(0)
            r_val_input_images.append(r_input_image)
            g_val_input_images.append(g_input_image)
            b_val_input_images.append(b_input_image)

            # Update original image displays
            original_displays_r[i].imshow(r_val_input_images[i].squeeze(0).numpy(), cmap='Reds')
            original_displays_r[i].set_title(f'Red - Label: {labels[i]}')
            original_displays_g[i].imshow(g_val_input_images[i].squeeze(0).numpy(), cmap='Greens')
            original_displays_g[i].set_title('Green')
            original_displays_b[i].imshow(b_val_input_images[i].squeeze(0).numpy(), cmap='Blues')
            original_displays_b[i].set_title('Blue')

            # Update original RGB image display
            original_rgb_image = np.transpose(images[i].numpy(), (1, 2, 0))
            original_displays_rgb[i].imshow(original_rgb_image)
            original_displays_rgb[i].set_title('Original RGB')

        update(None)  # Reconstruct with new images

    b_next.on_clicked(next_images)

    plt.show()

if __name__ == '__main__':
    main()