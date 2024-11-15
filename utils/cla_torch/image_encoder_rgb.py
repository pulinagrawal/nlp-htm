import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from matplotlib.widgets import Slider, Button

from image_encoder import ImageBinEncoder, ImageEncoder, reconstruct_from_binary_encoding, topk_per_location, binary_topk_encoding

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
    # Load MNIST test dataset
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

    num_images = 10  # Number of images to display

    # Get the first num_images images
    images = []
    labels = []
    for i in range(num_images):
        idx = (cifar_dataset.index + i) % len(cifar_dataset)
        image, label = cifar_dataset[idx]
        images.append(image)
        labels.append(label)
    cifar_dataset.index = (cifar_dataset.index + num_images) % len(cifar_dataset)

    # Seperating the R G B Values
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

    # Set up the figure and axes
    fig, axes = plt.subplots(6, num_images, figsize=(num_images * 2, 6))
    plt.subplots_adjust(left=0.05, bottom=0.3, right=0.95, top=0.9, wspace=0.8, hspace=0.5)

    # Display the original images
    reconstructed_images_r, reconstructed_images_g, reconstructed_images_b = [], [], []
    original_displays_r, original_displays_g, original_displays_b = [], [], []
    reconstructed_displays_r, reconstructed_displays_g, reconstructed_displays_b = [], [], []
    for i in range(num_images):
        
        ax_original_r = axes[0, i]
        ax_reconstructed_r = axes[1, i]
        ax_original_g = axes[2, i]
        ax_reconstructed_g = axes[3, i]
        ax_original_b = axes[4, i]
        ax_reconstructed_b = axes[5, i]

        ax_original_r.imshow(r_val_input_images[i].squeeze(0).numpy(), cmap='Reds')
        ax_original_r.set_title(f'Red - Label: {labels[i]}')
        ax_original_r.axis('off')
        ax_original_g.imshow(g_val_input_images[i].squeeze(0).numpy(), cmap='Greens')
        ax_original_g.set_title(f'Green')
        ax_original_g.axis('off')
        ax_original_b.imshow(b_val_input_images[i].squeeze(0).numpy(), cmap='Blues')
        ax_original_b.set_title(f'Blue')
        ax_original_b.axis('off')


        # Initialize reconstructed image with zeros
        reconstructed_image_r = np.zeros((28, 28))
        reconstructed_display_r = ax_reconstructed_r.imshow(reconstructed_image_r, cmap='gray', vmin=0, vmax=1)
        ax_reconstructed_r.axis('off')
        reconstructed_image_g = np.zeros((28, 28))
        reconstructed_display_g = ax_reconstructed_g.imshow(reconstructed_image_g, cmap='gray', vmin=0, vmax=1)
        ax_reconstructed_g.axis('off')
        reconstructed_image_b = np.zeros((28, 28))
        reconstructed_display_b = ax_reconstructed_b.imshow(reconstructed_image_b, cmap='gray', vmin=0, vmax=1)
        ax_reconstructed_b.axis('off')
        
        
        
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
    start_height = 0.2

    # First column of sliders
    ax_n_filters = plt.axes([0.1, start_height, slider_width, slider_height], facecolor=axcolor)
    ax_filter_size = plt.axes([0.1, start_height - spacing, slider_width, slider_height], facecolor=axcolor)
    ax_std_dev_min = plt.axes([0.1, start_height - 2 * spacing, slider_width, slider_height], facecolor=axcolor)
    ax_std_dev_max = plt.axes([0.1, start_height - 3 * spacing, slider_width, slider_height], facecolor=axcolor)

    # Second column of sliders
    ax_center_min = plt.axes([0.55, start_height, slider_width, slider_height], facecolor=axcolor)
    ax_center_max = plt.axes([0.55, start_height - spacing, slider_width, slider_height], facecolor=axcolor)
    ax_sparsity = plt.axes([0.55, start_height - 2 * spacing, slider_width, slider_height], facecolor=axcolor)
    ax_k = plt.axes([0.55, start_height - 3 * spacing, slider_width, slider_height], facecolor=axcolor)

    # Create sliders
    s_n_filters = Slider(
        ax_n_filters, 'n_filters', 10, 100, valinit=n_filters, valstep=1)
    s_filter_size = Slider(
        ax_filter_size, 'filter_size', 3, 15, valinit=filter_size, valstep=1)
    s_std_dev_min = Slider(
        ax_std_dev_min, 'std_dev_min', 0.1, 1.0, valinit=std_dev_min)
    s_std_dev_max = Slider(
        ax_std_dev_max, 'std_dev_max', 0.1, 1.0, valinit=std_dev_max)
    s_center_min = Slider(
        ax_center_min, 'center_min', 0.0, 7.0, valinit=center_min)
    s_center_max = Slider(
        ax_center_max, 'center_max', 0.0, 7.0, valinit=center_max)
    s_sparsity = Slider(
        ax_sparsity, 'sparsity', 0.0, 1.0, valinit=sparsity)
    s_k = Slider(ax_k, 'k', 1, 20, valinit=k, valstep=1)

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
            output_r = encoder(input_image_r)
            output_g = encoder(input_image_g)
            output_b = encoder(input_image_b)
            topk_values_r, topk_indices_r = topk_per_location(output_r, k)
            topk_values_g, topk_indices_g = topk_per_location(output_g, k)
            topk_values_b, topk_indices_b = topk_per_location(output_b, k)
            
            mask_r = torch.zeros_like(output_r.permute(2,1,0))
            masked_encoding_r = mask_r.scatter_(dim=2, index=topk_indices_r, src=topk_values_r) 
            mask_g = torch.zeros_like(output_r.permute(2,1,0))
            masked_encoding_g = mask_g.scatter_(dim=2, index=topk_indices_g, src=topk_values_g) 
            mask_b = torch.zeros_like(output_r.permute(2,1,0))
            masked_encoding_b = mask_b.scatter_(dim=2, index=topk_indices_b, src=topk_values_b)  
                                  
            input_shape_r = input_image_r.shape
            input_shape_g = input_image_g.shape
            input_shape_b = input_image_b.shape
            
            reconstructed_image_r = reconstruct_from_binary_encoding(
                masked_encoding_r, encoder, input_shape_r)
            reconstructed_image_g = reconstruct_from_binary_encoding(
                masked_encoding_g, encoder, input_shape_g)
            reconstructed_image_b = reconstruct_from_binary_encoding(
                masked_encoding_b, encoder, input_shape_b)

            reconstructed_image_r = reconstructed_image_r[:, 2:-2, 2:-2]  # Remove padding
            reconstructed_image_r = reconstructed_image_r.detach().numpy().squeeze(0)
            reconstructed_image_r = np.clip(reconstructed_image_r, 0, 1)
            
            reconstructed_image_g = reconstructed_image_g[:, 2:-2, 2:-2]  # Remove padding
            reconstructed_image_g = reconstructed_image_g.detach().numpy().squeeze(0)
            reconstructed_image_g = np.clip(reconstructed_image_g, 0, 1)
            
            reconstructed_image_b = reconstructed_image_b[:, 2:-2, 2:-2]  # Remove padding
            reconstructed_image_b = reconstructed_image_b.detach().numpy().squeeze(0)
            reconstructed_image_b = np.clip(reconstructed_image_b, 0, 1)

            # Update the display
            reconstructed_displays_r[i].set_data(reconstructed_image_r)
            reconstructed_displays_r[i].set_clim(0, 1)  # Update color limits if necessary
            reconstructed_displays_g[i].set_data(reconstructed_image_g)
            reconstructed_displays_g[i].set_clim(0, 1)  # Update color limits if necessary
            reconstructed_displays_b[i].set_data(reconstructed_image_b)
            reconstructed_displays_b[i].set_clim(0, 1)  # Update color limits if necessary

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
        
            # Update original image display
            original_displays_r[i].imshow(r_val_input_images[i].squeeze(0).numpy(), cmap='Reds')
            original_displays_r[i].set_title(f'Red - Label: {labels[i]}')
            original_displays_g[i].imshow(g_val_input_images[i].squeeze(0).numpy(), cmap='Greens')
            original_displays_g[i].set_title(f'Green')
            original_displays_b[i].imshow(b_val_input_images[i].squeeze(0).numpy(), cmap='Blues')
            original_displays_b[i].set_title(f'Blue')

        update(None)  # Reconstruct with new images

    b_next.on_clicked(next_images)

    plt.show()
    
def viz_main():
    # Load CIFAR Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    cifar_dataset.index = 0

    # Initialize the encoder
    encoder = ImageBinEncoder(n_filters=50, filter_size=2, k=3)

    # Prepare lists to store images and labels
    original_images = []
    reconstructed_images = []
    labels = []

    # Process and reconstruct the first 10 images
    for idx in range(10):
        image, label = cifar_dataset[idx]  # Get the image and label
        labels.append(label)

        # Preprocess the image
        rgb_extractor = RGBExtractor()
    
        r_channel, g_channel, b_channel = rgb_extractor.extract_rgb(image)
        r_input_image = r_channel.unsqueeze(0)
        g_input_image = g_channel.unsqueeze(0)
        b_input_image = b_channel.unsqueeze(0)

        input_image = r_input_image

        # Store the original image (after padding)
        original_images.append(input_image.squeeze(0).numpy())

        # Encode the image
        output = encoder(input_image)

        # Reconstruct the input from the binary encoding
        input_shape = input_image.shape  # (1, 32, 32)
        reconstructed_image = reconstruct_from_binary_encoding(output, encoder, input_shape)

        # Remove padding to compare with original image
        reconstructed_image = reconstructed_image[:, 2:-2, 2:-2]  # Remove padding

        # Store the reconstructed image
        reconstructed_images.append(reconstructed_image.squeeze(0).detach().numpy())

    # Visualization
    num_images = len(original_images)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

    for i in range(num_images):
        # Original image
        axes[0, i].imshow(original_images[i], cmap='gray')
        axes[0, i].set_title(f'Label: {labels[i]}')
        axes[0, i].axis('off')

        # Reconstructed image
        axes[1, i].imshow(reconstructed_images[i], cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original', fontsize=14)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=14)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()