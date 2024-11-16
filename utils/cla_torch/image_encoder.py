import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from dilobe_filters import generate_dilobe_filters
from matplotlib.widgets import Slider, Button

# Include all the provided functions here (create_coordinate_grid, generate_random_parameters, etc.)
# For brevity, I'm assuming those functions are already defined as per your code.

# Existing Encoder class and functions
class ImageEncoder(torch.nn.Module):
    def __init__(self, n_filters=50, filter_size=8, std_dev_range=(0.2, 0.4), center_range=(3, 5), sparsity=0.7, device='cpu'):
        super(ImageEncoder, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        filters = generate_dilobe_filters(n_filters, filter_size, std_dev_range, center_range)
        self.register_buffer('filters', filters)  # Store filters as a buffer (non-trainable)

    def forward(self, x):
        # x: [in_channels, height, width]
        in_channels, height, width = x.shape
        n_filters, _, filter_size, _ = self.filters.shape

        # Prepare filters for grouped convolution
        # Repeat filters for each input channel
        filters = self.filters.repeat(in_channels, 1, 1, 1)  # Shape: [in_channels * n_filters, 1, filter_size, filter_size]

        # Reshape input to add batch dimension
        x = x.unsqueeze(0)  # Shape: [1, in_channels, height, width]

        # Perform convolution with groups=in_channels
        output = F.conv2d(x, filters, bias=None, stride=1, padding=0, groups=in_channels)

        # Remove batch dimension
        output = output.squeeze(0)  # Shape: [in_channels * n_filters, output_height, output_width]

        return output

class ImageBinEncoder(ImageEncoder):
  def __init__(self, n_filters=50, filter_size=8, std_dev_range=(0.2, 0.4), center_range=(3, 5),
               dilobe_sp=0.7, k=2):
      super().__init__(n_filters, filter_size, std_dev_range, center_range, dilobe_sp)
      self.k = k

  def forward(self, image):
    output = super().forward(image)
    _, topk_ind = self.topk_per_location(output, self.k)
    return self.binary_topk_encoding(topk_ind, self.n_filters)


  def topk_per_location(self, output, k):
      """
      Selects the top-k filters per spatial location over the filter dimension.

      Args:
          output (torch.Tensor): Output tensor from the encoder of shape [num_filters, H, W]
          k (int): Number of top filters to select per spatial location.

      Returns:
          topk_values (torch.Tensor): Top-k activation values per spatial location, shape [H, W, k]
          topk_indices (torch.Tensor): Indices of the top-k filters per spatial location, shape [H, W, k]
      """
      num_filters, H, W = output.shape
      # Permute dimensions to [H, W, num_filters]
      output = output.permute(1, 2, 0)  # Shape: [H, W, num_filters]
      # Apply topk over the filter dimension
      topk_values, topk_indices = torch.topk(output, k=k, dim=2)
      return topk_values, topk_indices

def binary_topk_encoding(topk_indices, topk_values, num_filters, threshold):
    """
    Creates a binary representation of the top-k encoding output, setting indices to 1
    only if the corresponding activation values are greater than the threshold.

    Args:
        topk_indices (torch.Tensor): Indices of the top-k filters per spatial location, shape [H, W, k]
        topk_values (torch.Tensor): Activation values of the top-k filters per spatial location, shape [H, W, k]
        num_filters (int): Total number of filters (in_channels * n_filters)
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
    binary_encoding_flat[positions[:, 0], indices_to_set] = 1

    # Reshape back to [H, W, num_filters]
    binary_encoding = binary_encoding_flat.view(H, W, num_filters)
    return binary_encoding

def reconstruct_from_binary_encoding(binary_encoding, encoder, input_shape):
    """
    Reconstructs the input image from the binary encoding using transposed convolution.

    Args:
        binary_encoding (torch.Tensor): Binary tensor of shape [H, W, num_filters]
        encoder (Encoder): The encoder object containing the filters
        input_shape (tuple): The shape of the original input image (in_channels, height, width)

    Returns:
        reconstructed_image (torch.Tensor): Reconstructed image of shape [in_channels, height, width]
    """
    H, W, num_filters = binary_encoding.shape
    in_channels, height, width = input_shape
    n_filters = encoder.filters.shape[0]
    filter_size = encoder.filters.shape[2]

    # Reshape binary encoding to match expected input shape for conv_transpose2d
    binary_encoding = binary_encoding.permute(2, 0, 1)  # Shape: [num_filters, H, W]
    binary_encoding = binary_encoding.unsqueeze(0)       # Shape: [1, num_filters, H, W]

    # Prepare filters for transposed convolution
    filters = encoder.filters  # Shape: [n_filters, 1, filter_size, filter_size]

    # Perform transposed convolution
    reconstructed = F.conv_transpose2d(
        binary_encoding, filters, bias=None, stride=1, padding=0, groups=1
    )  # Output shape: [1, 1, height + filter_size - 1, width + filter_size - 1]

    # Remove batch dimension
    reconstructed_image = reconstructed.squeeze(0)  # Shape: [1, output_height, output_width]

    # Crop the reconstructed image to match the original input size
    start_h = filter_size//2 - 1
    start_w = filter_size//2 - 1
    reconstructed_image = reconstructed_image[:, start_h:start_h + height, start_w:start_w + width]

    return reconstructed_image

def topk_per_location(output, k):
    """
    Selects the top-k filters per spatial location over the filter dimension.

    Args:
        output (torch.Tensor): Output tensor from the encoder of shape [num_filters, H, W]
        k (int): Number of top filters to select per spatial location.

    Returns:
        topk_values (torch.Tensor): Top-k activation values per spatial location, shape [H, W, k]
        topk_indices (torch.Tensor): Indices of the top-k filters per spatial location, shape [H, W, k]
    """
    num_filters, H, W = output.shape
    # Permute dimensions to [H, W, num_filters]
    output = output.permute(1, 2, 0)  # Shape: [H, W, num_filters]
    # Apply topk over the filter dimension
    topk_values, topk_indices = torch.topk(output, k=k, dim=2)
    
    return topk_values, topk_indices

def binary_topk_encoding(topk_indices, num_filters):
    """
    Creates a binary representation of the top-k encoding output.

    Args:
        topk_indices (torch.Tensor): Indices of the top-k filters per spatial location, shape [H, W, k]
        num_filters (int): Total number of filters (in_channels * n_filters)

    Returns:
        binary_encoding (torch.Tensor): Binary tensor indicating presence of top-k filters at each location,
                                        shape [H, W, num_filters], dtype torch.float32
    """
    H, W, k = topk_indices.shape
    # Flatten H and W dimensions
    topk_indices_flat = topk_indices.view(-1, k)  # Shape: [H*W, k]
    # Create binary encoding
    binary_encoding_flat = torch.zeros((H*W, num_filters), dtype=torch.float32)
    # Use scatter to set the top-k indices to 1
    binary_encoding_flat.scatter_(dim=1, index=topk_indices_flat, value=1)
    # Reshape back to [H, W, num_filters]
    binary_encoding = binary_encoding_flat.view(H, W, num_filters)
    return binary_encoding


def main():
    # Load MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    mnist_dataset.index = 0  # Initialize index

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
        idx = (mnist_dataset.index + i) % len(mnist_dataset)
        image, label = mnist_dataset[idx]
        images.append(image)
        labels.append(label)
    mnist_dataset.index = (mnist_dataset.index + num_images) % len(mnist_dataset)

    input_images = []
    for image in images:
        input_image = image.squeeze(0)  # Shape: [28, 28]
        input_image = F.pad(input_image, (2, 2, 2, 2))  # Pad to 32x32
        input_image = input_image.unsqueeze(0)  # Shape: [1, 32, 32]
        input_images.append(input_image)

    # Set up the figure and axes
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 6))
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.8, hspace=0.2)

    # Display the original images
    reconstructed_images = []
    original_displays = []
    reconstructed_displays = []
    for i in range(num_images):
        ax_original = axes[0, i]
        ax_reconstructed = axes[1, i]

        ax_original.imshow(input_images[i].squeeze(0).numpy(), cmap='gray')
        ax_original.set_title(f'Label: {labels[i]}')
        ax_original.axis('off')

        # Initialize reconstructed image with zeros
        reconstructed_image = np.zeros((28, 28))
        reconstructed_display = ax_reconstructed.imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
        ax_reconstructed.axis('off')

        reconstructed_images.append(reconstructed_image)
        original_displays.append(ax_original)
        reconstructed_displays.append(reconstructed_display)

    # Adjusted parameters for sliders and button positions
    axcolor = 'lightgoldenrodyellow'
    slider_height = 0.03
    slider_width = 0.35
    spacing = 0.05
    start_height = 0.55

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
            input_image = input_images[i]
            output = encoder(input_image)
            topk_values, topk_indices = topk_per_location(output, k)
            num_filters = output.shape[0]
            mask = torch.zeros_like(output.permute(2,1,0))
            # Place the top k values into the mask at the correct positions
            masked_encoding = mask.scatter_(dim=2, index=topk_indices, src=topk_values)
            input_shape = input_image.shape
            reconstructed_image = reconstruct_from_binary_encoding(
                masked_encoding, encoder, input_shape)
            reconstructed_image = reconstructed_image[:, 2:-2, 2:-2]  # Remove padding
            reconstructed_image = reconstructed_image.detach().numpy().squeeze(0)
            reconstructed_image = np.clip(reconstructed_image, 0, 1)

            # Update the display
            reconstructed_displays[i].set_data(reconstructed_image)
            reconstructed_displays[i].set_clim(0, 1)  # Update color limits if necessary

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
            idx = (mnist_dataset.index + i) % len(mnist_dataset)
            image, label = mnist_dataset[idx]
            images.append(image)
            labels.append(label)
        mnist_dataset.index = (mnist_dataset.index + num_images) % len(mnist_dataset)

        input_images.clear()
        for i in range(num_images):
            input_image = images[i].squeeze(0)  # Shape: [28, 28]
            input_image = F.pad(input_image, (2, 2, 2, 2))  # Pad to 32x32
            input_image = input_image.unsqueeze(0)  # Shape: [1, 32, 32]
            input_images.append(input_image)

            # Update original image display
            original_displays[i].imshow(input_images[i].squeeze(0).numpy(), cmap='gray')
            original_displays[i].set_title(f'Label: {labels[i]}')

        update(None)  # Reconstruct with new images

    b_next.on_clicked(next_images)

    plt.show()

def viz_main():
    # Load MNIST test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Initialize the encoder
    encoder = ImageBinEncoder(n_filters=50, filter_size=8)
    k = 5  # Number of top filters to select

    # Prepare lists to store images and labels
    original_images = []
    reconstructed_images = []
    labels = []

    # Process and reconstruct the first 10 images
    for idx in range(10):
        image, label = mnist_dataset[idx]  # Get the image and label
        labels.append(label)

        # Preprocess the image
        input_image = image.squeeze(0)  # Shape: [28, 28]
        input_image = F.pad(input_image, (2, 2, 2, 2))  # Pad to 32x32
        input_image = input_image.unsqueeze(0)  # Shape: [1, 32, 32]

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