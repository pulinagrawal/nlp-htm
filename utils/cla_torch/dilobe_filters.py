import torch
import matplotlib.pyplot as plt
import numpy as np

def create_coordinate_grid(filter_size, n_filters):
    """Creates a coordinate grid for the filters."""
    x = torch.linspace(0, filter_size - 1, filter_size)
    y = torch.linspace(0, filter_size - 1, filter_size)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    x_grid = x_grid.unsqueeze(0).expand(n_filters, -1, -1)  # Shape: (n_filters, filter_size, filter_size)
    y_grid = y_grid.unsqueeze(0).expand(n_filters, -1, -1)
    return x_grid, y_grid

def generate_random_parameters(n_filters, filter_size, center_range, std_dev_range):
    """Generates random parameters for the positive and negative Gaussians."""
    params = {}
    for sign in ['p', 'n']:  # 'p' for positive, 'n' for negative Gaussians
        params[f'x0_{sign}'] = torch.rand(n_filters, 1, 1) * (center_range[1] - center_range[0]) + center_range[0]
        params[f'y0_{sign}'] = torch.rand(n_filters, 1, 1) * (center_range[1] - center_range[0]) + center_range[0]
        params[f'sx_{sign}'] = torch.rand(n_filters, 1, 1) * (std_dev_range[1] - std_dev_range[0]) + std_dev_range[0]
        params[f'sy_{sign}'] = torch.rand(n_filters, 1, 1) * (std_dev_range[1] - std_dev_range[0]) + std_dev_range[0]
        params[f'theta_{sign}'] = torch.rand(n_filters, 1, 1) * np.pi  # Angle between 0 and pi
    return params

def compute_gaussian(x_grid, y_grid, x0, y0, sx, sy, theta):
    """Computes a 2D Gaussian given grid coordinates and parameters."""
    x_diff = x_grid - x0
    y_diff = y_grid - y0
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rot = cos_theta * x_diff + sin_theta * y_diff
    y_rot = -sin_theta * x_diff + cos_theta * y_diff
    exponent = - ((x_rot ** 2) / (2 * sx ** 2) + (y_rot ** 2) / (2 * sy ** 2))
    G = torch.exp(exponent)
    return G

def create_filters(G_p, G_n):
    """Creates the filters by combining positive and negative Gaussians."""
    filters = G_p - G_n  # Shape: (n_filters, filter_size, filter_size)
    filters = filters.unsqueeze(1)  # Shape: (n_filters, 1, filter_size, filter_size)
    return filters

def normalize_filters(filters):
    """Normalizes filters for visualization purposes."""
    filters_vis = filters.clone()
    min_vals = torch.amin(filters_vis, dim=(2, 3), keepdim=True)
    filters_vis -= min_vals
    max_vals = torch.amax(filters_vis, dim=(2, 3), keepdim=True)
    filters_vis /= max_vals + 1e-8  # Adding epsilon to avoid division by zero
    return filters_vis

def visualize_filters(filters, n_cols=10):
    """Visualizes the filters in a grid layout."""
    n_filters = filters.shape[0]
    n_rows = n_filters // n_cols + int(n_filters % n_cols > 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = axes.flatten()

    for i in range(n_filters):
        ax = axes[i]
        ax.imshow(filters[i, 0].numpy(), cmap='gray')
        ax.axis('off')

    # Hide any remaining subplots if n_filters is not a multiple of n_cols
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def generate_dilobe_filters(n_filters=50, filter_size=8, center_range=(3, 5), std_dev_range=(.1, .9)):
    """Generates the dilobe filters using the defined functions."""
    x_grid, y_grid = create_coordinate_grid(filter_size, n_filters)
    params = generate_random_parameters(n_filters, filter_size, center_range, std_dev_range)

    # Compute the positive Gaussians
    G_p = compute_gaussian(
        x_grid, y_grid,
        params['x0_p'], params['y0_p'],
        params['sx_p'], params['sy_p'],
        params['theta_p']
    )

    # Compute the negative Gaussians
    G_n = compute_gaussian(
        x_grid, y_grid,
        params['x0_n'], params['y0_n'],
        params['sx_n'], params['sy_n'],
        params['theta_n']
    )

    # Create the filters
    filters = create_filters(G_p, G_n)
    return filters

def main():
    n_filters = 50
    filter_size = 8
    center_range = (2, 6)
    std_dev_range = (0.2, 0.8)
    filters = generate_dilobe_filters(n_filters, filter_size, center_range, std_dev_range)
    visualize_filters(filters)

if __name__ == '__main__':
    main()