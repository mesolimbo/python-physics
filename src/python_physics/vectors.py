import numpy as np
import matplotlib
from matplotlib.patches import Arc
import matplotlib.pyplot as plt


# matplotlib.rcParams['text.usetex'] = False

def _angle_between_vectors(v1, v2):
    """Calculate the angle in radians between two vectors."""
    prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = prod / (norm_v1 * norm_v2)

    # Clamp cos_angle to [-1, 1] to handle floating-point precision errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)

def _calculate_projection(v1, v2):
    """Calculate the projection of v2 onto v1."""
    proj_scalar = np.dot(v2, v1) / np.dot(v1, v1)
    projection = proj_scalar * v1
    return projection

def _draw_projection_visualization(ax, v1, v2, projection):
    """Draw the projection vector and perpendicular line."""
    # Draw projection vector
    ax.quiver(0, 0, projection[0], projection[1],
             angles='xy', scale_units='xy', scale=1,
             color='gray', alpha=0.7,
             label='Projection of Vector 2 onto Vector 1', width=0.003)

    # Draw perpendicular line from v2 to projection
    ax.plot([v2[0], projection[0]], [v2[1], projection[1]],
           'k--', alpha=0.5, linewidth=1, label='Perpendicular')

    # Add annotation showing the projection length
    proj_length = np.linalg.norm(projection)
    mid_proj = projection / 2
    ax.annotate(f'Proj length: {proj_length:.2f}',
               xy=mid_proj, xytext=(mid_proj[0], mid_proj[1]-0.5),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

def _draw_angle_arc(ax, v1, v2, angle):
    """Draw an arc showing the angle between two vectors."""
    # Calculate angles for the arc - convert numpy scalars to Python float
    angle1 = float(np.degrees(np.arctan2(v1[1], v1[0])))
    angle2 = float(np.degrees(np.arctan2(v2[1], v2[0])))

    # Ensure we draw the smaller angle
    if angle2 < angle1:
        angle1, angle2 = angle2, angle1

    # Arc radius (scale with vector magnitudes)
    arc_radius = min(float(np.linalg.norm(v1)), float(np.linalg.norm(v2))) * 0.3

    # Draw the arc
    arc = Arc((0, 0), 2*arc_radius, 2*arc_radius,
             angle=0.0, theta1=angle1, theta2=angle2,
             color='purple', linewidth=2, alpha=0.8)
    ax.add_patch(arc)

    # Add angle text
    mid_angle = np.radians((angle1 + angle2) / 2)
    text_radius = arc_radius * 1.2
    ax.annotate(f'{angle:.1f}°',
               xy=(text_radius * np.cos(mid_angle), text_radius * np.sin(mid_angle)),
               fontsize=10, ha='center', va='center',
               color='purple', weight='bold')

def _setup_plot_axes(ax, vectors):
    """Configure plot axes, limits, and grid."""
    # Set equal aspect ratio and grid
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with some padding
    all_coords = np.array(vectors).flatten()
    max_coord = np.max(np.abs(all_coords))
    ax.set_xlim(-max_coord*0.2, max_coord*1.2)
    ax.set_ylim(-max_coord*0.2, max_coord*1.2)

def _get_plot_title(vectors, angle):
    """Create the plot title with dot product information."""
    dot_product = np.dot(vectors[0], vectors[1])
    v1_norm = np.linalg.norm(vectors[0])
    v2_norm = np.linalg.norm(vectors[1])

    title = 'Vector Visualization of Dot Product Geometry'
    title += f'\nDot Product: {dot_product} = |v1| × |v2| × cos(θ)'
    title += f'\n= {v1_norm:.2f} × {v2_norm:.2f} × cos({angle:.1f}°) = {dot_product}'

    return title

# Create a function to plot vectors
def plot_vectors_with_projection(*vectors, labels=None, colors=None, show_projection=True):
    """Plot multiple vectors with dot product projection visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, vector in enumerate(vectors):
        color = colors[i % len(colors)]
        label = labels[i] if labels else f'Vector {i+1}'

        # Plot vector as arrow from origin
        ax.quiver(0, 0, vector[0], vector[1],
                 angles='xy', scale_units='xy', scale=1,
                 color=color, label=label, width=0.005)

    angle = np.degrees(_angle_between_vectors(vectors[0], vectors[1]))

    # Show projection if we have exactly 2 vectors
    if show_projection and len(vectors) == 2:
        v1, v2 = vectors[0], vectors[1]

        # Calculate and draw projection
        projection = _calculate_projection(v1, v2)
        _draw_projection_visualization(ax, v1, v2, projection)

        # Draw angle arc
        _draw_angle_arc(ax, v1, v2, angle)

    # Setup axes and create title
    _setup_plot_axes(ax, vectors)
    title = _get_plot_title(vectors, angle)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
