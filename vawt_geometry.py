"""
VAWT Geometry Generator
Generates rotating blade geometry for Vertical Axis Wind Turbines (Darrieus type).
Uses NACA 4-digit symmetric profiles (e.g., NACA 0018).
"""
import numpy as np


def naca_symmetric(t, chord=100):
    """
    Generates coordinates for a symmetric NACA 4-digit airfoil (NACA 00XX).
    t: max thickness as fraction (e.g., 0.18 for NACA 0018)
    chord: length in pixels
    Returns: x, y_upper, y_lower coordinates
    """
    x = np.linspace(0, chord, int(chord) + 1)
    
    # NACA symmetric thickness distribution
    term1 = 0.2969 * np.sqrt(x / chord)
    term2 = -0.1260 * (x / chord)
    term3 = -0.3516 * (x / chord) ** 2
    term4 = 0.2843 * (x / chord) ** 3
    term5 = -0.1015 * (x / chord) ** 4
    yt = 5 * t * chord * (term1 + term2 + term3 + term4 + term5)
    
    return x, yt, -yt


def rotate_point(x, y, cx, cy, angle_rad):
    """
    Rotate point (x, y) around center (cx, cy) by angle_rad.
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_shifted = x - cx
    y_shifted = y - cy
    x_rot = x_shifted * cos_a - y_shifted * sin_a + cx
    y_rot = x_shifted * sin_a + y_shifted * cos_a + cy
    return x_rot, y_rot


def get_blade_positions(n_blades, radius, azimuth_deg, chord_length):
    """
    Get the center position and orientation of each blade.
    n_blades: number of blades
    radius: turbine radius (distance from center to blade center)
    azimuth_deg: current rotation angle of turbine (degrees)
    chord_length: blade chord length
    
    Returns: list of (center_x, center_y, blade_angle_rad) for each blade
    """
    positions = []
    azimuth_rad = np.radians(azimuth_deg)
    
    for i in range(n_blades):
        # Blades are equally spaced around the circle
        blade_angle = azimuth_rad + (2 * np.pi * i / n_blades)
        
        # Blade center position (on the circle of radius R)
        cx = radius * np.cos(blade_angle)
        cy = radius * np.sin(blade_angle)
        
        # Blade is oriented tangent to the circle (perpendicular to radius)
        # The chord line is tangent, so blade angle = blade_angle + 90°
        orientation = blade_angle + np.pi / 2
        
        positions.append((cx, cy, orientation))
    
    return positions


def get_vawt_mask(n_blades, thickness, radius, azimuth_deg, width, height, chord_fraction=0.25):
    """
    Creates a boolean mask (True = Obstacle) for VAWT blades at given azimuth.
    
    n_blades: number of blades (typically 2-5)
    thickness: NACA thickness parameter (e.g., 0.18 for NACA 0018)
    radius: turbine radius as fraction of domain width
    azimuth_deg: current rotation angle (degrees)
    width, height: domain dimensions
    chord_fraction: blade chord as fraction of diameter
    
    Returns: boolean mask array
    """
    mask = np.zeros((height, width), dtype=bool)
    
    # Convert to pixels - use larger scaling for visibility
    domain_size = min(width, height)
    R_pixels = int(radius * domain_size * 0.45)  # Radius in pixels (increased from 0.4)
    chord_pixels = max(15, int(chord_fraction * 2 * R_pixels))  # Minimum 15 pixels chord
    
    # Domain center
    center_x = width // 2
    center_y = height // 2
    
    # Get blade geometry (local coordinates) with finer resolution
    x_local, yu_local, yl_local = naca_symmetric(thickness, chord=chord_pixels)
    
    # Shift to center the blade at origin (center of chord)
    x_local = x_local - chord_pixels / 2
    
    # Get blade positions
    blade_positions = get_blade_positions(n_blades, R_pixels, azimuth_deg, chord_pixels)
    
    # Draw each blade using polygon filling approach
    for (bx, by, blade_angle) in blade_positions:
        # Create filled blade by iterating through chord positions
        for i in range(len(x_local)):
            # Get upper and lower y values for this x position
            y_top = yu_local[i]
            y_bot = yl_local[i]
            
            # Create vertical line of points from bottom to top of blade
            n_fill = max(3, int(abs(y_top - y_bot) * 2) + 1)
            for y_val in np.linspace(y_bot, y_top, n_fill):
                # Rotate the point
                x_rot, y_rot = rotate_point(x_local[i], y_val, 0, 0, blade_angle)
                
                # Translate to blade position in domain
                px = int(round(center_x + bx + x_rot))
                py = int(round(center_y + by + y_rot))
                
                if 0 <= px < width and 0 <= py < height:
                    mask[py, px] = True
    
    return mask



def get_blade_surface_points(n_blades, thickness, radius, azimuth_deg, width, height, chord_fraction=0.15):
    """
    Get the surface points of all blades for force calculation.
    Returns list of (x, y, nx, ny) where (nx, ny) is outward normal.
    """
    surface_points = []
    
    R_pixels = int(radius * min(width, height) * 0.4)
    chord_pixels = int(chord_fraction * 2 * R_pixels)
    
    center_x = width // 2
    center_y = height // 2
    
    x_local, yu_local, yl_local = naca_symmetric(thickness, chord=chord_pixels)
    x_local = x_local - chord_pixels / 2
    
    blade_positions = get_blade_positions(n_blades, R_pixels, azimuth_deg, chord_pixels)
    
    for blade_idx, (bx, by, blade_angle) in enumerate(blade_positions):
        # Upper surface (outward normal points up in local coords)
        for i in range(len(x_local) - 1):
            x_rot, y_rot = rotate_point(x_local[i], yu_local[i], 0, 0, blade_angle)
            px = center_x + bx + x_rot
            py = center_y + by + y_rot
            
            # Normal direction (rotated 90° from tangent)
            dx = x_local[i+1] - x_local[i]
            dy = yu_local[i+1] - yu_local[i]
            nx_local, ny_local = -dy, dx  # Perpendicular, pointing up
            norm = np.sqrt(nx_local**2 + ny_local**2) + 1e-10
            nx_local, ny_local = nx_local/norm, ny_local/norm
            
            # Rotate normal
            nx_rot, ny_rot = rotate_point(nx_local, ny_local, 0, 0, blade_angle)
            
            surface_points.append((px, py, nx_rot, ny_rot, blade_idx))
        
        # Lower surface (outward normal points down in local coords)
        for i in range(len(x_local) - 1):
            x_rot, y_rot = rotate_point(x_local[i], yl_local[i], 0, 0, blade_angle)
            px = center_x + bx + x_rot
            py = center_y + by + y_rot
            
            dx = x_local[i+1] - x_local[i]
            dy = yl_local[i+1] - yl_local[i]
            nx_local, ny_local = dy, -dx  # Perpendicular, pointing down
            norm = np.sqrt(nx_local**2 + ny_local**2) + 1e-10
            nx_local, ny_local = nx_local/norm, ny_local/norm
            
            nx_rot, ny_rot = rotate_point(nx_local, ny_local, 0, 0, blade_angle)
            
            surface_points.append((px, py, nx_rot, ny_rot, blade_idx))
    
    return surface_points


if __name__ == "__main__":
    # Test the geometry
    import matplotlib.pyplot as plt
    
    width, height = 400, 400
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for idx, azimuth in enumerate([0, 30, 60, 90]):
        ax = axes[idx // 2, idx % 2]
        mask = get_vawt_mask(n_blades=3, thickness=0.18, radius=0.5, 
                             azimuth_deg=azimuth, width=width, height=height)
        ax.imshow(mask, origin='lower', cmap='gray')
        ax.set_title(f"Azimuth = {azimuth}°")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    plt.tight_layout()
    plt.savefig("vawt_geometry_test.png", dpi=150)
    print("Saved vawt_geometry_test.png")
    plt.show()
