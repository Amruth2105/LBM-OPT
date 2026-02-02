import numpy as np

def naca4(m, p, t, chord=100):
    """
    Generates coordinates for a NACA 4-digit airfoil.
    m: max camber (0.0 to 0.095)
    p: position of max camber (0.0 to 0.9)
    t: max thickness (0.0 to 0.30)
    chord: length in pixels
    """
    # Create x coordinates (0 to chord)
    x = np.linspace(0, chord, chord+1)
    
    # Thickness distribution equation (NACA standard)
    term1 = 0.2969 * np.sqrt(x/chord)
    term2 = -0.1260 * (x/chord)
    term3 = -0.3516 * (x/chord)**2
    term4 = 0.2843 * (x/chord)**3
    term5 = -0.1015 * (x/chord)**4
    yt = 5 * t * chord * (term1 + term2 + term3 + term4 + term5)
    
    # Camber line
    yc = np.zeros_like(x)
    if p > 0:
        # Front part of camber
        mask_front = x <= p*chord
        yc[mask_front] = (m / p**2) * (2*p*(x[mask_front]/chord) - (x[mask_front]/chord)**2)
        
        # Back part of camber
        mask_back = x > p*chord
        yc[mask_back] = (m / (1-p)**2) * ((1-2*p) + 2*p*(x[mask_back]/chord) - (x[mask_back]/chord)**2)
        
        yc = yc * chord

    # Upper and Lower surfaces
    yu = yc + yt
    yl = yc - yt
    
    return x, yu, yl

def get_mask(m, p, t, width, height, angle_of_attack=0):
    """
    Creates a boolean mask (True = Obstacle) for the LBM solver.
    Includes simple rotation for Angle of Attack.
    """
    chord_len = int(width * 0.2) # Blade is 20% of tunnel width
    x, yu, yl = naca4(m, p, t, chord=chord_len)
    
    mask = np.zeros((height, width), dtype=bool)
    
    # Position: Centered vertically, 1/4 from left
    offset_x = width // 4
    offset_y = height // 2
    
    # Simple rasterization
    for i in range(len(x)):
        # Apply Angle of Attack (simple vertical shift)
        # Note: True rotation requires matrix math, but this works for small angles
        rot_y = int(x[i] * np.sin(np.radians(angle_of_attack)))
        
        top = int(offset_y + yu[i] - rot_y)
        bot = int(offset_y + yl[i] - rot_y)
        col = int(offset_x + x[i])
        
        if 0 <= col < width:
            # Clip to screen bounds
            top = min(max(top, 0), height-1)
            bot = min(max(bot, 0), height-1)
            mask[bot:top, col] = True
            
    return mask
