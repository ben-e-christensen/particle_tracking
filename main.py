import numpy as np
import pandas as pd
import pims
import trackpy as tp
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
# NOTE: The user's uploaded file name is used here.
IMAGE_FILENAME = "1758927055861_frame_000000.jpg"
# Estimated particle diameter in pixels (must be an ODD integer)
ESTIMATED_DIAMETER = 15
# Change to minmass (no underscore) for compatibility.
# Decrease this to find dimmer features; increase to be more selective.
MIN_MASS_CUTOFF = 50 
# ---------------------

def run_trackpy_location():
    """
    Loads a single image, locates the particles using trackpy, and plots the results.
    """
    if not os.path.exists(IMAGE_FILENAME):
        print(f"Error: The image file '{IMAGE_FILENAME}' was not found.")
        print("Please ensure the file is in the same directory as this script.")
        # If the script is run in an environment without the image, we can't proceed.
        return
        
    print(f"Loading image: {IMAGE_FILENAME}")
    try:
        # pims.open() is the standard way to load image/video data for trackpy
        frames = pims.open(IMAGE_FILENAME)
        
        # Check if the load was successful and we got at least one frame
        if not frames:
            print("Error: Could not load frames from the file.")
            return

        # We are only using the first (and only) frame
        frame = frames[0]

        # Convert to grayscale if it's not already (trackpy works best on grayscale)
        if frame.ndim == 3:
            # Simple conversion to grayscale (average of RGB channels)
            frame = np.mean(frame, axis=2).astype(frame.dtype)
        
        print(f"Locating features with diameter={ESTIMATED_DIAMETER} and minmass={MIN_MASS_CUTOFF}...")
        
        # 1. Feature Finding (Locate)
        # FIX: Changed 'min_mass' to 'minmass' for older trackpy version compatibility.
        f = tp.locate(frame, ESTIMATED_DIAMETER, minmass=MIN_MASS_CUTOFF)
        
        print(f"\nFound {len(f)} particles.")
        
        # 2. Display Results
        
        # Create a figure and axis for plotting
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the original image
        ax.imshow(frame, cmap='gray')
        
        # Overlay the located features as yellow circles
        # FIX: Changed deprecated 'tp.plot_spots' to 'tp.annotate'.
        # tp.annotate is the modern and robust way to draw features onto the image.
        tp.annotate(f, frame, ax=ax)

        # Set plot title and hide axes ticks for clarity
        ax.set_title(f"Trackpy Feature Location: {len(f)} particles found")
        ax.axis('off')

        # Save and show the plot
        output_filename = "located_particles.png"
        plt.savefig(output_filename, bbox_inches='tight', dpi=300)
        print(f"Results saved to '{output_filename}'")
        plt.show()
        
        # Display the first few rows of the feature dataframe
        print("\n--- Feature Data Head ---")
        print(f[['x', 'y', 'mass', 'size', 'ecc']].head())

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        # Print a helpful message for common library issues
        if "No module named" in str(e):
            print("\nPlease ensure you have installed the required libraries:")
            print("pip install trackpy pims matplotlib pandas numpy")
        
if __name__ == '__main__':
    run_trackpy_location()
