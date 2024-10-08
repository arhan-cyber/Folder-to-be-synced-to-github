import numpy as np
import matplotlib.pyplot as plt

# Define some complex numbers
complex_numbers = [-2 - 1j, -1 - 2j, -1 + 1j, -2 - 1j, 3 + 0j]

# Separate the real and imaginary parts
real_parts = [z.real for z in complex_numbers]
imaginary_parts = [z.imag for z in complex_numbers]

# Create the plot
plt.figure(figsize=(8, 6))

# Create a list of colors for each vector
colors = ['blue', 'orange', 'green', 'red', 'purple']

# Use quiver to plot vectors
plt.quiver(
    np.zeros(len(complex_numbers)),  # X origins (0 for all)
    np.zeros(len(complex_numbers)),  # Y origins (0 for all)
    real_parts,                     # X components (real parts)
    imaginary_parts,                # Y components (imaginary parts)
    angles='xy', scale_units='xy', scale=1, color=colors
)

# Annotate the points
for z in complex_numbers:
    plt.text(z.real, z.imag, f'{z}', fontsize=12, ha='right', color='black')

# Set the limits and labels
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.title('Complex Numbers as Vectors on the Argand Plane', fontsize=14)
plt.xlabel('Real Part', fontsize=12)
plt.ylabel('Imaginary Part', fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')

# Show the plot
plt.show()
