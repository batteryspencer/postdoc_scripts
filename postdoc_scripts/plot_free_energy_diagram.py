import matplotlib.pyplot as plt
import numpy as np

# Data
states = {
    "propane*": -0.31,   # Pt:C3H8 binding energy
    "1-propyl*": 0.12,   # Pt:1-Propyl (C3H7) binding energy
    "2-propyl*": 0.11,   # Pt:2-Propyl (C3H7) binding energy
    "propylene*": -1.06  # Pt:C3H6_di-sigma-bonded binding energy
}

# Normalize energies to propane* energy
reference_energy = states["propane*"]
for state in states:
    states[state] -= reference_energy

# Create figure and plot
fig, ax = plt.subplots()

# Horizontal positions and widths
positions = {
    "propane*": 0,
    "1-propyl*": 1,
    "2-propyl*": 1,
    "propylene*": 2
}
width = 0.2  # Width of the horizontal lines

# Plot horizontal lines for each state
for state, energy in states.items():
    x_pos = positions[state]
    ax.hlines(y=energy, xmin=x_pos - width, xmax=x_pos + width, colors='black')

    # Label the lines
    ax.text(x_pos, energy, f'  {state} ({energy:.2f} eV)', verticalalignment='bottom')

# Draw lines for transitions (assuming no barriers for simplicity)
transitions = [
    ("propane*", "1-propyl*"),
    ("propane*", "2-propyl*"),
    ("1-propyl*", "propylene*"),
    ("2-propyl*", "propylene*")
]

# Draw transitions
for start, end in transitions:
    start_pos = positions[start]
    end_pos = positions[end]
    start_energy = states[start]
    end_energy = states[end]

    # Adjust for a parabolic curve if barriers were known
    # For now, using a straight line from endpoint to startpoint
    start_x = start_pos + width if end_pos > start_pos else start_pos - width
    end_x = end_pos - width if end_pos > start_pos else end_pos + width
    ax.plot([start_x, end_x], [start_energy, end_energy], 'gray', linestyle='--')

# Set labels and title
ax.set_xlabel('Reaction Coordinate')
ax.set_ylabel('Free Energy (eV)')
ax.set_title('Free Energy Diagram for Propane to Propylene Conversion')
ax.set_xticks([])
ax.set_xlim(-0.5, 2.5)

plt.tight_layout()
plt.savefig('free_energy_diagram.png', dpi=300)
