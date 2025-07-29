import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy as sp

# Define font sizes and tick parameters as constants
LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
TICK_LABELSIZE = 14
LEGEND_FONTSIZE = 14
TICK_LENGTH_MAJOR = 8
TICK_WIDTH_MAJOR = 1

def find_coefficients(x1, y1, x2, y2, ymax):
    # Define the variables
    a, b, c = sp.symbols('a b c')
    
    # Define the equations
    eq1 = sp.Eq(y1, a*x1**2 + b*x1 + c)
    eq2 = sp.Eq(y2, a*x2**2 + b*x2 + c)
    eq3 = sp.Eq(c, ymax + b**2/(4*a))
    
    # Solve the system of equations
    solution = sp.solve((eq1, eq2, eq3), (a, b, c))
    
    # Return the solutions
    return solution

def find_valid_coefficients(x1, y1, x2, y2, ymax):
    solutions = find_coefficients(x1, y1, x2, y2, ymax)
    
    valid_solution = None
    for sol in solutions:
        a, b, c = sol
        xmax = -b / (2 * a)
        if x1 < xmax < x2 or x2 < xmax < x1:  # Check if xmax is between x1 and x2
            valid_solution = sol
            break
    
    return valid_solution


# Data
states = {
    "CH4*": 0.00,
    "CH3*": 0.46,
    "CH2*": 0.92,
    "CH*": 0.62,
    "C*": 0.84,
}

# Barriers
barriers = {
    ("CH4*", "CH3*"): 1.14,
    ("CH3*", "CH2*"): 1.02,
    ("CH2*", "CH*"): 0.49,
    ("CH*", "C*"): 0.91,
}

# Normalize energies to methane* energy
reference_energy = states["CH4*"]
for state in states:
    states[state] -= reference_energy

# Create figure and plot
fig, ax = plt.subplots()

# Horizontal positions and widths
positions = {
    "CH4*": 0,
    "CH3*": 1,
    "CH2*": 2,
    "CH*": 3,
    "C*": 4
}
width = 0.2  # Width of the horizontal lines

# Plot horizontal lines for each state
for state, energy in states.items():
    x_pos = positions[state]
    ax.hlines(y=energy, xmin=x_pos - width, xmax=x_pos + width, colors='black')

    # Label the lines
    ax.text(x_pos, energy - 0.1, f'  {state} ({energy:.2f} eV)', verticalalignment='bottom')

# Draw lines and barriers for transitions
transitions = [
    ("CH4*", "CH3*"),
    ("CH3*", "CH2*"),
    ("CH2*", "CH*"),
    ("CH*", "C*"),
]

colors = ['red', 'red', 'red', 'red']

for i, (start, end) in enumerate(transitions):
    start_pos = positions[start]
    end_pos = positions[end]
    start_energy = states[start]
    end_energy = states[end]

    if (start, end) in barriers:
        barrier_energy = barriers[(start, end)] + start_energy

        # The parabola should pass through the start and end points
        x1, y1 = start_pos + width, start_energy
        x2, y2 = end_pos - width, end_energy

        valid_solution = find_valid_coefficients(x1, y1, x2, y2, barrier_energy)
        a, b, c = valid_solution
        barrier_pos = -b / (2 * a)

        x = np.linspace(start_pos + width, end_pos - width, 100)
        y = a * x**2 + b * x + c

        ax.plot(x, y, 'gray', linestyle='--')
        ax.scatter([barrier_pos], [barrier_energy], color=colors[i])
        ax.text(barrier_pos, barrier_energy, f'  E$_a$={barrier_energy - start_energy:.2f} eV', verticalalignment='bottom', color=colors[i])
    else:
        # Plot a straight line if no barrier is provided
        ax.plot([start_pos + width, end_pos - width], [start_energy, end_energy], 'gray', linestyle='--')

# Set the y-axis ticks every 0.5
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

# Set labels and title
ax.set_xlabel('Reaction Coordinate', fontsize=LABEL_FONTSIZE)
ax.set_ylabel('Free Energy (eV)', fontsize=LABEL_FONTSIZE)
ax.set_title('CH$_4$ to C Conversion on Pt(111)', fontsize=TITLE_FONTSIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE, length=TICK_LENGTH_MAJOR, width=TICK_WIDTH_MAJOR)
ax.set_xticks([])
# ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.25, ax.get_ylim()[1] + 0.25)

legend = plt.gca().legend(['Solvent Phase'], loc='upper left')
legend.get_texts()[0].set_color("green")

plt.tight_layout()
plt.savefig('free_energy_diagram_solvent_CH4.png', dpi=300)

