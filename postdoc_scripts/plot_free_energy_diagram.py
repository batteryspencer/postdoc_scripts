"""Plot free energy diagram for CH4 dehydrogenation on Pt(111) surface with solvent effects."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy as sp
from dataclasses import dataclass

@dataclass
class PlotConfig:
    label_fontsize: int = 16
    title_fontsize: int = 18
    tick_labelsize: int = 12
    legend_fontsize: int = 14
    tick_length_major: int = 8
    tick_width_major: int = 1
    width: float = 0.2

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

def plot_free_energy_diagram(states: dict[str, float],
                             barriers: dict[tuple[str, str], float],
                             positions: dict[str, float],
                             config: PlotConfig):
    """
    Plot free energy diagram given states, barriers, and positions.
    """
    fig, ax = plt.subplots()

    # Plot horizontal lines for states
    for state, energy in states.items():
        x = positions[state]
        ax.hlines(y=energy, xmin=x - config.width, xmax=x + config.width, colors='black')
        ax.text(x, energy - 0.1, f'  {state} ({energy:.2f} eV)',
                verticalalignment='bottom')

    # Plot barriers and transitions
    for (start, end), barrier in barriers.items():
        start_x, end_x = positions[start], positions[end]
        start_e, end_e = states[start], states[end]
        barrier_e = barrier + start_e

        x1, y1 = start_x + config.width, start_e
        x2, y2 = end_x - config.width, end_e
        sol = find_valid_coefficients(x1, y1, x2, y2, barrier_e)
        a, b, c = sol
        xs = np.linspace(x1, x2, 100)
        ys = a * xs**2 + b * xs + c
        ax.plot(xs, ys, 'gray', linestyle='--')
        xc = -b / (2*a)
        ax.scatter([xc], [barrier_e], color='red')
        ax.text(xc, barrier_e, f'  E$_a$={(barrier_e - start_e):.2f} eV',
                verticalalignment='bottom', color='red')

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel('Reaction Coordinate', fontsize=config.label_fontsize)
    ax.set_ylabel('Free Energy (eV)', fontsize=config.label_fontsize)
    ax.set_title('CH$_4$ to C Conversion on Pt(111)', fontsize=config.title_fontsize)
    plt.tick_params(axis='both', which='major',
                    labelsize=config.tick_labelsize,
                    length=config.tick_length_major,
                    width=config.tick_width_major)
    ax.set_xticks([])
    ax.set_ylim(-0.25, ax.get_ylim()[1] + 0.25)
    legend = ax.legend(['Solvent Phase'], loc='upper left')
    legend.get_texts()[0].set_color("green")
    return fig, ax

def main():
    # Data
    states = {
        "CH4*": 0.00,
        "CH3*": 0.46,
        "CH2*": 0.92,
        "CH*": 0.62,
        "C*": 0.84,
    }

    barriers = {
        ("CH4*", "CH3*"): 1.14,
        ("CH3*", "CH2*"): 1.02,
        ("CH2*", "CH*"): 0.49,
        ("CH*", "C*"): 0.91,
    }

    positions = {state: i for i, state in enumerate(states)}

    # Normalize energies
    ref = states["CH4*"]
    for key in states:
        states[key] -= ref

    config = PlotConfig()
    fig, ax = plot_free_energy_diagram(states, barriers, positions, config)
    plt.tight_layout()
    plt.savefig('free_energy_diagram_solvent_CH4_Pt111.png', dpi=300)

if __name__ == "__main__":
    main()
