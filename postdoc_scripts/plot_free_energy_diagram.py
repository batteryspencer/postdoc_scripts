"""Plot free energy diagram for CH4 dehydrogenation on Pt(111) surface with solvent effects."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sympy as sp
from dataclasses import dataclass

@dataclass
class PlotConfig:
    label_fontsize: int = 16
    title_fontsize: int = 17
    tick_labelsize: int = 12
    legend_fontsize: int = 14
    tick_length_major: int = 8
    tick_width_major: int = 1
    width: float = 0.2
    capsize: int = 4

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
                             forward_barriers: dict[tuple[str, str], float],
                             reverse_barriers: dict[tuple[str, str], float],
                             forward_errors: dict[tuple[str, str], float],
                             reverse_errors: dict[tuple[str, str], float],
                             positions: dict[str, float],
                             config: PlotConfig):
    """
    Plot free energy diagram given states, barriers, errors, and positions.
    """
    fig, ax = plt.subplots()

    # Plot horizontal lines for states with annotation
    for state, energy in states.items():
        x = positions[state]
        ax.plot([x - config.width, x + config.width], [energy, energy], color='black')
        ax.text(x - 0.2, energy - 0.2, f'  {state}\n{energy:.2f} eV',
                verticalalignment='bottom', horizontalalignment='left')

    # Plot barriers and transitions with error bars
    for (start, end) in forward_barriers:
        start_x, end_x = positions[start], positions[end]
        start_e, end_e = states[start], states[end]
        Ea = forward_barriers[(start, end)]
        Ea_err = forward_errors[(start, end)]
        Eb = reverse_barriers[(start, end)]
        Eb_err = reverse_errors[(start, end)]
        barrier_e = Ea + start_e

        x1, y1 = start_x + config.width, start_e
        x2, y2 = end_x - config.width, end_e
        sol = find_valid_coefficients(x1, y1, x2, y2, barrier_e)
        a, b, c = sol
        xs = np.linspace(x1, x2, 100)
        ys = a * xs**2 + b * xs + c
        ax.plot(xs, ys, 'gray', linestyle='--')
        xc = -b / (2*a)
        ax.text(xc, barrier_e + 0.12, f'E$_a$={Ea:.2f} eV',
                verticalalignment='bottom', horizontalalignment='center', color='red')
        ax.text(xc, barrier_e + 0.10, f'E$_b$={Eb:.2f} eV',
                verticalalignment='top', horizontalalignment='center', color='blue')

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
    # Define steps with energies and errors
    steps = [
        {"start":"CH4*","end":"CH3*","Ea":1.14,"Ea_err":0.04,"Eb":0.68,"Eb_err":0.04,"deltaG":0.46,"deltaG_err":0.05},
        {"start":"CH3*","end":"CH2*","Ea":1.03,"Ea_err":0.05,"Eb":0.57,"Eb_err":0.05,"deltaG":0.47,"deltaG_err":0.07},
        {"start":"CH2*","end":"CH*","Ea":0.51,"Ea_err":0.05,"Eb":0.78,"Eb_err":0.04,"deltaG":-0.27,"deltaG_err":0.06},
        {"start":"CH*","end":"C*","Ea":0.92,"Ea_err":0.05,"Eb":0.68,"Eb_err":0.07,"deltaG":0.24,"deltaG_err":0.08},
    ]

    # Build states dictionary
    states = {steps[0]["start"]: 0.0}
    for step in steps:
        start = step["start"]
        end = step["end"]
        states[end] = states[start] + step["deltaG"]

    # Build barrier and error dictionaries
    forward_barriers = {}
    reverse_barriers = {}
    forward_errors = {}
    reverse_errors = {}
    positions = {}

    # Assign positions based on order in states
    for i, state in enumerate(states):
        positions[state] = i

    for step in steps:
        key = (step["start"], step["end"])
        forward_barriers[key] = step["Ea"]
        forward_errors[key] = step["Ea_err"]
        reverse_barriers[key] = step["Eb"]
        reverse_errors[key] = step["Eb_err"]

    config = PlotConfig()
    fig, ax = plot_free_energy_diagram(states, forward_barriers, reverse_barriers,
                                       forward_errors, reverse_errors, positions, config)
    plt.tight_layout()
    plt.savefig('free_energy_diagram_solvent_CH4_Pt111.png', dpi=300)

if __name__ == "__main__":
    main()
