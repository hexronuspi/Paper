import random
import matplotlib.pyplot as plt

def strategy_A(history):
    for choice_A, choice_B in history:
        if not choice_B:  # If Mr. B ever defects, Mr. A will always defect from now on
            return False
    return True  # Mr. A always chooses to cooperate until Mr. B defects

def strategy_B(history):
    for choice_A, choice_B in history:
        if not choice_A:  # If Mr. A ever defects, Mr. B will always defect from now on
            return False
    return random.choice([True, False])  # Mr. B randomly decides to admit or not

def simulate_game(num_simulations, num_iterations):
    avg_years_in_prison_A_list = []
    avg_years_in_prison_B_list = []

    for _ in range(num_simulations):
        total_years_in_prison_A = 0
        total_years_in_prison_B = 0

        for _ in range(num_iterations):
            history = []  # History of previous rounds

            for _ in range(num_iterations):
                admit_A = strategy_A(history)
                admit_B = strategy_B(history)

                if admit_A and not admit_B:
                    total_years_in_prison_A += 0
                    total_years_in_prison_B += 3
                elif not admit_A and admit_B:
                    total_years_in_prison_A += 3
                    total_years_in_prison_B += 0
                elif admit_A and admit_B:
                    total_years_in_prison_A += 1
                    total_years_in_prison_B += 1
                else:
                    total_years_in_prison_A += 2
                    total_years_in_prison_B += 2

                history.append((admit_A, admit_B))

        avg_years_in_prison_A = total_years_in_prison_A / num_iterations
        avg_years_in_prison_B = total_years_in_prison_B / num_iterations

        avg_years_in_prison_A_list.append(avg_years_in_prison_A)
        avg_years_in_prison_B_list.append(avg_years_in_prison_B)

    return avg_years_in_prison_A_list, avg_years_in_prison_B_list

if __name__ == "__main__":
    num_simulations = 100
    num_iterations = 100

    avg_years_in_prison_A_list, avg_years_in_prison_B_list = simulate_game(num_simulations, num_iterations)
    plt.figure(dpi=450)

    # Plot the graph for Mr. A and Mr. B with y-values divided by 100
    plt.plot(range(1, num_simulations + 1), [y / 100 for y in avg_years_in_prison_A_list], label="Mr. A (Average)")
    plt.plot(range(1, num_simulations + 1), [y / 100 for y in avg_years_in_prison_B_list], label="Mr. B (Average)")
    plt.xlabel("Simulation")
    plt.ylabel("Average Years in Prison ")
    plt.legend()
    plt.show()
