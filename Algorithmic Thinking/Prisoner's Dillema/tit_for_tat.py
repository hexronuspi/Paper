import random
import matplotlib.pyplot as plt

def strategy_tit_for_tat(history):
    if len(history) == 0:
        return random.choice([True, False])  # Start with a random decision

    prev_outcome_A, prev_outcome_B = history[-1]
    return prev_outcome_B  # Tit-for-tat strategy: Copy opponent's last move

def simulate_game(num_simulations, num_iterations):
    avg_years_in_prison_A_list = []
    avg_years_in_prison_B_list = []

    for _ in range(num_simulations):
        total_years_in_prison_A = 0
        total_years_in_prison_B = 0
        history = []  # History of previous rounds

        for _ in range(num_iterations):
            admit_A = strategy_tit_for_tat(history)
            admit_B = strategy_tit_for_tat(history)

            if admit_A and not admit_B:  # Mr. A admits, Mr. B doesn't
                total_years_in_prison_A += 0
                total_years_in_prison_B += 3
            elif not admit_A and admit_B:  # Mr. A doesn't admit, Mr. B admits
                total_years_in_prison_A += 3
                total_years_in_prison_B += 0
            elif admit_A and admit_B:  # Both admit
                total_years_in_prison_A += 1
                total_years_in_prison_B += 1
            else:  # Both don't admit
                total_years_in_prison_A += 2
                total_years_in_prison_B += 2

            # Update the history with the current round's decisions
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

    # Plot the graph for Mr. A and Mr. B
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_A_list, label="Mr. A (Average)")
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_B_list, label="Mr. B (Average)")
    plt.xlabel("Simulation")
    plt.ylabel("Average Years in Prison")
    plt.legend()
    plt.show()
