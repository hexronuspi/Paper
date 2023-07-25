import random
import matplotlib.pyplot as plt

def strategy_A(probability):
    return random.random() < probability

def strategy_B(probability):
    return random.random() < probability

def simulate_game(num_simulations, num_iterations, prob_A, prob_B):
    avg_years_in_prison_A_list = []
    avg_years_in_prison_B_list = []

    for _ in range(num_simulations):
        total_years_in_prison_A = 0
        total_years_in_prison_B = 0

        for _ in range(num_iterations):
            admit_A = strategy_A(prob_A)
            admit_B = strategy_B(prob_B)

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

        avg_years_in_prison_A = total_years_in_prison_A / num_iterations
        avg_years_in_prison_B = total_years_in_prison_B / num_iterations

        avg_years_in_prison_A_list.append(avg_years_in_prison_A)
        avg_years_in_prison_B_list.append(avg_years_in_prison_B)

    return avg_years_in_prison_A_list, avg_years_in_prison_B_list

if __name__ == "__main__":
    num_simulations = 100
    num_iterations = 100

    prob_A_values = [i / 100 for i in range(101)]
    prob_B_values = [i / 100 for i in range(101)]

    min_avg_years = float('inf')
    best_prob_A = None
    best_prob_B = None

    for prob_A in prob_A_values:
        for prob_B in prob_B_values:
            avg_years_in_prison_A_list, avg_years_in_prison_B_list = simulate_game(num_simulations, num_iterations, prob_A, prob_B)
            avg_years_A = sum(avg_years_in_prison_A_list) / num_simulations
            avg_years_B = sum(avg_years_in_prison_B_list) / num_simulations
            if avg_years_A < min_avg_years or avg_years_B < min_avg_years:
                min_avg_years = min(avg_years_A, avg_years_B)
                best_prob_A = prob_A
                best_prob_B = prob_B

    print("Best Probability of Betrayal for Mr. A:", best_prob_A)
    print("Best Probability of Betrayal for Mr. B:", best_prob_B)

    # Plot the graph for Mr. A and Mr. B
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_A_list, label="Mr. A (Average)")
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_B_list, label="Mr. B (Average)")
    plt.xlabel("Simulation")
    plt.ylabel("Average Years in Prison")
    plt.legend()
    plt.show()
