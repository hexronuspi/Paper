import random
import matplotlib.pyplot as plt

def strategy_A(history):
    if len(history) == 0:
        return random.choice([True, False])  # Start with a random decision

    prev_outcome_A, prev_outcome_B = history[-1]
    if prev_outcome_A:  # Repeat the previous action if it resulted in a win
        return prev_outcome_A
    else:  # Change the action if it resulted in a loss
        return not prev_outcome_A

def strategy_B(history):
    if len(history) == 0:
        return random.choice([True, False])  # Start with a random decision

    prev_outcome_A, prev_outcome_B = history[-1]
    if prev_outcome_B:  # Repeat the previous action if it resulted in a win
        return prev_outcome_B
    else:  # Change the action if it resulted in a loss
        return not prev_outcome_B

def calculate_betrayal_probability(history):
    betrayals_A = sum(1 for outcome_A, _ in history if outcome_A)
    betrayals_B = sum(1 for _, outcome_B in history if outcome_B)
    total_iterations = len(history)

    probability_A = betrayals_A / total_iterations
    probability_B = betrayals_B / total_iterations

    return probability_A, probability_B

def simulate_game(num_simulations, num_iterations):
    avg_years_in_prison_A_list = []
    avg_years_in_prison_B_list = []
    betrayal_probabilities_A = []
    betrayal_probabilities_B = []

    for sim in range(num_simulations):
        total_years_in_prison_A = 0
        total_years_in_prison_B = 0
        history = []  # History of previous rounds

        for _ in range(num_iterations):
            admit_A = strategy_A(history)
            admit_B = strategy_B(history)

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

        betrayal_prob_A, betrayal_prob_B = calculate_betrayal_probability(history)
        betrayal_probabilities_A.append(betrayal_prob_A)
        betrayal_probabilities_B.append(betrayal_prob_B)

        avg_years_in_prison_A_list.append(avg_years_in_prison_A)
        avg_years_in_prison_B_list.append(avg_years_in_prison_B)

        print(f"Simulation {sim + 1}:")
        print(f"Probability of Betrayal for Mr. A: {betrayal_prob_A:.2f}")
        print(f"Probability of Betrayal for Mr. B: {betrayal_prob_B:.2f}")
        print(f"Average Years in Prison for Mr. A: {avg_years_in_prison_A:.2f}")
        print(f"Average Years in Prison for Mr. B: {avg_years_in_prison_B:.2f}")
        print()

    return avg_years_in_prison_A_list, avg_years_in_prison_B_list, betrayal_probabilities_A, betrayal_probabilities_B

if __name__ == "__main__":
    num_simulations = 100
    num_iterations = 100

    avg_years_in_prison_A_list, avg_years_in_prison_B_list, betrayal_probabilities_A, betrayal_probabilities_B = simulate_game(num_simulations, num_iterations)
    plt.figure(dpi=450)
    # Plot the graph for Mr. A and Mr. B
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_A_list, label="Mr. A (Average)")
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_B_list, label="Mr. B (Average)")
    plt.xlabel("Simulation")
    plt.ylabel("Average Years in Prison")
    plt.legend()
    plt.show()
