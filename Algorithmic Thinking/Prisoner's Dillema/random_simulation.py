import random
import matplotlib.pyplot as plt

def strategy_A():
    return random.choice([True, False])  # Randomly decide to admit or not

def strategy_B():
    return random.choice([True, False])  # Randomly decide to admit or not

def simulate_game(num_simulations, num_playoffs):
    avg_years_in_prison_A_list = []
    avg_years_in_prison_B_list = []

    for sim in range(num_simulations):
        total_years_in_prison_A = 0
        total_years_in_prison_B = 0
        total_betrayals_A = 0
        total_betrayals_B = 0

        for _ in range(num_playoffs):
            admit_A = strategy_A()
            admit_B = strategy_B()

            if admit_A and not admit_B:  # Mr. A admits, Mr. B doesn't
                total_years_in_prison_A += 0
                total_years_in_prison_B += 3
                total_betrayals_A += 1
            elif not admit_A and admit_B:  # Mr. A doesn't admit, Mr. B admits
                total_years_in_prison_A += 3
                total_years_in_prison_B += 0
                total_betrayals_B += 1
            elif admit_A and admit_B:  # Both admit
                total_years_in_prison_A += 1
                total_years_in_prison_B += 1
                total_betrayals_A += 1
                total_betrayals_B += 1
            else:  # Both don't admit
                total_years_in_prison_A += 2
                total_years_in_prison_B += 2

        avg_years_in_prison_A = total_years_in_prison_A / num_playoffs
        avg_years_in_prison_B = total_years_in_prison_B / num_playoffs
        prob_betray_A = total_betrayals_A / num_playoffs
        prob_betray_B = total_betrayals_B / num_playoffs

        avg_years_in_prison_A_list.append(avg_years_in_prison_A)
        avg_years_in_prison_B_list.append(avg_years_in_prison_B)

        print(f"Simulation {sim + 1}:")
        print(f"Probability of A betraying: {prob_betray_A:.2f}")
        print(f"Probability of B betraying: {prob_betray_B:.2f}")
        print(f"Average Years in Prison for A: {avg_years_in_prison_A:.2f}")
        print(f"Average Years in Prison for B: {avg_years_in_prison_B:.2f}")
        print()

    return avg_years_in_prison_A_list, avg_years_in_prison_B_list

if __name__ == "__main__":
    num_simulations = 100
    num_playoffs = 100

    avg_years_in_prison_A_list, avg_years_in_prison_B_list = simulate_game(num_simulations, num_playoffs)

    # Plot the results on a line graph
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_A_list, label="Mr. A (Average)")
    plt.plot(range(1, num_simulations + 1), avg_years_in_prison_B_list, label="Mr. B (Average)")
    plt.xlabel("Simulation")
    plt.ylabel("Average Years in Prison")
    plt.legend()
    plt.show()
