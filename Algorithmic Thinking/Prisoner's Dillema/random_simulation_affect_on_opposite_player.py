import random
import matplotlib.pyplot as plt

def strategy_A():
    return random.choice([True, False])  # Randomly decide to admit or not

def strategy_B():
    return random.choice([True, False])  # Randomly decide to admit or not

def simulate_game(num_simulations, num_playoffs):
    prob_betray_A_list = []
    prob_betray_B_list = []
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

        prob_betray_A = total_betrayals_A / num_playoffs
        prob_betray_B = total_betrayals_B / num_playoffs
        avg_years_in_prison_A = total_years_in_prison_A / num_playoffs
        avg_years_in_prison_B = total_years_in_prison_B / num_playoffs

        prob_betray_A_list.append(prob_betray_A)
        prob_betray_B_list.append(prob_betray_B)
        avg_years_in_prison_A_list.append(avg_years_in_prison_A)
        avg_years_in_prison_B_list.append(avg_years_in_prison_B)

    return prob_betray_A_list, prob_betray_B_list, avg_years_in_prison_A_list, avg_years_in_prison_B_list

if __name__ == "__main__":
    num_simulations = 100
    num_playoffs = 100

    prob_betray_A_list, prob_betray_B_list, avg_years_in_prison_A_list, avg_years_in_prison_B_list = simulate_game(num_simulations, num_playoffs)

    # Plot Probability of Betrayal vs Prison Time for B
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(prob_betray_A_list, avg_years_in_prison_B_list, 'o')
    plt.xlabel("Probability of A Betraying")
    plt.ylabel("Prison Time for B")
    plt.title("A's Betrayal vs B's Prison Time")
    plt.axhline(y=1.50, color='gray', linestyle='--')
    plt.axvline(x=0.5, color='gray', linestyle='--')

    # Plot Probability of Betrayal vs Prison Time for A
    plt.subplot(1, 2, 2)
    plt.plot(prob_betray_B_list, avg_years_in_prison_A_list, 'o', color='orange')
    plt.xlabel("Probability of B Betraying")
    plt.ylabel("Prison Time for A")
    plt.title("B's Betrayal vs A's Prison Time")
    plt.axhline(y=1.50, color='gray', linestyle='--')
    plt.axvline(x=0.5, color='gray', linestyle='--')

    plt.tight_layout()
    plt.show()

    # Count number of dots in each quadrant for both graphs
    count_quadrant1_A = sum(1 for prob_A, prison_B in zip(prob_betray_A_list, avg_years_in_prison_B_list) if prob_A >= 0.5 and prison_B >= 1.50)
    count_quadrant2_A = sum(1 for prob_A, prison_B in zip(prob_betray_A_list, avg_years_in_prison_B_list) if prob_A < 0.5 and prison_B >= 1.50)
    count_quadrant3_A = sum(1 for prob_A, prison_B in zip(prob_betray_A_list, avg_years_in_prison_B_list) if prob_A >= 0.5 and prison_B < 1.50)
    count_quadrant4_A = sum(1 for prob_A, prison_B in zip(prob_betray_A_list, avg_years_in_prison_B_list) if prob_A < 0.5 and prison_B < 1.50)

    count_quadrant1_B = sum(1 for prob_B, prison_A in zip(prob_betray_B_list, avg_years_in_prison_A_list) if prob_B >= 0.5 and prison_A >= 1.50)
    count_quadrant2_B = sum(1 for prob_B, prison_A in zip(prob_betray_B_list, avg_years_in_prison_A_list) if prob_B < 0.5 and prison_A >= 1.50)
    count_quadrant3_B = sum(1 for prob_B, prison_A in zip(prob_betray_B_list, avg_years_in_prison_A_list) if prob_B >= 0.5 and prison_A < 1.50)
    count_quadrant4_B = sum(1 for prob_B, prison_A in zip(prob_betray_B_list, avg_years_in_prison_A_list) if prob_B < 0.5 and prison_A < 1.50)

    print(f"Number of dots in Quadrant 1 for A: {count_quadrant1_A}")
    print(f"Number of dots in Quadrant 2 for A: {count_quadrant2_A}")
    print(f"Number of dots in Quadrant 3 for A: {count_quadrant3_A}")
    print(f"Number of dots in Quadrant 4 for A: {count_quadrant4_A}")
    print(f"Number of dots in Quadrant 1 for B: {count_quadrant1_B}")
    print(f"Number of dots in Quadrant 2 for B: {count_quadrant2_B}")
    print(f"Number of dots in Quadrant 3 for B: {count_quadrant3_B}")
    print(f"Number of dots in Quadrant 4 for B: {count_quadrant4_B}")
