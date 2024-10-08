import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

file_path = 'lyrics.csv'


def ex1(file, number_of_elements):
    data = pd.read_csv(file, header=None)
    df = pd.DataFrame(data)

    arr = []
    first_column = df.iloc[0]
    for element in first_column:
        arr.append(element)

    print(f"initial array: {arr}")

    if number_of_elements > len(arr) - 1:
        number_of_elements %= len(arr)

    # return random.sample(arr, k=number_of_elements)
    # sau
    return np.random.choice(arr, size=number_of_elements, replace=False)


# exercise number2

def throw_a_die():
    die = [1, 2, 3, 4, 5, 6]
    return np.random.choice(die)

# prob1 for heads
# prob2 for tails
def do_game(prob1, prob2):

    # 0-tails, 1-heads
    # Calculate total probability
    total = prob1 + prob2

    # Calculate the probabilities for heads and tails
    p_heads = prob1 / total  # Probability of heads
    p_tails = prob2 / total  # Probability of tails

    deficit = 0

    while True:
        first_result = np.random.choice([1, 0], p=[p_heads, p_tails])
        if first_result == 1:
            # al doilea arunca zarul
            number = throw_a_die()
            sum_of_money = number - 3
            deficit += sum_of_money
            break
        else:
            deficit -= 0.5

    return deficit

def simulate_game(number_of_simulations=1000, p1=0.5, p2=0.5):

    total_deficit = 0
    results_arr = []

    for _ in range(number_of_simulations):
        current_deficit = do_game(p1, p2)
        print(f"current_deficit is : {current_deficit}")
        total_deficit += current_deficit

        results_arr.append(current_deficit)

    print(f"total_deficit is {total_deficit}")
    mean = total_deficit / number_of_simulations
    print(f"Mean deficit per game is {mean}")

    plt.hist(results_arr, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distributia pierderilor jucatorului doi')
    plt.xlabel('Pierderi(LEI)')
    plt.ylabel('Indexul jocului')
    plt.show()


def ex3(num_clients=10000):
    lambda1 = 3
    lambda2 = 6
    lambda3 = 4

    # Probabilities of selecting each barber
    prob1 = 3 / 13
    prob2 = 6 / 13
    prob3 = 4 / 13

    arr = []
    barber_indices = np.random.choice([1, 2, 3], size=num_clients, p=[prob1, prob2, prob3])

    for index in barber_indices:
        if index == 1:
            time = np.random.exponential(scale=1 / lambda1)
        elif index == 2:
            time = np.random.exponential(scale=1 / lambda2)
        else:
            time = np.random.exponential(scale=1 / lambda3)
        arr.append(time)

    service_times = np.array(arr)

    # Calculate mean and standard deviation
    mean_service_time = np.mean(service_times)
    std_service_time = np.std(service_times)

    print(f"Mean service time: {mean_service_time:.2f} hours")
    print(f"Standard deviation of service time: {std_service_time:.2f} hours")

    az.plot_kde(service_times)

    plt.title('Densitatea distributiei timpului de servire')
    plt.xlabel('Timpul de servire (hours)')
    plt.ylabel('Densitatea')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # EX1
    # result_arr = ex1(file_path, number_of_elements=74)
    # print(f "resulted array is {result_arr}")

    # EX2
    # i)the geometric distribution (the first success of the trial)
    # simulate_game(10000, 0.5, 0.5)

    #EX3
    ex3()
