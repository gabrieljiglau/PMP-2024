import math
from scipy import stats


def ex2():

    lambda_arg = 20
    clients = stats.poisson.rvs(mu=lambda_arg, size=1000)  # rvs = random variables

    scale_exp = 2
    time_in_location = stats.expon.rvs(scale=scale_exp, size=1000)

    alpha_scale = calculate_alpha(15, 0.95)
    prep_time = stats.expon.rvs(scale=alpha_scale, size=1000)

    total_time = time_in_location + prep_time

    print(f"Mean number of clients that enter: {clients.mean()}")
    print(f"Mean time spent for a client: {total_time.mean()} (waiting in line + food cooking")

def calculate_alpha(max_time, prob):

    if prob > 1:
        prob /= 100

    alpha = -max_time/math.log(1 - prob)
    print(f"Alpha with max_time={max_time} and prob={prob} is {alpha}")
    return alpha


if __name__ == '__main__':
    ex2()
