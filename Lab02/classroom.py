import random

def lab_exercise(num_simulations=10000):
    red_count = 0

    for _ in range(num_simulations):
        # initial
        urn = ['red'] * 3 + ['blue'] * 4 + ['black'] * 2

        # dam cu zarul
        die_roll = random.randint(1, 6)

        # modificam urna
        if die_roll in [2, 3, 5]:
            urn.append('black')
        elif die_roll == 6:
            urn.append('red')
        else:  # 1 or 4
            urn.append('blue')

        # scoatem o bila
        drawn_ball = random.choice(urn)

        if drawn_ball == 'red':
            red_count += 1

    red_probability = red_count / num_simulations
    return red_probability


"""
probabilitatea teoretica = 0.31(6)
explicatii:

1/2 sansa ca nr. de pe zar sa fie prim(2,3,5); adaugam o bila neagra; P(rosu) = 3/10

1/6 sansa ca nr de pe zar sa fie 6; adaugam o bila rosie; P(rosu) = 4/10

1/3 sansa 'altfel' (sansa 1 sau 4); adaugam o bila albastra; P(rosu) = 3/10

P(rosu per total) = 1/2 * 3/10 + 1/6 * 4/10 + 1/3 * 3/10 = 3/10 + 1/15 + 1/10 = 0.31(6)



simuland experimentul, primesc : 'Estimated probability of drawing a red ball: 0.3165', cu 100000 de simulari

concluzia: viata bate filmul
"""

if __name__ == '__main__':
    estimated_probability = lab_exercise(100000)
    print(f"Estimated probability of drawing a red ball: {estimated_probability:.4f}")
