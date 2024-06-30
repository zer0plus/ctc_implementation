def combination_binomial_probabilities(p, r):
    n = len(p)
    prob = 0.0
    for combo in gen_possible_combinations(range(n), r):
        prob_combo = 1.0
        for i in range(n):
            if i in combo:
                prob_combo *= p[i]
            if not i in combo:
                prob_combo *= (1 - p[i])
        prob += prob_combo
    print("total_combo_prob:", prob)
    return prob


def gen_possible_combinations(items, r):
    n = len(items)
    if r > n:
        return

    # generate the initial combination
    combination = list(range(r))
    yield tuple(items[i] for i in combination)

    while True:
        # Find the rightmost index that can be incremented
        i = r - 1
        while i >= 0 and combination[i] == i + n - r:
            i -= 1

        if i < 0:
            return

        # increment the rightmost index and reset the ones to its right
        combination[i] += 1
        for j in range(i + 1, r):
            combination[j] = combination[j - 1] + 1

        # yield next combo
        yield tuple(items[i] for i in combination)


def prob_rain_more_than_n(p, n):
    total_days = len(p)
    prob = 0.0
    for r in range(n, total_days+1):
        print("combination_binomial_probabilities([p], r):", p, r)
        prob += combination_binomial_probabilities(p, r)

    return prob


if __name__ == "__main__":
    p_test = [0.3, 0.5, 0.2, 0.55, 0.3]
    n_test = 3

    actual_output = prob_rain_more_than_n(p_test, n_test)
    print(f"Probability of atleast {n_test} rainy days: {actual_output:.4f}")