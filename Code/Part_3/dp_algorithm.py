import numpy as np
import matplotlib.pyplot as plt

n_inf = float("-inf")
class Subcampaign:
    def __init__(self, bid, min_budget, max_budget, number_of_clicks):
        self.bid = bid
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.number_of_clicks = number_of_clicks #function

    # x is budget.
    def n(self,x):
        arm = int(x/10)
        if x < self.min_budget:
            return n_inf
        elif x < self.bid:
            return 0
        elif x > self.max_budget:
            return n_inf
        else:
            return self.number_of_clicks[arm]

def n1(x):
    y = [n_inf, 90, 100, 105, 110]
    return y[int(x/10)]
def n2(x):
    y = [0, 82, 90, 92]
    return y[int(x/10)]
def n3(x):
    y = [0, 80, 83, 85,86]
    return y[int(x/10)]
def n4(x):
    y = [n_inf, 90, 110, 115, 118, 120]
    return y[int(x/10)]
def n5(x):
    y = [n_inf, 111, 130, 138, 142, 148, 155]
    return y[int(x/10)]

budgets = np.linspace(0.0, 70.0, 8)

# TODO matrice n_subc = 5 righe, n_budget = B colonne
campaigns = [
    [n_inf, 90, 100, 105, 110],
    [0, 82, 90, 92],
    [0, 80, 83, 85,86],
    [n_inf, 90, 110, 115, 118, 120],
    [n_inf, 111, 130, 138, 142, 148, 155]
]

subcampaigns = np.array([])
subcampaigns = np.append(subcampaigns, Subcampaign(2, 10, 40, campaigns[0]))
subcampaigns = np.append(subcampaigns, Subcampaign(1, 0, 30, campaigns[1]))
subcampaigns = np.append(subcampaigns, Subcampaign(0.5, 0, 40, campaigns[2]))
subcampaigns = np.append(subcampaigns, Subcampaign(0.5, 10, 50, campaigns[3]))
subcampaigns = np.append(subcampaigns, Subcampaign(1, 10, 60, campaigns[4]))

init_table = np.zeros(shape=(len(subcampaigns),len(budgets)))
index_s = 0
for s in subcampaigns:
    row = []
    for b in budgets:
        row.append(s.n(b))
    init_table[index_s]=row
    index_s = index_s+1

table_result = np.array([])
previous_row = np.zeros(len(budgets))
# considero budget b. Per quell'iterazione, considero tutti i budget della subcampaign b_s s.t. b >= budget(previous_row) + budget(subcampaign)
index_s = 0
for s in subcampaigns:
    print(index_s)
    print()
    print(previous_row)
    print()
    print(init_table[index_s])
    print()
    index_b = 0
    results = np.array([])  # array representing solution when adding subcampaign s
    for b in budgets:
        # when I am in subcampaign s, I have previous row containing the best allocation for each budget value
        # fill array of choices of budget for pair (s, b). A choice is to be considered if budget
        choices = np.array([])
        if b > s.max_budget:
            choices = np.append(choices, n_inf)
        else:
            # selezionare gli indici di previous_row che sono sotto a budget
            filtered_choices_pr = previous_row[0:index_b+1]#lista temporanea contenente i casi di previous_row che sono associati ad un budget complementare
            #print(filtered_choices_pr)
            # selezionare per ogni valore di filtered_choices_pr l'associato della riga della subcampaign
            for i in range(0, len(filtered_choices_pr)):
                num_click_pr = previous_row[i]
                # find index for associated complementary budget
                j = np.where(budgets+budgets[i] == b)
                j = j[0][0]
                #print(i,j)
                current_num_click_s = init_table[index_s][j]
                choices = np.append(choices, current_num_click_s + num_click_pr)
                #print(current_num_click_s, num_click_pr)

        #print(choices)
        # find maximum
        max_val = np.amax(choices)
        results = np.append(results, max_val)
        index_b = index_b + 1

    #table_result = np.put(table_result, index_s, results)
    table_result = np.concatenate((table_result, results), axis=0)
    previous_row = results
    index_s = index_s + 1

table_result = table_result.reshape(len(subcampaigns), len(budgets))

plt.plot(budgets, table_result[-1])
