import numpy as np
import matplotlib.pyplot as plt

n_inf = float("-inf")
class Subcampaign:
    def __init__(self, min_budget, max_budget, number_of_clicks):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.number_of_clicks = number_of_clicks #function

    # x is arm, aka, budget index.
    def n(self,x):
        if x < self.min_budget:
            return n_inf
        elif x > self.max_budget:
            return n_inf
        else:
            return self.number_of_clicks[x]

class DPAlgorithm:
    def __init__(self, arms, n_subcampaigns, num_clicks, min_budgets, max_budgets):
        self.arms = arms
        self.n_subcampaigns = n_subcampaigns
        self.num_clicks = num_clicks
        self.min_budgets = min_budgets
        self.max_budgets = max_budgets

        assert len(min_budgets) == n_subcampaigns and len(max_budgets) == n_subcampaigns

        self.campaigns = []
        for i, (min_budget, max_budget, num_click) in enumerate(zip(self.min_budgets, self.max_budgets, self.num_clicks)):
            self.campaigns.append(Subcampaign(min_budget, max_budget, num_click))

    def get_budgets(self):
        subcampaigns = self.campaigns
        budgets = self.arms
        init_table = np.zeros(shape=(len(subcampaigns),len(budgets)))
        index_s = 0
        for s in subcampaigns:
            row = []
            for arm, b in enumerate(budgets):
                row.append(s.n(arm))
            init_table[index_s]=row
            index_s = index_s+1

        table_result = np.array([])
        previous_row = np.zeros(len(budgets))
        # considero budget b. Per quell'iterazione, considero tutti i budget della subcampaign b_s s.t. b >= budget(previous_row) + budget(subcampaign)
        index_s = 0
        # need to take trace of pairs (i,j) for each subcampaign, s.t. i is the previous row, j is the current value in the dp algorithm
        # this matrix allows to compute how budget are instanciated in the maximum allocation
        pairs_previous_current_for_subcampaign = []
        for s in subcampaigns:
            index_b = 0
            results = np.array([])  # array representing solution when adding subcampaign s
            for b in budgets:
                combination_indices = []
                # when I am in subcampaign s, I have previous row containing the best allocation for each budget value
                # fill array of choices of budget for pair (s, b). A choice is to be considered if budget
                choices = np.array([])
                if b > s.max_budget:
                    choices = np.append(choices, n_inf)
                else:
                    # selezionare gli indici di previous_row che sono sotto a budget
                    filtered_choices_pr = previous_row[0:index_b+1]#lista temporanea contenente i casi di previous_row che sono associati ad un budget
                    # selezionare per ogni valore di filtered_choices_pr l'associato della riga della subcampaign
                    for i in range(0, len(filtered_choices_pr)):
                        num_click_pr = previous_row[i]
                        # find index for associated complementary budget
                        j = np.where(budgets+budgets[i] == b)
                        j = j[0][0]
                        current_num_click_s = init_table[index_s][j]
                        choices = np.append(choices, current_num_click_s + num_click_pr)
                        combination_indices.append((i,j))

                # find maximum
                max_val = np.amax(choices)
                max_val_i = np.argmax(choices)
                if(len(combination_indices) > 0):
                    val = combination_indices[max_val_i]
                else:
                    val = -1
                pairs_previous_current_for_subcampaign.append(val)
                results = np.append(results, max_val)
                index_b = index_b + 1

            #table_result = np.put(table_result, index_s, results)
            table_result = np.concatenate((table_result, results), axis=0)
            previous_row = results
            index_s = index_s + 1

        table_result = table_result.reshape(len(subcampaigns), len(budgets))
        pairs_previous_current_for_subcampaign = np.array(pairs_previous_current_for_subcampaign).reshape(len(subcampaigns), len(budgets))

        # backward computation of the budget allocation for each subcampaign
        optimal_value = np.amax(table_result[-1])
        optimal_value_i = np.argmax(table_result[-1])

        budget_for_subcampaign = [0 for s in subcampaigns]
        current_index_subcampaign = len(subcampaigns) - 1

        while current_index_subcampaign >= 0:
            current_row_pairs = pairs_previous_current_for_subcampaign[current_index_subcampaign][optimal_value_i]
            budget_for_subcampaign[current_index_subcampaign] = budgets[current_row_pairs[1]]
            optimal_value_i = current_row_pairs[0]
            current_index_subcampaign -= 1

        return (optimal_value, budget_for_subcampaign)
