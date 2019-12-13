import numpy as np

n_inf = float("-inf")


class SubcampaignDP:
    def __init__(self, min_budget, max_budget, number_of_clicks):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.number_of_clicks = number_of_clicks  # function

    # x is arm, aka, budget index.
    def n(self, x, arm):
        if arm < self.min_budget:
            return n_inf
        elif arm > self.max_budget:
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

        self.campaigns = []
        for i, (min_budget, max_budget, num_click) in enumerate(
                zip(self.min_budgets, self.max_budgets, self.num_clicks)):
            self.campaigns.append(SubcampaignDP(min_budget, max_budget, num_click))

    def get_budgets(self):
        subcampaigns = self.campaigns
        budgets = self.arms
        init_table = np.zeros(shape=(len(subcampaigns), len(budgets)))
        for idx_s, s in enumerate(subcampaigns):
            row = []
            for idx_arm, arm in enumerate(budgets):
                row.append(s.n(idx_arm, arm))
            init_table[idx_s] = row

        table_result = np.array([])
        previous_row = np.zeros(len(budgets))
        # considero budget b. Per quell'iterazione, considero tutti i budget della subcampaign b_s s.t. b >= budget(previous_row) + budget(subcampaign)
        index_s = 0
        # need to take trace of pairs (i,j) for each subcampaign, s.t. i is the previous row, j is the current value in the dp algorithm
        # this matrix allows to compute how budget are instantiated in the maximum allocation
        pairs_previous_current_for_subcampaign = []
        for idx_s, s in enumerate(subcampaigns):
            results = np.array([])  # array representing solution when adding subcampaign s
            for index_b, b in enumerate(budgets):
                combination_indices = []
                # when I am in subcampaign s, I have previous row containing the best allocation for each budget value
                # fill array of choices of budget for pair (s, b). A choice is to be considered if budget
                choices = np.array([])
                # selezionare gli indici di previous_row che sono sotto a budget
                filtered_choices_pr = previous_row[
                                      0:index_b + 1]  # lista temporanea contenente i casi di previous_row che sono associati ad un budget
                # selezionare per ogni valore di filtered_choices_pr l'associato della riga della subcampaign
                for i in range(0, len(filtered_choices_pr)):
                    num_click_pr = previous_row[i]
                    # find index for associated complementary budget
                    j = np.where(budgets + budgets[i] == b)
                    j = j[0][0]
                    current_num_click_s = init_table[index_s][j]
                    choices = np.append(choices, current_num_click_s + num_click_pr)
                    combination_indices.append((i, j))

                # find maximum
                max_val = np.amax(choices)
                max_val_i = np.argmax(choices)
                if len(combination_indices) > 0:
                    val = combination_indices[max_val_i]
                else:
                    val = (-1, -1)
                pairs_previous_current_for_subcampaign.append(val[0])
                pairs_previous_current_for_subcampaign.append(val[1])
                results = np.append(results, max_val)

            # table_result = np.put(table_result, index_s, results)
            table_result = np.concatenate((table_result, results), axis=0)
            previous_row = results
            index_s = index_s + 1

        table_result = table_result.reshape(len(subcampaigns), len(budgets))
        # pairs_previous_current_for_subcampaign = np.array(pairs_previous_current_for_subcampaign).reshape(len(subcampaigns), len(budgets))

        # backward computation of the budget allocation for each subcampaign
        optimal_value = np.amax(table_result[-1])
        optimal_value_i = np.argmax(table_result[-1])

        budget_for_subcampaign = [0 for s in subcampaigns]
        current_index_subcampaign = len(subcampaigns) - 1

        while current_index_subcampaign >= 0:
            index_pairs = 2 * (current_index_subcampaign * len(budgets) + optimal_value_i)
            previous_value_dp = pairs_previous_current_for_subcampaign[
                index_pairs]  # pairs_previous_current_for_subcampaign[current_index_subcampaign][optimal_value_i]
            current_sub_value_dp = pairs_previous_current_for_subcampaign[index_pairs + 1]
            budget_for_subcampaign[current_index_subcampaign] = budgets[current_sub_value_dp]
            optimal_value_i = previous_value_dp
            current_index_subcampaign -= 1

        return optimal_value, budget_for_subcampaign
