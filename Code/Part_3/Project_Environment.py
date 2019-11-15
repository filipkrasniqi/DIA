import numpy as np


class ProjectEnvironment:
    def __init__(self, n_camp, n_subs, prob_subs, bids, slopes, sigmas):
        self.campaign = [Campaign(n_subs[i], prob_subs[i], bids[i], slopes[i], sigmas[i]) for i in range(0, n_camp)]

    def get_single_campaign(self, camp):
        return self.campaign[camp]


class Campaign:
    def __init__(self, n_sub, prob_sub, bids, slopes, sigmas):
        self.sub_campaign = [SubCampaign(bids[i], slopes[i], sigmas[i]) for i in range(0, n_sub)]
        self.sigma = np.max(sigmas)
        self.prob_sub = prob_sub

    def get_clicks_real_aggregate(self, bid):
        return np.sum(sub.get_clicks_real(bid) * self.prob_sub[i] for i, sub in enumerate(self.sub_campaign))

    def get_clicks_noise_aggregate(self, bid):
        y = self.get_clicks_real_aggregate(bid)
        mu, sigma = 0, self.sigma
        sample = np.random.normal(mu, sigma)
        return max(0, y + sample)

    def get_single_sub(self, sub):
        return self.sub_campaign[sub]


class SubCampaign:
    def __init__(self, bid, slope, sigma):
        self.bid = bid
        self.slope = slope
        self.sigma = sigma

    def get_clicks_real(self, bid):
        return max((1 - np.exp(-self.slope * (bid - self.bid))), 0)

    def get_clicks_noise(self, bid):
        y = self.get_clicks_real(bid)
        mu, sigma = 0, self.sigma
        sample = np.random.normal(mu, sigma)
        return max(0, y + sample)