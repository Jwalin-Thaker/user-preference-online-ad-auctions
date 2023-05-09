'''Entity class definitions module'''
# 3P library import
import numpy as np
import math
import scipy.stats as stats
# custom module import
import settings


def pprint(idn, classtype, value_dict):
    '''Pretty print any dictionary'''
    print(f'\n=========== {classtype} {idn} ===========\n')
    for key, value in value_dict.items():
        print(f'{key} : {value}')
    print('\n=========== XXXXXXXXXXX ===========\n')


class User:
    '''User Class'''

    def __init__(self, idn):
        print('User class initialized')
        self.idn = idn
        self.preferences = settings.USER_FEATURES

    def set_preferences(self, idn):
        '''Create and set User preferences'''
        print(f'Setting user preferences - {list(self.preferences.keys())}')
        user_preference = settings.USER_FEATURES.copy()
        random_normal = np.random.randn()  # returns 1 sample form normal dist
        user_preference['trust_score'] = float(stats.norm.cdf(
            random_normal))  # get value from cdf of that number
        user_preference['relevance_score'] = float(
            np.random.rand())  # unifrom distribution
        user_preference['privacy_threshold'] = float(
            np.random.beta(3, 2))  # left skewed distribution
        self.preferences = user_preference
        pprint(idn, self.__class__.__name__, self.preferences)


class Bidder:
    '''Bidder Class'''

    def __init__(self, idn):
        print('Bidder class initialized')
        self.idn = idn
        self.preferences = settings.BIDDER_FEATURES
        self.gamma = np.random.rand()

    def set_preferences(self, idn, users):
        '''Create and set Bidder preferences'''
        print(f'Setting bidder preferences - {list(self.preferences.keys())}')
        bidder_preference = settings.BIDDER_FEATURES.copy()
        bidder_preference['max_budget'] = 1.0
        bidder_preference['auction_participate_cost'] = 0.05
        avg_trust_score = np.average(
            [user.preferences['trust_score'] for user in users])
        avg_relevance_score = np.average(
            [user.preferences['relevance_score'] for user in users])
        avg_privacy_threshold = np.average(
            [user.preferences['privacy_threshold'] for user in users])
        user_impact = (avg_trust_score * avg_relevance_score) / \
            np.sqrt(avg_privacy_threshold)
        print('user_impact', user_impact)
        bidder_preference['ctr'] = float(
            self.gamma * user_impact)
        self.preferences = bidder_preference
        pprint(idn, self.__class__.__name__, self.preferences)


class Auctioneer:
    '''Auctioneer Class'''

    def __init__(self, idn):
        print('Auctioneer class initialized')
        self.idn = idn
        self.preferences = settings.AUCTIONEER_FEATURES

    def set_preferences(self, idn):
        '''Create and set Auctioneer preferences'''
        print(
            f'Setting auctioneer preferences - {list(self.preferences.keys())}')
        auctioneer_preference = settings.AUCTIONEER_FEATURES.copy()
        auctioneer_preference['min_bid'] = float(np.random.rand())
        auctioneer_preference['auction_host_cost'] = 0.05
        self.preferences = auctioneer_preference
        pprint(idn, self.__class__.__name__, self.preferences)
