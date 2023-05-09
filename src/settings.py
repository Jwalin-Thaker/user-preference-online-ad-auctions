'''Settings module'''
import numpy as np

ITEMS = '1'
SCENARIO = '1Ux1Ax2B'  # item vs user vs auctioneer vs bidder
ITERATIONS = 500
USER_FEATURES = {
    'trust_score': 0.,
    'relevance_score': 0.,
    'privacy_threshold': 0.,
}
BIDDER_FEATURES = {
    'max_budget': 0.,
    'ctr': 0.,
    'auction_participate_cost': 0.,
    'valuation': 0.
}
AUCTIONEER_FEATURES = {
    'min_bid': 0.,
    'auction_host_cost': 0.
}
