'''Scenarios module'''
# custom module import
from utils import entity
from models import user_net_arch


def run_scenario(item_id, identifier):
    # identify the scenario by parsing
    n_users, n_auctioneers, n_bidders = [
        int(val[0]) for val in identifier.split('x')]

    # create user(s)
    users = []
    for i in range(n_users):
        user = entity.User(idn=i)
        user.set_preferences(idn=i)
        users.append(user)

    # create bidder(s)
    bidders = []
    for i in range(n_bidders):
        bidder = entity.Bidder(idn=i)
        bidder.set_preferences(idn=i, users=users)
        bidders.append(bidder)

    # create auctioneer(s)
    auctioneers = []
    for i in range(n_auctioneers):
        auctioneer = entity.Auctioneer(idn=i)
        auctioneer.set_preferences(idn=i)
        auctioneers.append(auctioneer)

    # initialize the model
    model = user_net_arch.OnlineAuctionModel(users, bidders, auctioneers)
    # train the model
    model.train()
    # predict using the model
    allocation, payment, item_probability = model.predict()

    return allocation, payment, item_probability, users, bidders, auctioneers
