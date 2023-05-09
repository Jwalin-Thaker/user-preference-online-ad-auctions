'''User net architecture module'''
import numpy as np
import random


class SimpleNN:
    def __init__(self, name, input_size, output_size, learning_rate, regularization_rate):
        self.model_name = name
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.weights = np.random.randn(
            input_size, output_size) / np.sqrt(input_size)
        print(f'Initial model weights {self.model_name}:', self.weights)

    def forward(self, x):
        return np.dot(x, self.weights)

    def backward(self, x, grad_output):
        x = x.reshape(-1, self.input_size)  # Reshape x into a 2D array
        # Reshape grad_output into a 2D array
        grad_output = grad_output.reshape(-1, self.output_size)
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(x.T, grad_output) + \
            self.regularization_rate * self.weights
        self.weights -= self.learning_rate * grad_weights
        return grad_input


class OnlineAuctionModel:
    def __init__(self, users, bidders, auctioneers):
        self.users = users
        self.bidders = bidders
        self.auctioneers = auctioneers

        self.learning_rate = 0.01
        self.regularization_rate = 0.1

        self.allocation_network = SimpleNN('allocation', len(self.users[0].preferences), len(
            self.bidders[0].preferences), self.learning_rate, self.regularization_rate)
        self.payment_network = SimpleNN('payment', len(self.bidders[0].preferences), len(
            self.auctioneers[0].preferences), self.learning_rate, self.regularization_rate)

    def second_price_auction(self):
        bids = []
        for bidder in self.bidders:
            bid1 = random.uniform(
                self.auctioneers[0].preferences['min_bid'], bidder.preferences['max_budget'])
            bid2 = bidder.preferences['ctr'] * \
                bidder.preferences['max_budget']
            valuation = max(bid1, bid2)
            bidder.preferences['valuation'] = valuation
            bids.append(max(bid1, bid2))
        sorted_bids = sorted(bids, reverse=True)
        allocation = np.zeros(len(self.bidders))
        payment = np.zeros(len(self.bidders))

        winner_bidder_id = bids.index(sorted_bids[0])
        payment[winner_bidder_id] = sorted_bids[1] if len(
            sorted_bids) > 1 else self.auctioneers[0].preferences['min_bid']
        allocation[winner_bidder_id] = 1

        return allocation, payment

    def train(self):
        user = self.users[0]
        allocation, payment = self.second_price_auction()
        user_preferences = np.array(list(user.preferences.values()))

        for bidder_id, bidder in enumerate(self.bidders):
            bidder_preferences = self.allocation_network.forward(
                user_preferences)
            predicted_auctioneer_preferences = self.payment_network.forward(
                bidder_preferences)

            actual_auctioneer_preferences = np.array(
                [allocation[bidder_id], payment[bidder_id]])
            grad_output = 2 * \
                (predicted_auctioneer_preferences -
                 actual_auctioneer_preferences)

            grad_bidder_preferences = self.payment_network.backward(
                bidder_preferences, grad_output)
            self.allocation_network.backward(
                user_preferences, grad_bidder_preferences)

    def predict(self):
        user_preferences = np.array(
            list(self.users[0].preferences.values()))

        bidder_errors = []
        for bidder in self.bidders:
            bidder_preferences = self.allocation_network.forward(
                user_preferences)
            predicted_auctioneer_preferences = self.payment_network.forward(
                bidder_preferences)
            error = np.abs(
                bidder.preferences['valuation'] - predicted_auctioneer_preferences[1])
            bidder_errors.append(error)

        # Find the bidder with the minimum error
        winner_bidder_id = np.argmin(bidder_errors)
        allocation = np.zeros(len(self.bidders))
        allocation[winner_bidder_id] = 1

        # Calculate the payment for the winning bidder using the payment_network model
        winning_bidder_preferences = self.allocation_network.forward(
            user_preferences)
        predicted_auctioneer_preferences = self.payment_network.forward(
            winning_bidder_preferences)
        payment = np.zeros(len(self.bidders))
        # Assuming the second value is the payment
        payment[winner_bidder_id] = predicted_auctioneer_preferences[1]

        # Calculate item allocation probabilities for each bidder
        normalized_errors = [error / sum(bidder_errors)
                             for error in bidder_errors]
        item_probability = [
            1 - normalized_error for normalized_error in normalized_errors]

        return allocation, payment, item_probability
