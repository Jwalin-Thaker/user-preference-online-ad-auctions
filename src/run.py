'''Main file to run different scenarios'''
import settings
from tqdm import tqdm
from tests import scenarios
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.stats import beta


# Store the results in lists
num_of_items = int(settings.ITEMS)
auction_results = {item_id: [] for item_id in range(num_of_items)}
user_preferences = {item_id: [] for item_id in range(num_of_items)}
bidder_preferences = {item_id: [] for item_id in range(num_of_items)}
auctioneer_preferences = {item_id: [] for item_id in range(num_of_items)}


def plot_user_privacy_threshold_vs_bidder_payouts():
    '''Plot to show user privacy threshold vs the bidder payouts'''
    fig = sp.make_subplots(rows=num_of_items, cols=1, subplot_titles=[
        f'Item {i}' for i in range(num_of_items)])
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    symbols = ['circle', 'square', 'diamond', 'cross', 'x']

    num_points = 100
    x_beta = np.linspace(0, 1, num_points)
    y_beta = beta.pdf(x_beta, 3, 2)

    for item_id, results in auction_results.items():
        allocations = [result[0] for result in results]
        payments = [result[1] for result in results]
        user_privacy_thresholds = [upref[0]['privacy_threshold']
                                   for upref in user_preferences[item_id]]
        bidders_payouts = [payment[payment >= 0.][0] for payment in payments]
        winning_bidders = [np.where(allocation == 1.)[0][0]
                           for allocation in allocations]

        fig.add_trace(go.Scatter(x=x_beta,
                                 y=y_beta,
                                 mode='lines',
                                 line=dict(color='black', dash='dot'),
                                 name='Beta Distribution',
                                 legendgroup='Beta Distribution',
                                 showlegend=(item_id == 0)),
                      row=item_id+1, col=1)

        for j in range(len(np.unique(winning_bidders))):
            x_vals = [user_privacy_thresholds[k] for k in range(
                len(winning_bidders)) if winning_bidders[k] == j]
            y_vals = [bidders_payouts[k] for k in range(
                len(winning_bidders)) if winning_bidders[k] == j]
            fig.add_trace(go.Scatter(x=x_vals,
                                     y=y_vals,
                                     mode='markers',
                                     marker=dict(size=12, symbol=symbols[j % len(
                                         symbols)], color=colors[j % len(colors)]),
                                     text=[j] * len(x_vals),
                                     hovertemplate="Winning Bidder: %{text}<br>" +
                                                   "User Privacy Threshold: %{x}<br>" +
                                                   "Bidder Payout: %{y}<extra></extra>",
                                     name=f"Winning Bidder {j}",
                                     legendgroup=f"Winning Bidder {j}",
                                     showlegend=(item_id == 0)),
                          row=item_id+1, col=1)

        # Update xaxis and yaxis titles for each subplot
        for i in range(num_of_items):
            fig.update_xaxes(
                title_text="User Privacy Threshold", row=i+1, col=1, title_standoff=30, title_font=dict(size=28))
            fig.update_yaxes(title_text="Bidder Payouts", row=i+1,
                             col=1, title_standoff=30, title_font=dict(size=28))

    fig.update_layout(title={'text': 'User Privacy Threshold vs Bidder Payouts',
                             'x': 0.5,
                             'y': 0.99,
                             'xanchor': 'center',
                             'yanchor': 'top',
                             'font': {'size': 24, 'color': 'black', 'family': "Courier New, monospace"}},
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=18)))
    fig.show()


def plot_bidder_valuations_vs_auction_payments():
    fig = sp.make_subplots(rows=num_of_items, cols=1, subplot_titles=[
        f'Item {i}' for i in range(num_of_items)])
    for item_id, results in auction_results.items():
        payments = [result[1] for result in results]
        payments = np.clip(payments, 0, None)
        payments_sum = np.sum(payments, axis=1)
        min_bid = [apref[0]['min_bid']
                   for apref in auctioneer_preferences[item_id]]
        valuation = [bpref[0]['valuation']
                     for bpref in bidder_preferences[item_id]]

        fig.add_trace(go.Scatter(x=list(range(1, settings.ITERATIONS+1)),
                                 y=payments_sum, mode='lines', fill='tonexty', name='Total Payments', legendgroup="Total Payments", showlegend=(item_id == 0)),
                      row=item_id+1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1, settings.ITERATIONS+1)),
                                 y=valuation, mode='lines', fill='tonexty', name='Total Valuation', legendgroup="Total Valuation", showlegend=(item_id == 0)),
                      row=item_id+1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1, settings.ITERATIONS+1)),
                                 y=min_bid, mode='lines', fill='tonexty', name='Minimum Bid', line=dict(dash='dash'), legendgroup="Minimum Bid", showlegend=(item_id == 0)),
                      row=item_id+1, col=1)

    # Update xaxis and yaxis titles for each subplot
    for i in range(num_of_items):
        fig.update_xaxes(title_text="Iterations", row=i+1, col=1,
                         title_standoff=30, title_font=dict(size=28))
        fig.update_yaxes(title_text="Values", row=i+1, col=1,
                         title_standoff=30, title_font=dict(size=28))

    fig.update_layout(title={'text': 'Auctioneer Minimum Bid vs Auction Valuations and Payments',
                             'x': 0.5,
                             'y': 0.99,
                             'xanchor': 'center',
                             'yanchor': 'top',
                             'font': {'size': 24, 'color': 'black', 'family': 'Arial Black'}},
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=18)))
    fig.show()


def plot_item_probability_for_bidder():
    fig = sp.make_subplots(rows=num_of_items, cols=1, subplot_titles=[
                           f'Item {i}' for i in range(num_of_items)])

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for item_id, results in auction_results.items():
        item_probabilities = [result[2] for result in results]

        sorted_bidder_probs = item_probabilities
        sorted_bidder_probs = sorted(item_probabilities, key=lambda x: x[0])

        fig.add_trace(go.Scatter(x=[prob[0] for prob in sorted_bidder_probs],
                                 y=[sum(prob[1:])
                                    for prob in sorted_bidder_probs],
                                 mode='lines+markers',
                                 fill='tozeroy',
                                 fillcolor='#E76161',
                                 line=dict(
                                     color='#A0D8B3'),
                                 name=f"Item {item_id} - Other Bidders",
                                 legendgroup=f"Item {item_id} - Other Bidders"),
                      row=item_id + 1, col=1)

        fig.add_trace(go.Scatter(x=[prob[0] for prob in sorted_bidder_probs],
                                 y=[prob[1]
                                    for prob in sorted_bidder_probs],
                                 mode='lines+markers',
                                 fill='tonexty',
                                 fillcolor='#D6E8DB',
                                 line=dict(
                                     color=colors[item_id % len(colors)]),
                                 name=f"Item {item_id} - Bidder Pairwise",
                                 legendgroup=f"Item {item_id} - Bidder Pairwise",
                                 showlegend=False),
                      row=item_id + 1, col=1)

        fig.update_xaxes(title_text='Bidder 1 Probability',
                         row=item_id + 1, col=1, title_standoff=30, title_font=dict(size=28))
        fig.update_yaxes(title_text='Bidder 2 Probability',
                         row=item_id + 1, col=1, title_standoff=30, title_font=dict(size=28))

    fig.update_layout(title={'text': 'Item Probability Distribution for Bidders',
                             'x': 0.5,
                             'y': 0.99,
                             'xanchor': 'center',
                             'yanchor': 'top',
                             'font': {'size': 24, 'color': 'black', 'family': 'Arial Black'}},
                      legend=dict(orientation='h', yanchor='bottom',
                                  y=1.02, xanchor='right', x=1, font=dict(size=18)),
                      plot_bgcolor='white')
    fig.update_yaxes(zerolinecolor='black')
    fig.show()


print("Start script")
# Run the simulation iterations
for i in tqdm(range(settings.ITERATIONS)):
    for item_id in range(num_of_items):
        np.random.seed(i)
        allocation, payment, item_probability, users, bidders, auctioneers = scenarios.run_scenario(item_id,
                                                                                                    settings.SCENARIO)
        auction_results[item_id].append(
            (allocation, payment, item_probability))
        user_preferences[item_id].append([user.preferences for user in users])
        bidder_preferences[item_id].append(
            [bidder.preferences for bidder in bidders])
        auctioneer_preferences[item_id].append(
            [auctioneer.preferences for auctioneer in auctioneers])
# Call the visualization functions
plot_user_privacy_threshold_vs_bidder_payouts()
plot_bidder_valuations_vs_auction_payments()
plot_item_probability_for_bidder()
print("End script")
