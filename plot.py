import matplotlib.pyplot as plt
import pandas as pd

def plot(data,final_signals,cumulative_returns):

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 3, 1]})

    # Get indices for buy and sell signals
    buy_indices = [i for i, signal in enumerate(final_signals) if signal == 1]
    sell_indices = [i for i, signal in enumerate(final_signals) if signal == -1]

    # Plotting Buy and Sell markers for positions in the second subplot
    axs[0].plot(data.index, data['Adj Close'], label='Price', color='black')
    axs[0].scatter(data.iloc[buy_indices].index, data.iloc[buy_indices]['Adj Close'], marker='^', color='green', label='Buy Signal', zorder=5)
    axs[0].scatter(data.iloc[sell_indices].index, data.iloc[sell_indices]['Adj Close'], marker='v', color='red', label='Sell Signal', zorder=5)
    axs[0].set_title('Buy/Sell Markers')
    axs[0].legend()

    def chart_area(axs, indices_1, indices_2, colour):
        for x, y in zip(indices_1, indices_2):
            if x < y:
                indices_range = list(range(x, y + 1))
                axs[1].axvspan(data.index[indices_range[0]], data.index[indices_range[-1]], facecolor=colour, alpha=0.05)

    # Assuming 'buy_indices', 'sell_indices', and 'data' are defined elsewhere

    pos_indices=[]
    neg_indices=[]
    for x in range(1,len(data)):
        if data['Adj Close'][x] > data ['Adj Close'][x-1]:
            pos_indices.append(x)
        else:
            neg_indices.append(x)

    # # Green areas for buy and sell indices
    chart_area(axs, pos_indices, neg_indices, 'green')
    chart_area(axs, neg_indices, pos_indices, 'red')

    # chart_area(axs, buy_indices, sell_indices, 'green')
    # chart_area(axs, sell_indices, buy_indices, 'red')

    # Red areas for remaining indices until the end of the data
    # last_index = len(data) - 1
    # plot_indices(axs, list(range(buy_indices[-1], last_index)), [last_index], 'red')

    # Plotting Price and Cumulative Returns in the same subplot
    axs[1].plot(data.index, data['Adj Close'], label='Price', color='black')
    axs[1].set_title('Price and Buy/Sell HeatMap')
    axs[1].legend()

    # Plotting gain (green) and loss (red) segments on historical price plot with thicker lines
    for buy_idx in buy_indices:
        sell_idx = min(filter(lambda x: x > buy_idx, sell_indices), default=None)
        if sell_idx is not None:
            color = 'green' if data['Adj Close'].iloc[sell_idx] > data['Adj Close'].iloc[buy_idx] else 'red'
            axs[1].plot(data.iloc[buy_idx:sell_idx+1].index, data.iloc[buy_idx:sell_idx+1]['Adj Close'], color=color, linewidth=3)  # Adjust the linewidth for thickness
            # if data['Adj Close'].iloc[sell_idx] > data['Adj Close'].iloc[buy_idx]:
            #     axs[1].axvspan(data.iloc[buy_idx:sell_idx+1].index[0], data.iloc[buy_idx:sell_idx+1].index[-1], facecolor='green', alpha=0.01)
            # else:
            #     axs[1].axvspan(data.iloc[buy_idx:sell_idx+1].index[0], data.iloc[buy_idx:sell_idx+1].index[-1], facecolor='red', alpha=0.01)

    # Plotting Cumulative Returns
    axs[2].plot(data.index, cumulative_returns, label='Cumulative Returns', color='orange')
    axs[2].set_title('Cumulative Returns')
    axs[2].legend()

    # Marking areas in Price plot based on Cumulative Returns movement
    for i in range(1, len(cumulative_returns)):
        if cumulative_returns[i] > cumulative_returns[i - 1]:
            axs[2].axvspan(data.index[i - 1], data.index[i], facecolor='green', alpha=0.3)
        elif cumulative_returns[i] < cumulative_returns[i - 1]:
            axs[2].axvspan(data.index[i - 1], data.index[i], facecolor='red', alpha=0.3)


    plt.tight_layout()
    plt.show()