import matplotlib.pyplot as plt
import pandas as pd
import pickle

color_list = {1:'b', 2:'g', 3:'r', 4:'c',5:'m', 6:'y', 7:'k', 8:'w'}

def graph_HL_stock(stock_list):
    color_pointer = 1
    for stock in stock_list:
        data = pd.read_csv('preprocessed_stocks/' + stock + ".csv")

        plt.plot(data['date'], data['high'] - data['low'], color=color_list[color_pointer])


        color_pointer += 1
        color_pointer = color_pointer % 6
    plt.show()

def graph_mean_open_iteration(val):
    color_pointer = 1
    cluster_means = []
    file_name = 'D:\Sem V\DWDM\DWDM Final\Stock-cluster\output\chckpt_' + str(val)
    print(file_name)
    file = (open(file_name, "rb"))
    cluster_means.append(pickle.load(file))
    print(cluster_means[0][1])
    data = pd.DataFrame(cluster_means[0][0], columns=["OC", "Volume", "High", "Close"])
    data2 = pd.DataFrame(cluster_means[0][2], columns=["OC", "Volume", "High", "Close"])
    print(data)
    print(data2)
    plt.plot(data['OC'], color=color_list[color_pointer])
    plt.plot(data2['OC'], color=color_list[color_pointer + 2])

    plt.show()


if __name__ == '__main__':
    graph_mean_open_iteration(20)