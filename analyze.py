import math, scipy, pandas as pd, matplotlib, sklearn as sk, numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

display_on_screen = False

data = pd.DataFrame([])
anxiety = pd.read_csv("anksiyete.csv", delimiter = ",")
depression = pd.read_csv("depresyon.csv", delimiter = ",")
suicide = pd.read_csv("intihar.csv", delimiter = ",")
psychologist = pd.read_csv("psikolog.csv", delimiter = ",")

turkish_types = {"anxiety": "anksiyete", "depression": "depresyon", "suicide": "intihar", "psychologist": "psikolog"}
type_colors = {"anxiety": "olive", "depression": "darkblue", "suicide": "maroon", "psychologist": "purple"}

data["week"] = anxiety["Week"]
data["anxiety"] = anxiety.iloc[:, [1]]
data["depression"] = depression.iloc[:, [1]]
data["suicide"] = suicide.iloc[:, [1]]
data["psychologist"] = psychologist.iloc[:, [1]]
print(data)

def analyze_and_display(type, save_to_file):
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.xaxis.grid(color='gray', linestyle='dashed')

    plt.title(type.capitalize() + "(" + turkish_types[type] + ")" + " Search Terms in Turkey between 01/03/2020 - 13/06/2021")
    plt.xticks(rotation = 30)
    ax1.plot(data["week"], data[type], c = type_colors[type], marker = '.', alpha = 0.8, linestyle = '-.')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Search Interest", rotation = 0, labelpad = 40)

    index = list(data.index)
    anxiety_data = []

    for i in range(len(index)):
        anxiety_data.append([data.index[i], data[type][1]])

    reg = LinearRegression().fit(anxiety_data, data[type])
    y_pred = reg.predict(anxiety_data)
    ax1.plot(y_pred, c = '#666666', marker = '^', markevery = len(data[type]) - 1, linewidth = 3)
    weekly_change = (y_pred[-1] - y_pred[0])/67

    four_week_regs = []
    for i in range (0, len(data[type]), 4):
        sub_list = data[type][i:i+4]
        sub_anxiety_data = anxiety_data[i:i+4]

        reg = LinearRegression().fit(sub_anxiety_data, sub_list)
        y_pred = reg.predict(sub_anxiety_data)
        four_week_regs.extend(y_pred)

    ax1.plot(data["week"][::4], four_week_regs[::4], c = 'black', marker = 'v', alpha = 0.8, linewidth = 2, linestyle = '-')
    ax1.legend([type, 'linear regression', '4-week linear regression'])

    if (save_to_file):
        fig1.savefig("fig_" + type)

analyze_and_display("anxiety", True)
analyze_and_display("depression", True)
analyze_and_display("suicide", True)
analyze_and_display("psychologist", True)

if (display_on_screen):
    plt.show()
