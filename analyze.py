import math, scipy, pandas as pd, matplotlib, sklearn as sk, numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression

data = pd.DataFrame([])
anxiety = pd.read_csv("anksiyete.csv", delimiter = ",")
depression = pd.read_csv("depresyon.csv", delimiter = ",")
suicide = pd.read_csv("intihar.csv", delimiter = ",")
psychologist = pd.read_csv("psikolog.csv", delimiter = ",")

data["week"] = anxiety["Week"]
data["anxiety"] = anxiety.iloc[:, [1]]
data["depression"] = depression.iloc[:, [1]]
data["suicide"] = suicide.iloc[:, [1]]
data["psychologist"] = psychologist.iloc[:, [1]]
print(data)

fig1, ax1 = plt.subplots(figsize=(14, 8))
ax1.xaxis.set_major_locator(ticker.MultipleLocator(4))
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.xaxis.grid(color='gray', linestyle='dashed')

plt.title("Anxiety Search Terms in Turkey between 01/03/2020 - 13/06/2021")
plt.xticks(rotation = 30)
ax1.plot(data["week"], data["anxiety"], c = 'red', marker = '.')
ax1.set_xlabel("Date")
ax1.set_ylabel("Search Interest", rotation = 0, labelpad = 40)

index = list(data.index)
anxiety_data = []

for i in range(len(index)):
    anxiety_data.append([data.index[i], data["anxiety"][1]])

reg = LinearRegression().fit(anxiety_data, data["anxiety"])
y_pred = reg.predict(anxiety_data)
ax1.plot(y_pred, c = 'green')
ax1.legend(['anxiety', 'linear regression'])
weekly_change = (y_pred[-1] - y_pred[0])/67
print(weekly_change)

"""
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.xaxis.set_major_locator(ticker.MultipleLocator(4))

plt.xticks(rotation = 30)
ax2.plot(data["week"], data["depression"], c = 'blue')
ax2.legend(['depression'])

fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.xaxis.set_major_locator(ticker.MultipleLocator(4))

plt.xticks(rotation = 30)
ax3.plot(data["week"], data["suicide"], c = 'black')
ax3.legend(['suicide'])

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.xaxis.set_major_locator(ticker.MultipleLocator(4))
plt.xticks(rotation = 30)

plt.xticks(rotation = 30)
ax4.plot(data["week"], data["psychologist"], c = 'purple')
ax4.legend(['psychologist'])
"""

plt.show()
