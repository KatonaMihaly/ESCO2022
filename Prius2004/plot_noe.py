import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linspace
from scipy import interpolate

from digital_twin_distiller import ModelDir

ModelDir.set_base(__file__)

excel_data = pd.read_excel(ModelDir.DATA / "NoE.xlsx")
#print(excel_data)
# For key: parA
parA = list(linspace(0.5, 3.0, 26))
for i in range(len(parA)):
    parA[i] = round(parA[i], 1)

# For key: Den1
a = 0
b = 0
Den0 = []
Den1 = []
Den2 = []
Den3 = []
Den4 = []
Den5 = []
Den6 = []
Den7 = []
Den8 = []
while a < len(parA):
    Den0.append(excel_data.iloc[b+0, 2])
    Den1.append(excel_data.iloc[b+1, 2])
    Den2.append(excel_data.iloc[b+2, 2])
    Den3.append(excel_data.iloc[b+3, 2])
    Den4.append(excel_data.iloc[b+4, 2])
    Den5.append(excel_data.iloc[b+5, 2])
    Den6.append(excel_data.iloc[b+6, 2])
    Den7.append(excel_data.iloc[b+7, 2])
    Den8.append(excel_data.iloc[b+8, 2])
    a = a + 1
    b = b + 9

lista = [[] for i in range(len(parA))]
for i in range(len(parA)):
        lista[i].append(Den0[i])
        lista[i].append(Den1[i])
        lista[i].append(Den2[i])
        lista[i].append(Den3[i])
        lista[i].append(Den4[i])
        lista[i].append(Den5[i])
        lista[i].append(Den6[i])
        lista[i].append(Den7[i])
        lista[i].append(Den8[i])

case = {'parA': parA,
        'Dens': [lista[i] for i in range(len(parA))]}

case = pd.DataFrame(case)

colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']

fig = plt.subplots(figsize=(6, 4))
for i, j in zip(range(0, len(parA), 3), range(9)):
    plt.scatter(range(9), (case['Dens'])[i], color = colors[j], label="A=" + str(0.5+i/10) + "mm")
    xdata = range(9)
    ydata = (case['Dens'])[i]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(linspace(0,8,50))
    yi = spline(xi)
    plt.plot(xi, yi, color = colors[j])
plt.xlabel('Mesh size [u.]', fontsize=12)
plt.ylabel('Number of nodes [u.]', fontsize=12)
plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
plt.minorticks_on()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["1", "0.85", "0.7", "0.55", "0.4", "0.25", "0.1", "0.05", "0.025"], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10)

plt.savefig(ModelDir.MEDIA / "ESCO_2.png", bbox_inches="tight", dpi=650)
plt.show()