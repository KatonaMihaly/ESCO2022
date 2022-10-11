
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import interpolate

from digital_twin_distiller import setup_matplotlib
from digital_twin_distiller import ModelDir
import json
import pandas as pd
import numpy as np

ModelDir.set_base(__file__)

x = []
y = []
alpha = []
T = []
with open(ModelDir.DATA / 'locked_rotor_50.json', 'r', encoding='utf-8') as f:
    res_ = json.load(f)
    alpha_50 = res_.pop('alpha')
    T_50 = res_.pop('T')

with open(ModelDir.DATA / 'locked50.csv', 'r', encoding='utf-8') as f:
    res_ = pd.read_csv(f)
    alpha50 = res_["x"]
    T50 = res_["y"]

with open(ModelDir.DATA / 'locked_rotor_75.json', 'r', encoding='utf-8') as f:
    res_ = json.load(f)
    T_75 = res_.pop('T')

with open(ModelDir.DATA / 'locked75.csv', 'r', encoding='utf-8') as f:
    res_ = pd.read_csv(f)
    alpha75 = res_["x"]
    T75 = res_["y"]

with open(ModelDir.DATA / 'locked_rotor_100.json', 'r', encoding='utf-8') as f:
    res_ = json.load(f)
    T_100 = res_.pop('T')

with open(ModelDir.DATA / 'locked100.csv', 'r', encoding='utf-8') as f:
    res_ = pd.read_csv(f)
    alpha100 = res_["x"]
    T100 = res_["y"]

with open(ModelDir.DATA / 'locked_rotor_125.json', 'r', encoding='utf-8') as f:
    res_ = json.load(f)
    T_125 = res_.pop('T')

with open(ModelDir.DATA / 'locked125.csv', 'r', encoding='utf-8') as f:
    res_ = pd.read_csv(f)
    alpha125 = res_["x"]
    T125= res_["y"]

with open(ModelDir.DATA / 'locked_rotor_150.json', 'r', encoding='utf-8') as f:
    res_ = json.load(f)
    T_150 = res_.pop('T')

with open(ModelDir.DATA / 'locked150.csv', 'r', encoding='utf-8') as f:
    res_ = pd.read_csv(f)
    alpha150 = res_["x"]
    T150 = res_["y"]

with open(ModelDir.DATA / 'locked_rotor_200.json', 'r', encoding='utf-8') as f:
    res_ = json.load(f)
    T_200 = res_.pop('T')

with open(ModelDir.DATA / 'locked200.csv', 'r', encoding='utf-8') as f:
    res_ = pd.read_csv(f)
    alpha200 = res_["x"]
    T200 = res_["y"]

with open(ModelDir.DATA / 'locked_rotor_250.json', 'r', encoding='utf-8') as f:
    res_ = json.load(f)
    T_250 = res_.pop('T')

with open(ModelDir.DATA / 'locked250.csv', 'r', encoding='utf-8') as f:
    res_ = pd.read_csv(f)
    alpha250 = res_["x"]
    T250 = res_["y"]
switch = 3
if switch == 0:
    #--------------------------------------------------------------------------------------------------------------

    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    plt.scatter(alpha_50, T_250, color=colors[0], marker='o')
    plt.scatter(alpha250, T250, color=colors[0], marker='x')
    xdata = alpha_50
    ydata = T_250
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[0], label='Simulation (250A)')
    xdata = alpha50
    ydata = T250
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 177, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[0], linestyle='--', label='Measurement (250A)')

    plt.scatter(alpha_50, T_150, color=colors[4], marker='o')
    plt.scatter(alpha250, T150, color=colors[4], marker='x')
    xdata = alpha_50
    ydata = T_150
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[4], label='Simulation (150A)')
    xdata = alpha50
    ydata = T150
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 177, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[4], linestyle='--', label='Measurement (150A)')

    plt.scatter(alpha_50, T_100, color=colors[6], marker='o')
    plt.scatter(alpha250, T100, color=colors[6], marker='x')
    xdata = alpha_50
    ydata = T_100
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[6], label='Simulation (100A)')
    xdata = alpha50
    ydata = T100
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 177, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[6], linestyle='--', label='Measurement (100A)')

    plt.xlabel('Load angle [deg (electrical)]', fontsize=12)
    plt.ylabel('Static torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0, 200, step=20), fontsize=12)
    plt.yticks(np.arange(-50, 400, step=50), fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "validation2.png", bbox_inches="tight", dpi=650)
    plt.show()
    #--------------------------------------------------------------------------------------------------------------

if switch == 1:
    setup_matplotlib()
    a = 4
    plt.figure(figsize=(6, 4))
    z = np.polyfit(alpha_50, T_250, a)
    predict = np.poly1d(z)
    y = predict(alpha_50)
    plt.plot(alpha_50, y, c="r", label="simulation (250A)")
    z = np.polyfit(alpha50, T250, a)
    predict = np.poly1d(z)
    y = predict(alpha_50)
    plt.plot(alpha_50, y, c="r", linestyle='--', label="measurement (250A)")
    #plt.scatter(alpha_50, T_250, marker="o", c="g")
    #plt.scatter(alpha50, T250, marker="x", c="r")

    z = np.polyfit(alpha_50, T_200, a)
    predict = np.poly1d(z)
    y = predict(alpha_50)
    plt.plot(alpha_50, y, c="g", label="simulation (200A)")
    z = np.polyfit(alpha75, T200, a)
    predict = np.poly1d(z)
    y = predict(alpha_50)
    plt.plot(alpha_50, y, c="g", linestyle='--', label="measurement (200A)")
    #plt.scatter(alpha_50, T_200, lw=2, c="g")
    #plt.scatter(alpha75, T200, lw=2, c="g")

    z = np.polyfit(alpha_50, T_150, a)
    predict = np.poly1d(z)
    y = predict(alpha_50)
    plt.plot(alpha_50, y, c="b", label="simulation (150A)")
    z = np.polyfit(alpha75, T150, a)
    predict = np.poly1d(z)
    y = predict(alpha_50)
    plt.plot(alpha_50, y, c="b", linestyle='--', label="measurement (150A)")
    #plt.scatter(alpha_50, T_150, lw=2, c="b")
    #plt.scatter(alpha75, T150, lw=2, c="b")

    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlabel("Electrical angle [deg]", fontsize=10)
    plt.ylabel("Torque [Nm]", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    # plt.savefig(ModelDir.MEDIA / "locked_rotor.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 2:
    with open(ModelDir.DATA / 'cogging_torque.csv', 'r', encoding='utf-8') as f:
        res_ = pd.read_csv(f)
        x = res_["x"]
        y1 = res_["y1"]
        y2 = res_["y2"]
        y3 = res_["y3"]

    a = 8
    plt.figure(figsize=(6, 4))
    z = np.polyfit(x, y1, a)
    predict = np.poly1d(z)
    y = predict(x)
    plt.plot(x*4, y, c='r', linestyle='-', label="case O")
    z = np.polyfit(x, y2, a)
    predict = np.poly1d(z)
    y = predict(x)
    plt.plot(x*4, y, c='g', linestyle='--', label="case F")
    z = np.polyfit(x, y3, a)
    predict = np.poly1d(z)
    y = predict(x)
    plt.plot(x*4, y, c='b', linestyle='dotted', label="case P")

    # xdata = x * 4
    # ydata = y1
    # spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    # xi = list(np.linspace(0, 30, 100))
    # yi = spline(xi)
    # plt.scatter(xdata, ydata, label="DTD model")
    # plt.plot(xi, yi, label="DTD model")

    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlabel("Electrical angle [deg]", fontsize=12)
    plt.ylabel("Torque [Nm]", fontsize=12)
    plt.xticks(np.arange(0, 35, 5), fontsize=12)
    plt.yticks(np.arange(-2.5, 3, 0.5), fontsize=12)
    plt.legend(fontsize=10)
    #plt.savefig(ModelDir.MEDIA / "cogging_torque.pdf", bbox_inches="tight")
    plt.savefig(ModelDir.MEDIA / "cogging.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 3:
    with open(ModelDir.DATA / 'cogging_torque.csv', 'r', encoding='utf-8') as f:
        res_ = pd.read_csv(f)
        x = res_["x"]
        y1 = res_["y1"]
        y2 = res_["y2"]
        y3 = res_["y3"]

    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    plt.figure(figsize=(6, 4))
    xdata = x
    ydata = y1
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(np.linspace(0, 7.5, 100))
    yi = spline(xi)
    plt.scatter(xdata, ydata, color=colors[0], label="DTD model")
    plt.plot(xi, yi, color=colors[0], label="DTD model")

    xdata = x
    ydata = y2
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(np.linspace(0, 7.5, 100))
    yi = spline(xi)
    plt.scatter(xdata, ydata, color=colors[6], label="Reference full model")
    plt.plot(xi, yi, color=colors[6], label="Reference full model")

    xdata = x
    ydata = y3
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(np.linspace(0, 7.5, 100))
    yi = spline(xi)
    plt.scatter(xdata, ydata, color=colors[4], label="Reference pole model")
    plt.plot(xi, yi, color=colors[4], label="Reference pole model")

    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlabel("Rotor position [deg (mechanical)]", fontsize=12)
    plt.ylabel("Cogging torque [Nm]", fontsize=12)
    plt.xticks(np.arange(0, 8.75, 1.25), fontsize=12)
    plt.yticks(np.arange(-2.5, 3, 0.5), fontsize=12)
    plt.legend(fontsize=10)
    #plt.savefig(ModelDir.MEDIA / "cogging_torque.pdf", bbox_inches="tight")
    plt.savefig(ModelDir.MEDIA / "validation1.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 4:
    # --------------------------------------------------------------------------------------------------------------

    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    case = pd.read_pickle(ModelDir.DATA / "df_locked.pkl")
    with open(ModelDir.DATA / 'motiv1.csv', 'r', encoding='utf-8') as f:
        res_ = pd.read_csv(f)
        blog1 = res_["A"]
        blog1 = [x for x in blog1 if str(x) != 'nan']
        blog1.sort()
        blog2 = res_["AA"]
        blog2 = [x for x in blog2 if str(x) != 'nan']
        pm1 = res_["E"]
        pm2 = res_["EE"]

    xdata = alpha50
    ydata = T250
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 177, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[0], linestyle='--', label='Measurement', linewidth=3)
    xdata = alpha_50
    ydata = T_250
    print(alpha_50)
    print(T_250)
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[1], label='Simulation A', linestyle='--', linewidth=2) # Pyleecan geommal
    xdata = blog1
    ydata = blog2
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[2], label='Simulation B', linestyle='-.', linewidth=2)  # Blogról
    xdata = pm1
    ydata = pm2
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[3], linestyle='dotted', label='Simulation C', linewidth=2) # PM software


    xdata = np.linspace(0, 180, 91)
    ydata = case['torque'].loc[16]
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[4], linestyle='-', label='Simulation D', linewidth=2) # saját


    plt.xlabel('Load angle [deg (electrical)]', fontsize=12)
    plt.ylabel('Static torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0, 200, step=20), fontsize=12)
    plt.yticks(np.arange(-50, 400, step=50), fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "motivation.png", bbox_inches="tight", dpi=650)
    plt.show()
    # --------------------------------------------------------------------------------------------------------------

if switch == 5:
    # --------------------------------------------------------------------------------------------------------------

    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    with open(ModelDir.DATA / 'static.csv', 'r', encoding='utf-8') as f:
        res_ = pd.read_csv(f)
        blog1 = res_["a"]
        blog2 = res_['blog']
        py1 = res_['a']
        py2 = res_['pyleecan']
        pm1 = res_['a']
        pm2 = res_['saját']

    # plt.scatter(res_["a"], res_['blog'], color=colors[2])
    # plt.scatter(res_["a"], res_['pyleecan'], color=colors[1])
    # plt.scatter(res_["a"], res_['saját'], color=colors[3])
    # plt.scatter(res_["a"], res_['mérés'], color=colors[0])

    xdata = res_['a']
    ydata = res_['mérés']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[0], linestyle='-', label='M (250A)')
    xdata = py1
    ydata = py2
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[1], label='P (250A)', linestyle='-') # Pyleecan geommal
    xdata = blog1
    ydata = blog2
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[2], label='B (250A)', linestyle='-')  # Blogról
    xdata = pm1
    ydata = pm2
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[3], linestyle='-', label='S (250A)') # Saját

    xdata = res_['a']
    ydata = res_['m']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[0], linestyle='--', label='M (150A)')
    xdata = res_['a']
    ydata = res_['p']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[1], label='P (150A)', linestyle='--')  # Pyleecan geommal
    xdata = res_['a']
    ydata = res_['b']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[2], label='B (150A)', linestyle='--')  # Blogról
    xdata = res_['a']
    ydata = res_['s']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[3], linestyle='--', label='S (150A)')  # Saját

    xdata = res_['a']
    ydata = res_['mm']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[0], linestyle='dotted', label='M (100A)')
    xdata = res_['a']
    ydata = res_['pp']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[1], label='P (100A)', linestyle='dotted')  # Pyleecan geommal
    xdata = res_['a']
    ydata = res_['bb']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[2], label='B (100A)', linestyle='dotted')  # Blogról
    xdata = res_['a']
    ydata = res_['ss']
    spline = scipy.interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(0, 181, 1))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[3], linestyle='dotted', label='S (100A)')  # Saját

    plt.plot(0,0, label='')
    plt.plot(0, 0, label='')
    plt.plot(0, 0, label='')
    plt.plot(0, 0, label='')

    plt.xlabel('Load angle [deg (electrical)]', fontsize=12)
    plt.ylabel('Static torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0, 200, step=20), fontsize=12)
    plt.yticks(np.arange(-50, 400, step=50), fontsize=12)
    plt.legend(ncol =1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "validation2.png", bbox_inches="tight", dpi=650)
    plt.show()
    # --------------------------------------------------------------------------------------------------------------