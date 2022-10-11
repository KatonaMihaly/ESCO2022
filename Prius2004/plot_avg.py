import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from digital_twin_distiller import ModelDir
import pandas as pd
import numpy as np
from scipy import interpolate

ModelDir.set_base(__file__)

# switch = -6 esco_fig8a
# switch = 13 esco_fig9a

switch = 10
if switch == -6:
    doe = pd.read_pickle(ModelDir.DATA / "df_doe250.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    xlist = list(range(124, 149, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    divid = spline(138.5)
    yi0 = yi0 / divid

    xdata = env['rot']
    ydata = env['maxt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    yi3 = yi3 / divid
    plt.plot(xi3, yi3, color=colors[0], label="Upper envelope curve", linestyle="-")

    plt.plot(xi0, yi0, color=colors[8], label="Mean of all entry")

    xdata = env['rot']
    ydata = env['mint']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi4 = xlist
    yi4 = spline(xi3)
    yi4 = yi4 / divid
    plt.plot(xi4, yi4, color=colors[4], label="Lower envelope curve", linestyle="-")

    plt.fill_between(xi3, yi3, yi4, color="gainsboro")

    a = 2
    b = 9
    plt.scatter(env['rot'].loc[a:b], env['maxt'].loc[a:b] / divid, color=colors[0])
    plt.scatter(env['rot'].loc[a:b], env['mint'].loc[a:b] / divid, color=colors[4])
    plt.scatter(env['rot'].loc[a:b], env['avgt'].loc[a:b] / divid, color=colors[8])

    xdata = doe['rot'].loc[144]
    ydata = doe['avg'].loc[144]
    ydata = ydata / divid
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1 mm, h = 1", linestyle='--')

    ax1.set_xlabel('Load angle [deg (electrical)]', fontsize=12)
    ax1.set_ylabel('Average Operational Torque (AOT) [%]', fontsize=12)
    ax1.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    ax1.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    ax1.minorticks_on()
    ax1.set_ylim(0.9425, 1.0125)
    ax1.set_xticks(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_xticklabels(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_yticks(np.arange(0.95, 1.01, step=0.01), fontsize=12)
    ax1.set_yticklabels(np.arange(95, 102, step=1), fontsize=12)
    ax1.set_xlim(115, 149)

    ax2 = ax1.twinx()
    ax2.set_ylim(0.9425, 1.0125)
    ax2.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01], fontsize=12)
    ax2.set_yticklabels(["", "", "", "", "", str(np.round(divid, 2)) + " Nm", ""], fontsize=12)
    label = ax2.yaxis.get_ticklabels()[5]
    label.set_bbox(dict(facecolor='white', edgecolor='none'))
    ax2.tick_params(axis="y", direction="in", pad=-255)
    plt.axvline(138.5, color='#000', ymin=0, ymax=0.825, linestyle='--')
    plt.axhline(1, color='#000', xmin=0.45, xmax=0.71, linestyle='--')
    ax1.legend(loc=3, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "esco_fig8a.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -5:
    doe = pd.read_pickle(ModelDir.DATA / "df_doe100.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env100.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    xlist = list(range(116, 141, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    divid = spline(130)
    yi0 = yi0 / divid

    xdata = env['rot']
    ydata = env['maxt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    yi3 = yi3 / divid
    plt.plot(xi3, yi3, color=colors[0], label="Upper envelope curve", linestyle="-")

    plt.plot(xi0, yi0, color=colors[8], label="Mean of all entry")

    xdata = env['rot']
    ydata = env['mint']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi4 = xlist
    yi4 = spline(xi3)
    yi4 = yi4 / divid
    plt.plot(xi4, yi4, color=colors[4], label="Lower envelope curve", linestyle="-")

    plt.fill_between(xi3, yi3, yi4, color="gainsboro")

    a = 0
    b = 6
    plt.scatter(env['rot'].loc[a:b], env['maxt'].loc[a:b] / divid, color=colors[0])
    plt.scatter(env['rot'].loc[a:b], env['mint'].loc[a:b] / divid, color=colors[4])
    plt.scatter(env['rot'].loc[a:b], env['avgt'].loc[a:b] / divid, color=colors[8])

    xdata = doe['rot'].loc[144]
    ydata = doe['avg'].loc[144]
    ydata = ydata / divid
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1 mm, h = 1", linestyle='--')

    ax1.set_xlabel('Load angle [deg (electrical)]', fontsize=12)
    ax1.set_ylabel('Average torque [%]', fontsize=12)
    ax1.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    ax1.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    ax1.minorticks_on()
    ax1.set_ylim(0.9425, 1.0125)
    ax1.set_xticks(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_xticklabels(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_yticks(np.arange(0.95, 1.01, step=0.01), fontsize=12)
    ax1.set_yticklabels(np.arange(95, 102, step=1), fontsize=12)

    ax2 = ax1.twinx()
    ax1.set_xlim(115, 149)
    ax2.set_ylim(0.9425, 1.0125)
    ax2.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01], fontsize=12)
    ax2.set_yticklabels(["", "", "", "", "", str(np.round(divid, 2)) + " Nm", ""], fontsize=12)
    label = ax2.yaxis.get_ticklabels()[5]
    label.set_bbox(dict(facecolor='white', edgecolor='none'))
    ax2.tick_params(axis="y", direction="in", pad=-100)
    plt.axvline(130, color='#000', ymin=0, ymax=0.825, linestyle='--')
    plt.axhline(1, color='#000', xmin=0.45, xmax=0.695, linestyle='--')
    ax1.legend(loc=4, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "avg100.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -4:
    doe = pd.read_pickle(ModelDir.DATA / "df_doe150.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env150.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    xlist = list(range(120, 145, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    divid = spline(134)
    yi0 = yi0 / divid

    xdata = env['rot']
    ydata = env['maxt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    yi3 = yi3 / divid
    plt.plot(xi3, yi3, color=colors[0], label="Upper envelope curve", linestyle="-")

    plt.plot(xi0, yi0, color=colors[8], label="Mean of all entry")

    xdata = env['rot']
    ydata = env['mint']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi4 = xlist
    yi4 = spline(xi3)
    yi4 = yi4 / divid
    plt.plot(xi4, yi4, color=colors[4], label="Lower envelope curve", linestyle="-")

    plt.fill_between(xi3, yi3, yi4, color="gainsboro")

    a = 1
    b = 7
    plt.scatter(env['rot'].loc[a:b], env['maxt'].loc[a:b] / divid, color=colors[0])
    plt.scatter(env['rot'].loc[a:b], env['mint'].loc[a:b] / divid, color=colors[4])
    plt.scatter(env['rot'].loc[a:b], env['avgt'].loc[a:b] / divid, color=colors[8])

    xdata = doe['rot'].loc[144]
    ydata = doe['avg'].loc[144]
    ydata = ydata / divid
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1 mm, h = 1", linestyle='--')

    ax1.set_xlabel('Load angle [deg (electrical)]', fontsize=12)
    ax1.set_ylabel('Average torque [%]', fontsize=12)
    ax1.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    ax1.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    ax1.minorticks_on()
    ax1.set_ylim(0.9425, 1.0125)
    ax1.set_xticks(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_xticklabels(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_yticks(np.arange(0.95, 1.01, step=0.01), fontsize=12)
    ax1.set_yticklabels(np.arange(95, 102, step=1), fontsize=12)

    ax2 = ax1.twinx()
    ax1.set_xlim(115, 149)
    ax2.set_ylim(0.9425, 1.0125)
    ax2.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01], fontsize=12)
    ax2.set_yticklabels(["", "", "", "", "", str(np.round(divid, 2)) + " Nm", ""], fontsize=12)
    label = ax2.yaxis.get_ticklabels()[5]
    print(label)
    label.set_bbox(dict(facecolor='white', edgecolor='none'))
    ax2.tick_params(axis="y", direction="in", pad=-75)
    plt.axvline(134, color='#000', ymin=0.33, ymax=0.825, linestyle='--')
    plt.axhline(1, color='#000', xmin=0.55, xmax=0.76, linestyle='--')
    ax1.legend(loc=4, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "avg150.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -3:
    doe = pd.read_pickle(ModelDir.DATA / "df_doe200.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env200.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    xlist = list(range(124, 149, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    divid = spline(137)
    yi0 = yi0 / divid

    xdata = env['rot']
    ydata = env['maxt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    yi3 = yi3 / divid
    plt.plot(xi3, yi3, color=colors[0], label="Upper envelope curve", linestyle="-")

    plt.plot(xi0, yi0, color=colors[8], label="Mean of all entry")

    xdata = env['rot']
    ydata = env['mint']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi4 = xlist
    yi4 = spline(xi3)
    yi4 = yi4 / divid
    plt.plot(xi4, yi4, color=colors[4], label="Lower envelope curve", linestyle="-")

    plt.fill_between(xi3, yi3, yi4, color="gainsboro")

    a = 2
    b = 8
    plt.scatter(env['rot'].loc[a:b], env['maxt'].loc[a:b] / divid, color=colors[0])
    plt.scatter(env['rot'].loc[a:b], env['mint'].loc[a:b] / divid, color=colors[4])
    plt.scatter(env['rot'].loc[a:b], env['avgt'].loc[a:b] / divid, color=colors[8])

    xdata = doe['rot'].loc[144]
    ydata = doe['avg'].loc[144]
    ydata = ydata / divid
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1 mm, h = 1", linestyle='--')

    ax1.set_xlabel('Load angle [deg (electrical)]', fontsize=12)
    ax1.set_ylabel('Average torque [%]', fontsize=12)
    ax1.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    ax1.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    ax1.minorticks_on()
    ax1.set_ylim(0.9425, 1.0125)
    ax1.set_xticks(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_xticklabels(np.arange(116, 152, step=4), fontsize=12)
    ax1.set_yticks(np.arange(0.95, 1.01, step=0.01), fontsize=12)
    ax1.set_yticklabels(np.arange(95, 102, step=1), fontsize=12)

    ax2 = ax1.twinx()
    ax1.set_xlim(115, 149)
    ax2.set_ylim(0.9425, 1.0125)
    ax2.set_yticks([0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01], fontsize=12)
    ax2.set_yticklabels(["", "", "", "", "", str(np.round(divid, 2)) + " Nm", ""], fontsize=12)
    label = ax2.yaxis.get_ticklabels()[5]
    label.set_bbox(dict(facecolor='white', edgecolor='none'))
    ax2.tick_params(axis="y", direction="in", pad=-275)
    plt.axvline(137, color='#000', ymin=0.0, ymax=0.82, linestyle='--')
    plt.axhline(1, color='#000', xmin=0.39, xmax=0.65, linestyle='--')
    ax1.legend(loc=3, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "avg200.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -2:
    res = pd.read_pickle(ModelDir.DATA / "df_avgear200.pkl")
    a1 = 0
    a2 = 11
    a3 = 10
    b1 = 31
    b2 = 38
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    print(colors)
    fig = plt.subplots(figsize=(6, 4))
    for c, e in zip(range(a1,a2,a3), range(9)):
        plt.scatter([((res["rotorangle"])[c])[d] for d in range(b1, b2)], [((res["tavg"])[c])[d] for d in range(b1, b2)], label="A=" + str(0.5+c/10) + "mm", color=colors[e])
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tavg"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(124,149,1))
        yi = spline(list(range(124,149,1)))
        plt.plot(xi,yi, color=colors[e])
    plt.xlabel('Load angle [deg]', fontsize=12)
    plt.ylabel('Average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(124, 150, step=2), fontsize=12)
    plt.yticks(np.arange(257, 274, step=2), fontsize=12)
    plt.legend(ncol=3, fontsize=10, loc=8)
    # plt.savefig(ModelDir.MEDIA / "ESCO03.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -1:
    res = pd.read_pickle(ModelDir.DATA / "df_avgear150.pkl")
    a1 = 0
    a2 = 26
    a3 = 3
    b1 = 30
    b2 = 37
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    for c, e in zip(range(a1,a2,a3), range(9)):
        plt.scatter([((res["rotorangle"])[c])[d] for d in range(b1, b2)], [((res["tavg"])[c])[d] for d in range(b1, b2)], label="A=" + str(0.5+c/10) + "mm", color=colors[e])
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tavg"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(120,145,1))
        yi = spline(list(range(120,145,1)))
        plt.plot(xi,yi, color=colors[e])
    plt.xlabel('Load angle [deg]', fontsize=12)
    plt.ylabel('Average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(120, 146, step=2), fontsize=12)
    plt.yticks(np.arange(200, 215, step=2), fontsize=12)
    plt.legend(ncol=3, fontsize=10, loc=8)
    plt.savefig(ModelDir.MEDIA / "ESCO02.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 0:
    res = pd.read_pickle(ModelDir.DATA / "df_avgear100.pkl")
    a1 = 0
    a2 = 26
    a3 = 3
    b1 = 29
    b2 = 36
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    print(colors)
    fig = plt.subplots(figsize=(6, 4))
    for c, e in zip(range(a1,a2,a3), range(9)):
        plt.scatter([((res["rotorangle"])[c])[d] for d in range(b1, b2)], [((res["tavg"])[c])[d] for d in range(b1, b2)], label="A=" + str(0.5+c/10) + "mm", color=colors[e])
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tavg"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(116,141,1))
        yi = spline(list(range(116,141,1)))
        plt.plot(xi,yi, color=colors[e])
    plt.xlabel('Load angle [deg]', fontsize=12)
    plt.ylabel('Average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(116, 142, step=2), fontsize=12)
    plt.yticks(np.arange(135, 145, step=1), fontsize=12)
    plt.legend(ncol=3, fontsize=10, loc=8)
    plt.savefig(ModelDir.MEDIA / "ESCO01.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 1:
    res = pd.read_pickle(ModelDir.DATA / "df_avgear250.pkl")
    a1 = 0
    a2 = 26
    a3 = 3
    b1 = 0
    b2 = 45
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    print(colors)
    fig = plt.subplots(figsize=(6, 4))
    for c, e in zip(range(a1,a2,a3), range(9)):
        plt.scatter([((res["rotorangle"])[c])[d] for d in range(b1, b2)], [((res["tavg"])[c])[d] for d in range(b1, b2)], label="A=" + str(0.5+c/10) + "mm", color=colors[e])
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tavg"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        print(xdata)
        xi = list(range(128,149,1))
        yi = spline(list(range(128,149,1)))
        plt.plot(xi,yi, color=colors[e])
    plt.xlabel('Load angle [deg]', fontsize=12)
    plt.ylabel('Torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks( np.arange(128, 150, step=2))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(ncol=3, fontsize=10, loc=8)
    plt.savefig(ModelDir.MEDIA / "ESCO04.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 2:
    res1 = pd.read_pickle(ModelDir.DATA / "df_avgear100.pkl")
    res2 = pd.read_pickle(ModelDir.DATA / "df_avgear150.pkl")
    res3 = pd.read_pickle(ModelDir.DATA / "df_avgear200.pkl")
    res4 = pd.read_pickle(ModelDir.DATA / "df_avgear250.pkl")
    a1 = 0
    a2 = 26
    a3 = 1
    b1 = 0
    b2 = 46
    # fig = plt.subplots(figsize=(6, 4))
    for c in range(a1, a2, a3):
        plt.plot([((res1["rotorangle"])[c])[d] for d in range(b1, b2)], [((res1["tavg"])[c])[d] for d in range(b1, b2)], color=('#50237F'))
        plt.plot([((res2["rotorangle"])[c])[d] for d in range(b1, b2)], [((res2["tavg"])[c])[d] for d in range(b1, b2)], color=('#B90276'))
        plt.plot([((res3["rotorangle"])[c])[d] for d in range(b1, b2)], [((res3["tavg"])[c])[d] for d in range(b1, b2)], color=('#008ECF'))
        plt.plot([((res4["rotorangle"])[c])[d] for d in range(b1, b2)], [((res4["tavg"])[c])[d] for d in range(b1, b2)], color=('#78BE20'))
    plt.xlabel('Load angle [deg]', fontsize=12)
    plt.ylabel('Average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0, 200, step=20))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    custom_lines = [Line2D([0], [0], color=('#78BE20'), lw=4),
                    Line2D([0], [0], color=('#008ECF'), lw=4),
                    Line2D([0], [0], color=('#B90276'), lw=4),
                    Line2D([0], [0], color=('#50237F'), lw=4)]
    plt.legend(custom_lines, ["i=250A", "i=200A","i=150A","i=100A"], fontsize=12)
    plt.savefig(ModelDir.MEDIA / "x.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 3:
    res = pd.read_pickle(ModelDir.DATA / "df_ripple100.pkl")
    a1 = 1
    a2 = 26
    a3 = 5
    b1 = 0
    b2 = 61
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    plt.scatter(res["earheight"], res["maxavg"], color=colors[2], label = "Simulation data")
    xdata = res["earheight"]
    ydata = res["maxavg"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(np.linspace(0.5,3,100))
    yi = spline(xi)
    plt.plot(xi, yi, color=colors[2], label="Approximation")
    plt.xlabel('Parameter A [deg]', fontsize=12)
    plt.ylabel('Maximum of the average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0.5, 3.5, step=0.5))
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(141.75, 144, 0.25), fontsize=12)
    plt.legend(fontsize=10)

    plt.savefig(ModelDir.MEDIA / "ESCO05.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 4:
    res = pd.read_pickle(ModelDir.DATA / "df_ripple150.pkl")
    a1 = 1
    a2 = 26
    a3 = 5
    b1 = 0
    b2 = 61
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    plt.scatter(res["earheight"], res["maxavg"], color=colors[1], label = "Simulation data")
    xdata = res["earheight"]
    ydata = res["maxavg"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(np.linspace(0.5,3,100))
    yi = spline(xi)
    plt.plot(xi, yi, color=colors[1], label="Approximation")
    plt.xlabel('Parameter A [deg]', fontsize=12)
    plt.ylabel('Maximum of the average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0.5, 3.5, step=0.5))
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(211, 213.75, 0.25), fontsize=12)
    plt.legend(fontsize=10)

    plt.savefig(ModelDir.MEDIA / "ESCO06.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 5:
    res = pd.read_pickle(ModelDir.DATA / "df_ripple200.pkl")
    a1 = 1
    a2 = 26
    a3 = 5
    b1 = 0
    b2 = 61
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    plt.scatter(res["earheight"], res["maxavg"], color=colors[3], label = "Simulation data")
    xdata = res["earheight"]
    ydata = res["maxavg"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(np.linspace(0.5,3,100))
    yi = spline(xi)
    plt.plot(xi, yi, color=colors[3], label="Approximation")
    plt.xlabel('Parameter A [deg]', fontsize=12)
    plt.ylabel('Maximum of the average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0.5, 3.5, step=0.5))
    plt.xticks(fontsize=12)
    plt.yticks(np.arange(270.25, 273.5, 0.5), fontsize=12)
    plt.legend(fontsize=10)

    plt.savefig(ModelDir.MEDIA / "ESCO07.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 6:
    res = pd.read_pickle(ModelDir.DATA / "df_ripple250.pkl")
    a1 = 1
    a2 = 26
    a3 = 5
    b1 = 0
    b2 = 61
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    plt.scatter(res["earheight"], res["maxavg"], color=colors[6], label= 'Simulation')
    xdata = res["earheight"]
    ydata = res["maxavg"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi = list(np.linspace(0.5,3,100))
    yi = spline(xi)
    plt.plot(xi, yi, color=colors[6], label= "Approximation")
    plt.xlabel('Parameter A [deg]', fontsize=12)
    plt.ylabel('Maximum of the average torque [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0.5, 3.5, step=0.5))
    plt.yticks(np.arange(315.5, 321, step=0.5))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)

    plt.savefig(ModelDir.MEDIA / "ESCO08.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 7:
    fig = plt.subplots(figsize=(6, 4))
    res = pd.read_pickle(ModelDir.DATA / "df_avgear250.pkl")
    a1 = 0
    a2 = 1
    a3 = 1
    b1 = 0
    b2 = 45
    colors = ['#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    print(colors)
    for c, e in zip(range(a1, a2, a3), range(9)):
        # plt.plot([((res["rotorangle"])[c])[d] for d in range(b1, b2)], [((res["tavg"])[c])[d] for d in range(b1, b2)], label="A=" + str(0.5+c/10) + "mm", color=colors[e])
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tavg"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color="#006249", label= "250A " + "- avg", linewidth=3.5)
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tmin"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color="#006249", linestyle='-.', label= "250A " + "- min", linewidth=3.5)
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tmax"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color="#006249", linestyle='--', label= "250A " + "- max", linewidth=3.5)

    res = pd.read_pickle(ModelDir.DATA / "df_avgear150.pkl")
    a1 = 0
    a2 = 1
    a3 = 1
    b1 = 0
    b2 = 45
    colors = ['#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    print(colors)
    for c, e in zip(range(a1, a2, a3), range(9)):
        # plt.plot([((res["rotorangle"])[c])[d] for d in range(b1, b2)], [((res["tavg"])[c])[d] for d in range(b1, b2)], label="A=" + str(0.5+c/10) + "mm", color=colors[e])
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tavg"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color='#005691', label= "150A " + "- avg", linewidth=3.5)
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tmin"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color='#005691', linestyle='-.', label= "150A " + "- max", linewidth=3.5)
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tmax"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color='#005691', linestyle='--', label= "150A " + "- max", linewidth=3.5)

    res = pd.read_pickle(ModelDir.DATA / "df_avgear100.pkl")
    a1 = 0
    a2 = 1
    a3 = 1
    b1 = 0
    b2 = 45
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    for c, e in zip(range(a1, a2, a3), range(9)):
        # plt.plot([((res["rotorangle"])[c])[d] for d in range(b1, b2)], [((res["tavg"])[c])[d] for d in range(b1, b2)], label="A=" + str(0.5+c/10) + "mm", color=colors[e])
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tavg"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color="#B90276", label="100A " + "- avg", linewidth=3.5)
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tmin"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color="#B90276", linestyle='-.', label="100A " + "- min", linewidth=3.5)
        xdata = [((res["rotorangle"])[c])[d] for d in range(b1, b2)]
        ydata = [((res["tmax"])[c])[d] for d in range(b1, b2)]
        spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
        xi = list(range(0, 180, 1))
        yi = spline(list(range(0, 180, 1)))
        plt.plot(xi, yi, color="#B90276", linestyle='--', label="100A " + "- max", linewidth=3.5)
    plt.xlabel('Load angle [deg]', fontsize=16)
    plt.ylabel('Torque [Nm]', fontsize=16)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(0, 200, step=20))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(ncol=1, fontsize=11, loc=2)

    plt.savefig(ModelDir.MEDIA / "ESCO6.svg", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 10:

    res = pd.read_pickle(ModelDir.DATA / "df_wave100.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xdata = res['A']
    ydata = res['max']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[0], label="Maximum amplitude")

    xdata = res['A']
    ydata = res['min']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(range(116, 149, 1))
    yi2 = spline(list(range(116, 149, 1)))
    plt.plot(xi2, yi2, color=colors[4], label = "Minimum amplitude")

    xdata = res['A']
    ydata = res['mean']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = list(range(116, 149, 1))
    yi3 = spline(xi3)
    plt.plot(xi3, yi3, color=colors[8], label="Mean amplitude", linestyle="--")
    print(spline(130))
    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    xdata = res['A']
    ydata = res['opt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1mm, h = 1", linestyle='--')

    plt.xlabel('Load angle [deg (electrical)]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.ylim(6, 16)
    plt.xticks(np.arange(116, 152, step=4), fontsize=12)
    plt.yticks(np.arange(6.5, 16, 1), fontsize=12)
    plt.axvline(130, color='#000', ymin=0, ymax=1, linestyle=':', label="Peak of AOT at \u03B4 = 130°", linewidth=2)
    plt.legend(loc=4, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "wav100.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 11:

    res = pd.read_pickle(ModelDir.DATA / "df_wave150.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xdata = res['A']
    ydata = res['max']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[0], label="Maximum amplitude")

    xdata = res['A']
    ydata = res['min']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(range(116, 149, 1))
    yi2 = spline(list(range(116, 149, 1)))
    plt.plot(xi2, yi2, color=colors[4], label = "Minimum amplitude")

    xdata = res['A']
    ydata = res['mean']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = list(range(116, 149, 1))
    yi3 = spline(xi3)
    plt.plot(xi3, yi3, color=colors[8], label="Mean amplitude", linestyle="--")
    print(spline(134))
    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    xdata = res['A']
    ydata = res['opt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1mm, h = 1", linestyle='--')

    plt.xlabel('Load angle [deg (electrical)]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(116, 152, step=4), fontsize=12)
    plt.ylim(6, 14)
    plt.yticks(np.arange(6.5, 14, 1), fontsize=12)
    plt.axvline(134, color='#000', ymin=0, ymax=1, linestyle=':', label="Peak of AOT at \u03B4 = 134°", linewidth=2)
    plt.legend(loc=2, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "wav150.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 12:

    res = pd.read_pickle(ModelDir.DATA / "df_wave200.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xdata = res['A']
    ydata = res['max']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[0], label="Maximum amplitude")

    xdata = res['A']
    ydata = res['min']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(range(116, 149, 1))
    yi2 = spline(list(range(116, 149, 1)))
    plt.plot(xi2, yi2, color=colors[4], label = "Minimum amplitude")

    xdata = res['A']
    ydata = res['mean']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = list(range(116, 149, 1))
    yi3 = spline(xi3)
    plt.plot(xi3, yi3, color=colors[8], label="Mean amplitude", linestyle="--")
    print(spline(137))
    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    xdata = res['A']
    ydata = res['opt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1mm, h = 1", linestyle='--')

    plt.xlabel('Load angle [deg (electrical)]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.ylim(6, 14)
    plt.xticks(np.arange(116, 152, step=4), fontsize=12)
    plt.yticks(np.arange(6.5, 14, 1), fontsize=12)
    plt.axvline(137, color='#000', ymin=0, ymax=1, linestyle=':', label="Peak of AOT at \u03B4 = 137°", linewidth=2)
    plt.legend(loc=2, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "wav200.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 13:

    res = pd.read_pickle(ModelDir.DATA / "df_wave250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xdata = res['A']
    ydata = res['max']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[0], label="Maximum amplitude")

    xdata = res['A']
    ydata = res['min']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(range(116, 149, 1))
    yi2 = spline(list(range(116, 149, 1)))
    plt.plot(xi2, yi2, color=colors[4], label = "Minimum amplitude")

    xdata = res['A']
    ydata = res['mean']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = list(range(116, 149, 1))
    yi3 = spline(xi3)
    plt.plot(xi3, yi3, color=colors[8], label="Mean amplitude", linestyle="--")
    print(spline(138.5))
    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    xdata = res['A']
    ydata = res['opt']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(range(116, 149, 1))
    yi1 = spline(list(range(116, 149, 1)))
    plt.plot(xi1, yi1, color=colors[6], label="A = 2.1mm, h = 1", linestyle='--')

    plt.xlabel('Load angle [deg (electrical)]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.ylim(8, 15)
    plt.xticks(np.arange(116, 152, step=4), fontsize=12)
    plt.yticks(np.arange(8, 16, 1), fontsize=12)
    plt.axvline(138.5, color='#000', ymin=0, ymax=1, linestyle=':', label="Peak of AOT at \u03F1 = 138.5°", linewidth=2)
    plt.legend(loc=2, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "wav250.png", bbox_inches="tight", dpi=650)
    plt.show()