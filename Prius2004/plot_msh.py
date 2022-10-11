import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from digital_twin_distiller import ModelDir
import pandas as pd
import numpy as np
from scipy import interpolate
from astropy.visualization import hist

ModelDir.set_base(__file__)

switch = -4
if switch == -1:
    env = pd.read_pickle(ModelDir.DATA / "df_env100.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe100.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xlist = list(range(116, 141, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    divid = spline(130)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = xlist
            yi = spline(xi)
            xm = 130
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    A = list(np.arange(0.5, 3.1, 0.1))
    A = [round(x, 2) for x in A]

    xdata = A
    ydata = tempmax
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(np.arange(0.5, 3.01, 0.01))
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    xdata = A
    ydata = tempmin
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(np.arange(0.5, 3.01, 0.01))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")
    plt.axhline(1, color='#000', xmin=0, xmax=1, linestyle='--',
                label="Mean value = " + str(np.round(divid, 2)) + " Nm")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Max of the average torque [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.4, 3.1)
    plt.ylim(0.995, 1.011)
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.yticks(list(np.arange(0.996, 1.01, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.axvline(2.1, color='#000', ymin=0, ymax=0.68, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.yticks(fontsize=12)
    plt.legend(loc=1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxi1.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -4:
    env = pd.read_pickle(ModelDir.DATA / "df_env250.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = list(range(124, 149, 1))
    yi3 = spline(xi3)
    divid = spline(138.5)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = list(range(124, 149, 1))
            yi = spline(xi)
            xm = 138.5
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    A = list(np.arange(0.5, 3.1, 0.1))
    A = [round(x, 2) for x in A]

    xdata = A
    ydata = tempmax
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(np.arange(0.5, 3.01, 0.01))
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    xdata = A
    ydata = tempmin
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(np.arange(0.5, 3.01, 0.01))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")
    plt.axhline(1, color='#000', xmin=0, xmax=1, linestyle='--', label = "Mean value = " + str(np.round(divid, 2)) + " Nm")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Maximum of AOT [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.4, 3.1)
    plt.ylim(0.995, 1.011)
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.yticks(list(np.arange(0.996, 1.01, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.axvline(2.1, color='#000', ymin=0, ymax=0.68, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.yticks(fontsize=12)
    plt.legend(loc=1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxi4.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -3:
    env = pd.read_pickle(ModelDir.DATA / "df_env200.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe200.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xlist = list(range(124, 149, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    divid = spline(137)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = xlist
            yi = spline(xi)
            xm = 137
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    A = list(np.arange(0.5, 3.1, 0.1))
    A = [round(x, 2) for x in A]

    xdata = A
    ydata = tempmax
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(np.arange(0.5, 3.01, 0.01))
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    xdata = A
    ydata = tempmin
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(np.arange(0.5, 3.01, 0.01))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")
    plt.axhline(1, color='#000', xmin=0, xmax=1, linestyle='--',
                label="Mean value = " + str(np.round(divid, 2)) + " Nm")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Max of the average torque [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.4, 3.1)
    plt.ylim(0.995, 1.011)
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.yticks(list(np.arange(0.996, 1.01, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.axvline(2.1, color='#000', ymin=0, ymax=0.68, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.yticks(fontsize=12)
    plt.legend(loc=1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxi3.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == -2:
    env = pd.read_pickle(ModelDir.DATA / "df_env150.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe150.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xlist = list(range(120, 145, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    divid = spline(134)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = xlist
            yi = spline(xi)
            xm = 134
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    A = list(np.arange(0.5, 3.1, 0.1))
    A = [round(x, 2) for x in A]

    xdata = A
    ydata = tempmax
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi2 = list(np.arange(0.5, 3.01, 0.01))
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    xdata = A
    ydata = tempmin
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = list(np.arange(0.5, 3.01, 0.01))
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")
    plt.axhline(1, color='#000', xmin=0, xmax=1, linestyle='--',
                label="Mean value = " + str(np.round(divid, 2)) + " Nm")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Max of the average torque [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.4, 3.1)
    plt.ylim(0.995, 1.011)
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.yticks(list(np.arange(0.996, 1.01, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.axvline(2.1, color='#000', ymin=0, ymax=0.68, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.yticks(fontsize=12)
    plt.legend(loc=1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxi2.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 0:
    res = pd.read_pickle(ModelDir.DATA / "df_rip100.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    mean = [np.mean(res['rip0']), np.mean(res['rip1']), np.mean(res['rip2']), np.mean(res['rip3']), np.mean(res['rip4'])
        , np.mean(res['rip5']), np.mean(res['rip6']), np.mean(res['rip7'])]
    min = [np.min(res['rip0']), np.min(res['rip1']), np.min(res['rip2']), np.min(res['rip3']), np.min(res['rip4'])
        , np.min(res['rip5']), np.min(res['rip6']), np.min(res['rip7'])]
    max = [np.max(res['rip0']), np.max(res['rip1']), np.max(res['rip2']), np.max(res['rip3']), np.max(res['rip4'])
        , np.max(res['rip5']), np.max(res['rip6']), np.max(res['rip7'])]
    plt.scatter(list(range(8)), mean, label="Mean of Sim. Data", color = colors[1])
    plt.scatter(list(range(8)), min, label="Neg. Max. of Sim. Data", color = colors[2])
    plt.scatter(list(range(8)), max, label="Pos. Max. of Sim. Data", color = colors[6])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), mean)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi,yi, label="Approx. of Mean Data", color = colors[1])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), min)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi,yi, label="Approx. of Neg. Min. Data", color = colors[2])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), max)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi,yi, label="Approx. of Pos. Min. Data", color = colors[6])

    plt.xlabel('Mesh size [u.]', fontsize=12)
    plt.ylabel('Absolute difference [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["0.85", "0.7", "0.55", "0.4", "0.25", "0.1", "0.05", "0.025"], fontsize=12)
    plt.yticks(list(np.arange(-0.2, 0.05, 0.025)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    # plt.savefig(ModelDir.MEDIA / "ESCO09.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 1:
    res = pd.read_pickle(ModelDir.DATA / "df_rip250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    mean = [np.mean(res['rip0']), np.mean(res['rip1']), np.mean(res['rip2']), np.mean(res['rip3']), np.mean(res['rip4'])
        , np.mean(res['rip5']), np.mean(res['rip6']), np.mean(res['rip7'])]
    min = [np.min(res['rip0']), np.min(res['rip1']), np.min(res['rip2']), np.min(res['rip3']), np.min(res['rip4'])
        , np.min(res['rip5']), np.min(res['rip6']), np.min(res['rip7'])]
    max = [np.max(res['rip0']), np.max(res['rip1']), np.max(res['rip2']), np.max(res['rip3']), np.max(res['rip4'])
        , np.max(res['rip5']), np.max(res['rip6']), np.max(res['rip7'])]
    plt.scatter(list(range(8)), mean, label="Mean of Sim. Data", color=colors[3])
    plt.scatter(list(range(8)), min, label="Neg. Max. of Sim. Data", color=colors[8])
    plt.scatter(list(range(8)), max, label="Pos. Max. of Sim. Data", color=colors[7])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), mean)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi, yi, label="Approx. of Mean Data", color=colors[3])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), min)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi, yi, label="Approx. of Neg. Min. Data", color=colors[8])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), max)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi, yi, label="Approx. of Pos. Min. Data", color=colors[7])

    plt.xlabel('Mesh size [u.]', fontsize=12)
    plt.ylabel('Absolute difference [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["0.85", "0.7", "0.55", "0.4", "0.25", "0.1", "0.05", "0.025"], fontsize=12)
    plt.yticks(list(np.arange(-0.2, 0.05, 0.025)), fontsize=12)
    plt.yticks(list(np.arange(-0.2, 0.05, 0.025)), fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "ESCO10.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 2:
    fig = plt.subplots(figsize=(6, 4))
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    res = pd.read_pickle(ModelDir.DATA / "df_rip100.pkl")
    ripsum = []
    for x in res['rip0']:
        ripsum.append(x)
    for x in res['rip1']:
        ripsum.append(x)
    for x in res['rip2']:
        ripsum.append(x)
    for x in res['rip3']:
        ripsum.append(x)
    for x in res['rip4']:
        ripsum.append(x)
    for x in res['rip5']:
        ripsum.append(x)
    for x in res['rip6']:
        ripsum.append(x)
    for x in res['rip7']:
        ripsum.append(x)
    x100 = ripsum
    q25, q75 = np.percentile(x100, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x100) ** (-1 / 3)
    bins = round(((max(x100) - min(x100)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    import scipy.stats as st

    plt.hist(x100, density=True, bins=bins, label="Average Operational Torque (AOT)", color=colors[4])
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(x100)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="Probability Density Function (PDF)", color=colors[6])

    plt.xlabel('Absolute difference [Nm]', fontsize=12)
    plt.ylabel('Probability Density [u.]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(-0.1, 0.025)
    plt.ylim(0, 50, 5)
    plt.xticks(list(np.arange(-0.1, 0.03, 0.025)), fontsize=12)
    plt.yticks(list(np.arange(0, 55, 5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "hist100.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 3:
    fig = plt.subplots(figsize=(6, 4))
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    res = pd.read_pickle(ModelDir.DATA / "df_rip250.pkl")
    ripsum = []
    for x in res['rip0']:
        ripsum.append(x)
    for x in res['rip1']:
        ripsum.append(x)
    for x in res['rip2']:
        ripsum.append(x)
    for x in res['rip3']:
        ripsum.append(x)
    for x in res['rip4']:
        ripsum.append(x)
    for x in res['rip5']:
        ripsum.append(x)
    for x in res['rip6']:
        ripsum.append(x)
    for x in res['rip7']:
        ripsum.append(x)
    x100 = ripsum
    q25, q75 = np.percentile(x100, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x100) ** (-1 / 3)
    bins = round(((max(x100) - min(x100)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    import scipy.stats as st

    plt.hist(x100, density=True, bins=bins, label="Average Operational Torque (AOT)", color=colors[6])
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(x100)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="Probability Density Function (PDF)", color=colors[5])

    plt.xlabel('Absolute difference [Nm]', fontsize=12)
    plt.ylabel('Probability Density [u.]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(-0.2, 0.05)
    plt.ylim(0, 20, 2)
    plt.xticks(list(np.arange(-0.2, 0.06, 0.05)), fontsize=12)
    plt.yticks(list(np.arange(0, 22, 2)), fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "hist250.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 4:
    res = pd.read_pickle(ModelDir.DATA / "df_rip150.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))
    mean = [np.mean(res['rip0']), np.mean(res['rip1']), np.mean(res['rip2']), np.mean(res['rip3']), np.mean(res['rip4'])
        , np.mean(res['rip5']), np.mean(res['rip6']), np.mean(res['rip7'])]
    min = [np.min(res['rip0']), np.min(res['rip1']), np.min(res['rip2']), np.min(res['rip3']), np.min(res['rip4'])
        , np.min(res['rip5']), np.min(res['rip6']), np.min(res['rip7'])]
    max = [np.max(res['rip0']), np.max(res['rip1']), np.max(res['rip2']), np.max(res['rip3']), np.max(res['rip4'])
        , np.max(res['rip5']), np.max(res['rip6']), np.max(res['rip7'])]
    plt.scatter(list(range(8)), mean, label="Mean of Sim. Data", color = colors[1])
    plt.scatter(list(range(8)), min, label="Neg. Max. of Sim. Data", color = colors[2])
    plt.scatter(list(range(8)), max, label="Pos. Max. of Sim. Data", color = colors[6])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), mean)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi,yi, label="Approx. of Mean Data", color = colors[1])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), min)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi,yi, label="Approx. of Neg. Min. Data", color = colors[2])
    spline = interpolate.InterpolatedUnivariateSpline(list(range(8)), max)
    xi = np.arange(0, 7.1, 0.1)
    yi = spline(xi)
    plt.plot(xi,yi, label="Approx. of Pos. Min. Data", color = colors[6])

    plt.xlabel('Mesh size [u.]', fontsize=12)
    plt.ylabel('Absolute difference [Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["0.85", "0.7", "0.55", "0.4", "0.25", "0.1", "0.05", "0.025"], fontsize=12)
    plt.yticks(list(np.arange(-0.2, 0.05, 0.025)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "ESCO13.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 5:
    fig = plt.subplots(figsize=(6, 4))
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    res = pd.read_pickle(ModelDir.DATA / "df_rip150.pkl")
    ripsum = []
    for x in res['rip0']:
        ripsum.append(x)
    for x in res['rip1']:
        ripsum.append(x)
    for x in res['rip2']:
        ripsum.append(x)
    for x in res['rip3']:
        ripsum.append(x)
    for x in res['rip4']:
        ripsum.append(x)
    for x in res['rip5']:
        ripsum.append(x)
    for x in res['rip6']:
        ripsum.append(x)
    for x in res['rip7']:
        ripsum.append(x)
    x100 = ripsum
    q25, q75 = np.percentile(x100, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x100) ** (-1 / 3)
    bins = round((max(x100) - min(x100) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    import scipy.stats as st

    plt.hist(x100, density=True, bins=bins, label="Average Torque", color=colors[3])
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(x100)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="Probability Density Function (PDF)", color=colors[4])

    plt.xlabel('Absolute difference [Nm]', fontsize=12)
    plt.ylabel('Probability [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(-0.2, 0.05)
    plt.xticks(list(np.arange(-0.2, 0.06, 0.05)), fontsize=12)
    plt.yticks(list(np.arange(0, 45, 5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "hist150.png", bbox_inches="tight", dpi=650)
    plt.show()

elif switch == 6:
    fig = plt.subplots(figsize=(6, 4))
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    res = pd.read_pickle(ModelDir.DATA / "df_rip200.pkl")
    ripsum = []
    for x in res['rip0']:
        ripsum.append(x)
    for x in res['rip1']:
        ripsum.append(x)
    for x in res['rip2']:
        ripsum.append(x)
    for x in res['rip3']:
        ripsum.append(x)
    for x in res['rip4']:
        ripsum.append(x)
    for x in res['rip5']:
        ripsum.append(x)
    for x in res['rip6']:
        ripsum.append(x)
    for x in res['rip7']:
        ripsum.append(x)
    x100 = ripsum
    q25, q75 = np.percentile(x100, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x100) ** (-1 / 3)
    bins = round((max(x100) - min(x100) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    import scipy.stats as st

    plt.hist(x100, density=True, bins=bins, label="Average Torque", color=colors[5])
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(x100)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="Probability Density Function (PDF)", color=colors[6])

    plt.xlabel('Absolute difference [Nm]', fontsize=12)
    plt.ylabel('Probability [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(-0.2, 0.05)
    plt.xticks(list(np.arange(-0.2, 0.06, 0.05)), fontsize=12)
    plt.yticks(list(np.arange(0, 45, 5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "hist200.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 10:
    res = pd.read_pickle(ModelDir.DATA / "df_wav100.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xlist = list(np.arange(0.5, 3.01, 0.01))

    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['max'])
    xi2 = xlist
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['min'])
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.ylim(9, 11.5)
    plt.yticks(list(np.arange(9, 12, 0.5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(res['max'].mean(), color='#000', xmin=0, xmax=1, linestyle='--', label="Mean amplitude (" + str(np.round(res['max'].mean(), 2)) + "%)")
    plt.axhline(res['min'].mean(), color='#000', xmin=0, xmax=1, linestyle='-.', label="Mean amplitude (" + str(np.round(res['min'].mean(), 2)) + "%)")
    plt.axvline(2.1, color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.legend(loc=2, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxrip1.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 11:
    res = pd.read_pickle(ModelDir.DATA / "df_wav150.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    # plt.scatter(res["A"], res['min'], color=colors[4])
    # plt.scatter(res["A"], res['max'], color=colors[0])

    xlist = list(np.arange(0.5, 3.01, 0.01))
    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['min'])
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['max'])
    xi2 = xlist
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.ylim(8.25, 11.25)
    plt.yticks(list(np.arange(8.5, 11.5, 0.5)), fontsize=12)
    plt.axhline(8.73, color='#000', xmin=0, xmax=1, linestyle='--', label="Mean amplitude (8.73 %)")
    plt.axvline(2.1, color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.yticks(fontsize=12)
    plt.legend(loc=2, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxrip2.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 12:
    res = pd.read_pickle(ModelDir.DATA / "df_wav200.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    # plt.scatter(res["A"], res['min'], color=colors[4])
    # plt.scatter(res["A"], res['max'], color=colors[0])

    xlist = list(np.arange(0.5, 3.01, 0.01))
    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['min'])
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['max'])
    xi2 = xlist
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.ylim(8.25, 11.25)
    plt.yticks(list(np.arange(8.5, 11.5, 0.5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(9.63, color='#000', xmin=0, xmax=1, linestyle='--', label="Mean amplitude (9.63 %)")
    plt.axvline(2.1, color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.legend(loc=2, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxrip3.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 13:
    res = pd.read_pickle(ModelDir.DATA / "df_wav250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    xlist = list(np.arange(0.5, 3.01, 0.01))
    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['max'])
    xi2 = xlist
    yi2 = spline(xi2)
    plt.plot(xi2, yi2, label="Upper envelope curve", color=colors[0])

    spline = interpolate.InterpolatedUnivariateSpline(res["A"], res['min'])
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, label="Lower envelope curve", color=colors[4])

    plt.fill_between(xi1, yi1, yi2, color="gainsboro")

    plt.xlabel('Parameter A [mm]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(list(np.arange(0.5, 3.1, 0.5)), fontsize=12)
    plt.ylim(11, 12.25)
    plt.yticks(list(np.arange(11, 12.5, 0.25)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(res['max'].mean(), color='#000', xmin=0, xmax=1, linestyle='--',
                label="Mean amplitude (" + str(np.round(res['max'].mean(), 2)) + "%)")
    plt.axhline(res['min'].mean(), color='#000', xmin=0, xmax=1, linestyle='-.',
                label="Mean amplitude (" + str(np.round(res['min'].mean(), 2)) + "%)")
    plt.axvline(2.1, color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.legend(loc=2, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "maxrip4.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 20:
    res = pd.read_pickle(ModelDir.DATA / "df_wav100.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env100.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe100.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    num = 130
    xlist = list(range(116, 141, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    divid = spline(num)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = xlist
            yi = spline(xi)
            xm = num
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    res1 = {'y': res['max'],
            'x': tempmax}
    res1 = pd.DataFrame(res1)
    res1.sort_values(by='x')
    res2 = {'y': res['min'],
            'x': tempmin}
    res2 = pd.DataFrame(res2)
    res2.sort_values(by='x')

    plt.scatter(res2['x'], res2['y'], label = 'Solution (fine mesh)', color=colors[3])
    plt.scatter(res1['x'], res1['y'], label = 'Solution (coarse mesh)', color=colors[0])

    plt.xlabel('Maximum of AOT [%]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.995, 1.011)
    plt.xticks(list(np.arange(0.996, 1.011, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.ylim(9, 11.5)
    plt.yticks(list(np.arange(9, 12, 0.5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(res1['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--', label="A = 2.1 mm design")
    plt.axvline(res1['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.axhline(res2['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--',)
    plt.axvline(res2['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', linewidth=2)
    plt.legend(loc=4, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "opt100.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 21:
    res = pd.read_pickle(ModelDir.DATA / "df_wav150.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env150.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe150.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    num = 134
    xlist = list(range(120, 145, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    divid = spline(num)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = xlist
            yi = spline(xi)
            xm = num
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    res1 = {'y': res['max'],
            'x': tempmax}
    res1 = pd.DataFrame(res1)
    res1.sort_values(by='x')
    res2 = {'y': res['min'],
            'x': tempmin}
    res2 = pd.DataFrame(res2)
    res2.sort_values(by='x')

    plt.scatter(res2['x'], res2['y'], label = 'Solution (fine mesh)', color=colors[3])
    plt.scatter(res1['x'], res1['y'], label = 'Solution (coarse mesh)', color=colors[0])

    plt.xlabel('Maximum of AOT [%]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.995, 1.011)
    plt.xticks(list(np.arange(0.996, 1.011, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.ylim(8.25, 11.25)
    plt.yticks(list(np.arange(8.5, 11.5, 0.5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(res1['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--', label="A = 2.1 mm design")
    plt.axvline(res1['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    # plt.axhline(res2['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--',)
    plt.axvline(res2['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', linewidth=2)
    plt.legend(loc=1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "opt150.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 22:
    res = pd.read_pickle(ModelDir.DATA / "df_wav200.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env200.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe200.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    num = 137
    xlist = list(range(124, 149, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    divid = spline(num)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = xlist
            yi = spline(xi)
            xm = num
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    res1 = {'y': res['max'],
            'x': tempmax}
    res1 = pd.DataFrame(res1)
    res1.sort_values(by='x')
    res2 = {'y': res['min'],
            'x': tempmin}
    res2 = pd.DataFrame(res2)
    res2.sort_values(by='x')

    plt.scatter(res2['x'], res2['y'], label = 'Solution (fine mesh)', color=colors[3])
    plt.scatter(res1['x'], res1['y'], label = 'Solution (coarse mesh)', color=colors[0])

    plt.xlabel('Maximum of AOT [%]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.995, 1.011)
    plt.xticks(list(np.arange(0.996, 1.011, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.ylim(8.25, 11.25)
    plt.yticks(list(np.arange(8.5, 11.5, 0.5)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(res1['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--', label="A = 2.1 mm design")
    plt.axvline(res1['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    plt.axhline(res2['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--',)
    plt.axvline(res2['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', linewidth=2)
    plt.legend(loc=4, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "opt200.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 23:
    res = pd.read_pickle(ModelDir.DATA / "df_wav250.pkl")
    env = pd.read_pickle(ModelDir.DATA / "df_env250.pkl")
    doe = pd.read_pickle(ModelDir.DATA / "df_doe250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    num = 138.2
    xlist = list(range(124, 149, 1))
    xdata = env["rot"]
    ydata = env["avgt"]
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi3 = xlist
    yi3 = spline(xi3)
    divid = spline(num)

    tempmin = []
    tempmax = []
    for j in range(26):
        temp = []
        for i in range((j * 9) + 0, (j * 9) + 9):
            xdata = env['rot']
            ydata = doe['avg'].loc[i]
            spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
            xi = xlist
            yi = spline(xi)
            xm = num
            ym = spline(xm)
            ym = float(ym)
            temp.append(ym)
        mn = min(temp) / divid
        mx = max(temp) / divid
        tempmax.append(mx)
        tempmin.append(mn)

    res1 = {'y': res['max'],
            'x': tempmax}
    res1 = pd.DataFrame(res1)
    res1.sort_values(by='x')
    res2 = {'y': res['min'],
            'x': tempmin}
    res2 = pd.DataFrame(res2)
    res2.sort_values(by='x')

    plt.scatter(res2['x'], res2['y'], label = 'Solution (fine mesh)', color=colors[3])
    plt.scatter(res1['x'], res1['y'], label = 'Solution (coarse mesh)', color=colors[0])

    plt.xlabel('Maximum of AOT [%]', fontsize=12)
    plt.ylabel('Torque ripple [%]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xlim(0.995, 1.011)
    plt.xticks(list(np.arange(0.996, 1.011, 0.002)), list(np.round(np.arange(99.6, 101, 0.2), 2)), fontsize=12)
    plt.ylim(11, 12.25)
    plt.yticks(list(np.arange(11, 12.5, 0.25)), fontsize=12)
    plt.axhline(res1['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--', label="A = 2.1 mm design")
    plt.axvline(res1['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', label="A = 2.1 mm design", linewidth=2)
    # plt.axhline(res2['y'].loc[16], color='#000', xmin=0, xmax=1, linestyle='--',)
    plt.axvline(res2['x'].loc[16], color='#000', ymin=0, ymax=1, linestyle=':', linewidth=2)
    plt.legend(loc=1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "opt250.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 89:
    res1 = pd.read_pickle(ModelDir.DATA / "df_doe100.pkl")
    res2 = pd.read_pickle(ModelDir.DATA / "df_doe150.pkl")
    res3 = pd.read_pickle(ModelDir.DATA / "df_doe200.pkl")
    res4 = pd.read_pickle(ModelDir.DATA / "df_doe250.pkl")
    fig = plt.subplots(figsize=(6, 4))
    line_props = dict(color='#005691', alpha=1, linewidth=1.5)
    bbox_props = dict(color='#005691', alpha=1, linewidth=1.5)
    flier_props = dict(marker="x", markersize=5)
    cap_props = dict(color='#005691', alpha=1, linewidth=1.5)
    median_props = dict(color="#B90276", alpha=1, linewidth=1.5)
    mean_props = dict(color="#B90276", alpha=1)
    plt.boxplot([res1["maxt"]/max(res1["maxt"]), res2["maxt"]/max(res2["maxt"]), res3["maxt"]/max(res3["maxt"]), res4["maxt"]/max(res4["maxt"])],
                positions=range(4), notch=True, whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props
                , capprops=cap_props, medianprops=median_props, meanprops=mean_props, showmeans=True)
    plt.xlabel('Current [A]', fontsize=12)
    plt.ylabel('Max of the specific torque [Nm/Nm]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks([0,1,2,3], [100, 150, 200, 250], fontsize=12)
    plt.yticks(list(np.arange(0.984, 1, 0.002)), fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(ModelDir.MEDIA / "stat1.png", bbox_inches="tight", dpi=650)
    plt.show()

    if switch == 99:
        res = pd.read_pickle(ModelDir.DATA / "df_msh250.pkl")
        colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
        fig = plt.subplots(figsize=(6, 4))
        x = 4
        a = 0 + x * 81
        b = 81 + x * 81
        y = 0
        xdata = []
        ydata = []
        for i in range(a, b, 9):
            plt.scatter(res["meshing"].iloc[6 + i], res["tavg"].iloc[6 + i], color=colors[y])
            xdata.append(res["meshing"].iloc[6 + i])
            ydata.append(res["tavg"].iloc[6 + i])
        xdatar = list(reversed(xdata))
        del xdatar[2]
        del ydata[6]
        print(ydata)
        spline = interpolate.InterpolatedUnivariateSpline(xdatar, ydata)
        xi = np.linspace(0.025, 1, 100)
        yi = spline(np.linspace(0.025, 1, 100))
        xir = list(reversed(xi))
        plt.plot(xir, yi, color=colors[0], label="Trendvonal")
        plt.gca().invert_xaxis()
        plt.xlabel('Hálózási sűrűség [e.]', fontsize=12)
        plt.ylabel('Nyomaték [Nm]', fontsize=12)
        plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
        plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
        plt.minorticks_on()
        plt.xticks([1, 0.85, 0.70, 0.55, 0.40, 0.25, 0.025], [1, 0.85, 0.70, 0.55, 0.40, 0.25, 0.025], fontsize=12)
        plt.yticks(fontsize=12)
        plt.scatter(res["meshing"].iloc[6 + i], res["tavg"].iloc[6 + i], color=colors[y], label="Szimulációs eredmények")
        plt.legend(fontsize=10, loc=3)
        plt.savefig(ModelDir.MEDIA / f'mait12.png', bbox_inches="tight", dpi=650)
        plt.show

