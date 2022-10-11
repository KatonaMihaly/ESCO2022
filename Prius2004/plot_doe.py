import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from digital_twin_distiller import ModelDir
import pandas as pd
import numpy as np
from scipy import interpolate
import scipy.stats as st

ModelDir.set_base(__file__)

switch = 12
#10 - esco_fig12a
#11 - esco_fig12b
#12 - esco_fig11
if switch == -5:
    bb = pd.read_pickle(ModelDir.DATA / "res_dif250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_dif250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_dif250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_dif250_pb.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))
    xlist = np.arange(0, 61, 0.1)
    x = range(61)

    xdata = x
    ydata = bb['emax']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    plt.plot(xi0, yi0)

    xdata = x
    ydata = bb['emin']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1)

    xdata = x
    ydata = pb['emax']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    plt.plot(xi0, yi0, linestyle = '--')

    xdata = x
    ydata = pb['emin']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1, linestyle = '--')

    # plt.fill_between(xi0, yi0, yi1, color="gainsboro")

    plt.show()

if switch == -4:
    res = pd.read_pickle(ModelDir.DATA / "res_dif250_pb.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))
    xlist = np.arange(0, 61, 0.1)
    x = range(61)

    xdata = x
    ydata = res['emax']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    plt.plot(xi0, yi0)

    xdata = x
    ydata = res['emin']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1)

    plt.fill_between(xi0, yi0, yi1, color="gainsboro")

    plt.show()

if switch == -3:
    res = pd.read_pickle(ModelDir.DATA / "res_dif250_wc.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))
    xlist = np.arange(0, 61, 0.1)
    x = range(61)

    xdata = x
    ydata = res['emax']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    plt.plot(xi0, yi0)

    xdata = x
    ydata = res['emin']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1)

    plt.fill_between(xi0, yi0, yi1, color="gainsboro")

    plt.show()

if switch == -2:
    res = pd.read_pickle(ModelDir.DATA / "res_dif250_cc.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))
    xlist = np.arange(0, 61, 0.1)
    x = range(61)

    xdata = x
    ydata = res['emax']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    plt.plot(xi0, yi0)

    xdata = x
    ydata = res['emin']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1)

    plt.fill_between(xi0, yi0, yi1, color="gainsboro")

    plt.show()

if switch == -1:
    res = pd.read_pickle(ModelDir.DATA / "res_dif250_bb.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))
    xlist = np.arange(0, 61, 0.1)
    x = range(61)

    xdata = x
    ydata = res['emax']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi0 = xlist
    yi0 = spline(xi0)
    plt.plot(xi0, yi0)

    xdata = x
    ydata = res['emin']
    spline = interpolate.InterpolatedUnivariateSpline(xdata, ydata)
    xi1 = xlist
    yi1 = spline(xi1)
    plt.plot(xi1, yi1)

    plt.fill_between(xi0, yi0, yi1, color="gainsboro")

    plt.show()

if switch == 0:
    ref = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    for i in range(len(ref)):
        plt.plot(ref['rot'].loc[i], ref['rip'].loc[i])

    plt.axhline(ref['min'].loc[0]+ref['amp'].loc[0])

    print(ref)
    print(ref['max'].loc[0]-ref['avg'].loc[0])

    # for i in range(len(ref)):
    #     plt.scatter(i, ref['avg'].loc[i])
    # plt.show()
    #
    # for i in range(len(ref)):
    #     plt.scatter(i, ref['amp'].loc[i])
    plt.show()

if switch == 1:
    ref = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    for i in range(len(ref)):
        plt.plot(ref['rot'].loc[i], ref['rip'].loc[i])
    plt.show()

    for i in range(len(ref)):
        plt.scatter(i, ref['avg'].loc[i])
    plt.show()

    for i in range(len(ref)):
        plt.scatter(i, ref['amp'].loc[i])
    plt.show()

if switch == 2:
    ref = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    for i in range(len(ref)):
        plt.plot(ref['rot'].loc[i], ref['rip'].loc[i])
    plt.show()

    for i in range(len(ref)):
        plt.scatter(i, ref['avg'].loc[i])
    plt.show()

    for i in range(len(ref)):
        plt.scatter(i, ref['amp'].loc[i])
    plt.show()

if switch == 3:
    ref = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    for i in range(len(ref)):
        plt.plot(ref['rot'].loc[i], ref['rip'].loc[i])
    plt.show()

    for i in range(len(ref)):
        plt.scatter(i, ref['avg'].loc[i])
    plt.show()

    for i in range(len(ref)):
        plt.scatter(i, ref['amp'].loc[i])
    plt.show()

if switch == 4:
    bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig, ax1 = plt.subplots(figsize=(6, 4))

    x = list(range(len(bb)))
    x = [i / len(bb) for i in x]
    y = list(bb['avg'])
    y.sort()
    plt.scatter(x, y, c='b')

    x = list(range(len(cc)))
    x = [i / len(cc) for i in x]
    y = list(cc['avg'])
    y.sort()
    plt.scatter(x, y, c='r')

    x = list(range(len(wc)))
    x = [i / len(wc) for i in x]
    y = list(wc['avg'])
    y.sort()
    plt.scatter(x, y, c='g')

    x = list(range(len(pb)))
    x = [i / len(pb) for i in x]
    y = list(pb['avg'])
    y.sort()
    plt.scatter(x, y, c='purple')
    # for i in range(len(wc)):
    #     plt.scatter(i-len(wc)/2, wc['avg'].loc[i], c='g')
    # for i in range(len(pb)):
    #     plt.scatter(i-len(pb)/2, pb['avg'].loc[i], c='y')
    # for i in range(len(bb)):
    #     plt.scatter(i - len(bb) / 2, bb['avg'].loc[i], c='r')
    plt.show()

if switch == 5:
    bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")

    lista = [list(wc['avg']), list(cc['avg']), list(pb['avg']), list(bb['avg'])]
    for i in range(len(lista)):
        lista[i].sort()

    fig = plt.subplots(figsize=(6, 4))
    line_props = dict(color='#005691', alpha=1, linewidth=1.5)
    bbox_props = dict(color='#005691', alpha=1, linewidth=1.5)
    flier_props = dict(marker="x", markersize=5)
    cap_props = dict(color='#005691', alpha=1, linewidth=1.5)
    median_props = dict(color="#B90276", alpha=1, linewidth=1.5)
    mean_props = dict(color="#B90276", alpha=1)
    plt.boxplot(lista,
                positions=range(len(lista)), notch=True, whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props
                , capprops=cap_props, medianprops=median_props, meanprops=mean_props, showmeans=True)
    # plt.xlabel('Current [A]', fontsize=12)
    # plt.ylabel('Max of the specific torque [Nm/Nm]', fontsize=12)
    # plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    # plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    # plt.minorticks_on()
    # plt.xticks([0, 1, 2, 3], [100, 150, 200, 250], fontsize=12)
    # plt.yticks(list(np.arange(0.984, 1, 0.002)), fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.savefig(ModelDir.MEDIA / "stat1.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 6:
    bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")

    lista = [list(wc['min']), list(cc['min']), list(pb['min']), list(bb['min'])]
    for i in range(len(lista)):
        lista[i].sort()

    fig = plt.subplots(figsize=(6, 4))
    line_props = dict(color='#005691', alpha=1, linewidth=1.5)
    bbox_props = dict(color='#005691', alpha=1, linewidth=1.5)
    flier_props = dict(marker="x", markersize=5)
    cap_props = dict(color='#005691', alpha=1, linewidth=1.5)
    median_props = dict(color="#B90276", alpha=1, linewidth=1.5)
    mean_props = dict(color="#B90276", alpha=1)
    plt.boxplot(lista,
                positions=range(len(lista)), notch=True, whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props
                , capprops=cap_props, medianprops=median_props, meanprops=mean_props, showmeans=True)
    # plt.xlabel('Current [A]', fontsize=12)
    # plt.ylabel('Max of the specific torque [Nm/Nm]', fontsize=12)
    # plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    # plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    # plt.minorticks_on()
    # plt.xticks([0, 1, 2, 3], [100, 150, 200, 250], fontsize=12)
    # plt.yticks(list(np.arange(0.984, 1, 0.002)), fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.savefig(ModelDir.MEDIA / "stat1.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 7:
    bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")

    lista = [list(wc['max']), list(cc['max']), list(pb['max']), list(bb['max'])]
    for i in range(len(lista)):
        lista[i].sort()

    fig = plt.subplots(figsize=(6, 4))
    line_props = dict(color='#005691', alpha=1, linewidth=1.5)
    bbox_props = dict(color='#005691', alpha=1, linewidth=1.5)
    flier_props = dict(marker="x", markersize=5)
    cap_props = dict(color='#005691', alpha=1, linewidth=1.5)
    median_props = dict(color="#B90276", alpha=1, linewidth=1.5)
    mean_props = dict(color="#B90276", alpha=1)
    plt.boxplot(lista,
                positions=range(len(lista)), notch=True, whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props
                , capprops=cap_props, medianprops=median_props, meanprops=mean_props, showmeans=True)
    # plt.xlabel('Current [A]', fontsize=12)
    # plt.ylabel('Max of the specific torque [Nm/Nm]', fontsize=12)
    # plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    # plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    # plt.minorticks_on()
    # plt.xticks([0, 1, 2, 3], [100, 150, 200, 250], fontsize=12)
    # plt.yticks(list(np.arange(0.984, 1, 0.002)), fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.savefig(ModelDir.MEDIA / "stat1.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 9:
    bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']
    fig = plt.subplots(figsize=(6, 4))

    x = 1/6
    plt.axhline(bb['avg'].min(), xmin=3.5*x, xmax=4.5*x, linestyle='--', color=colors[1], label = 'Box-Behnken')
    plt.axhline(bb['avg'].max(), xmin=3.5*x, xmax=4.5*x, linestyle='--', color=colors[1])
    plt.axhline(cc['avg'].min(), xmin=1.5*x, xmax=2.5*x, linestyle='-.', color=colors[2], label = 'Central Composit')
    plt.axhline(cc['avg'].max(), xmin=1.5*x, xmax=2.5*x, linestyle='-.', color=colors[2])
    plt.axhline(wc['avg'].min(), xmin=2.5*x, xmax=3.5*x, linestyle='dotted', color=colors[3], label = 'Worst Case')
    plt.axhline(wc['avg'].max(), xmin=2.5*x, xmax=3.5*x, linestyle='dotted', color=colors[4])
    plt.axhline(pb['avg'].min(), xmin=4.5*x, xmax=5.5*x, linestyle=':', color=colors[5], label = 'Plackett-Burman')
    plt.axhline(pb['avg'].max(), xmin=4.5*x, xmax=5.5*x, linestyle=':', color=colors[5])

    plt.axvline(0.5, ymin=0.0, ymax=0.9)
    plt.axvline(1.5, ymin=0.0, ymax=0.9)
    plt.axvline(2.5, ymin=0.0, ymax=0.9)
    plt.axvline(3.5, ymin=0.0, ymax=0.9)
    plt.axvline(4.5, ymin=0.0, ymax=0.9)
    plt.axvline(5.5, ymin=0.0, ymax=0.9)

    plt.xlabel('Number of DOE cases [u.]', fontsize=12)
    plt.ylabel('Average torque [%]', fontsize=12)
    plt.xlim(-1, 5)
    plt.ylim(315.75, 318.25)
    plt.xticks([0, 1, 2, 3, 4], ['6561 \n (FF)', '273 \n (CC)', '256 \n (WC)', '113 \n (BB)', '12 \n (PB)'],
               fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc=1, fontsize=10)
    # plt.savefig(ModelDir.MEDIA / "doe0.png", bbox_inches="tight", dpi=650)
    plt.show()
    r"My long label with $\Sigma_{C}$ math \n continues here"

if switch == 10:
    bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")
    ff = pd.read_pickle(ModelDir.DATA / "res_ref250_ff.pkl")
    base = pd.read_pickle(ModelDir.DATA / "res_base250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']

    mini = 315.964
    maxi = 317.986

    x = list(ff['avg'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Full Factorial", color=colors[0])

    x = list(wc['avg'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Central Composite", color=colors[1])

    x = list(cc['avg'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Fractional Factorial", color=colors[2])

    x = list(bb['avg'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Box-Behnken", color=colors[3])

    x = list(pb['avg'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Plackett-Burman", color=colors[5])

    plt.axvline(base['avg'].loc[0], linestyle='--', linewidth=3, color=colors[8], label='Original (317.02 Nm)')
    plt.axvline(mini, linestyle='--', linewidth=2, ymax=0.1, color=colors[8], label='Min (315.96 Nm)')
    plt.axvline(maxi, linestyle='--', linewidth=2, ymax=0.1, color=colors[8], label='Max (317.99 Nm)')


    plt.xlabel('Maximum of AOT [Nm]', fontsize=12)
    plt.ylabel('Frequency [u.]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(315.75, 318.5, 0.5), fontsize=12)
    # plt.xticks(np.arange(315.75, 318.5, 0.5),
    #            np.round(np.arange(315.75 / base['avg'].loc[0], 318.5 / base['avg'].loc[0], 0.5 / base['avg'].loc[0]),
    #                     3), fontsize=12)
    plt.yticks(np.arange(0, 450, 50), fontsize=12)
    plt.legend(loc=1, fontsize=10)
    plt.savefig(ModelDir.MEDIA / "doe1.png", bbox_inches="tight", dpi=650)
    plt.show()

if switch == 11:
    switch = 1
    if switch == 0:
        bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
        cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
        wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
        pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")
        ff = pd.read_pickle(ModelDir.DATA / "res_ref250_ff.pkl")
        base = pd.read_pickle(ModelDir.DATA / "res_base250.pkl")
        colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']

        dev = 0.390
        mini = 315.964
        maxi = 317.986
        meani =  316.971
        a = meani - dev
        b = meani + dev

        mn, mx = [315.5, 318.5]
        reso = 1000

        x = list(ff['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, reso)
        kdeff = st.gaussian_kde(x)
        plt.plot(kde_xs, kdeff.pdf(kde_xs), label="Full Factorial", color=colors[0], linestyle='-', linewidth=2)
        pff = kdeff.integrate_box_1d(a, b)

        x = list(cc['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, reso)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Central Composite", color=colors[1], linestyle='--', linewidth=2)
        pcc = kde.integrate_box_1d(a, b)

        x = list(wc['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, reso)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Worst Case", color=colors[2], linestyle='-.', linewidth=2)
        pwc = kde.integrate_box_1d(a, b)

        x = list(bb['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, reso)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Box-Behnken", color=colors[3], linestyle='--', linewidth=2)
        pbb = kde.integrate_box_1d(a, b)

        x = list(pb['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, reso)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Plackett-Burman", color=colors[5], linestyle='-', linewidth=2.5)
        ppb = kde.integrate_box_1d(a, b)

        plt.axvline(base['avg'].loc[0], linestyle='--', linewidth=3, color=colors[8], label='Original')

        plt.axvline(mini, linestyle='--', linewidth=2, ymax=1, color=colors[8], label=str(mini) + ' Nm')
        plt.axvline(maxi, linestyle='--', linewidth=2, ymax=0.5, color=colors[8], label=str(maxi) + ' Nm')

        plt.axvline(a, linestyle=':', linewidth=2, ymax=0.31, color=colors[8])
        plt.axvline(b, linestyle=':', linewidth=2, ymax=0.31, color=colors[8])

        print([pff*100, pcc*100, pwc*100, pbb*100, ppb*100])

        x = list(ff['avg'])
        kde = st.gaussian_kde(x)
        plt.fill_between(np.linspace(a, b, 300), 0, kde.pdf(np.linspace(a, b, 300)), color="gainsboro")

        plt.xlabel('Maximum of AOT [Nm]', fontsize=12)
        plt.ylabel('Probability Density [u.]', fontsize=12)
        plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
        plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
        plt.minorticks_on()
        plt.ylim(0, 1.5)
        plt.xticks(np.arange(mn, mx+0.5, 0.5), fontsize=12)
        plt.yticks(np.arange(0, 1.65, 0.15), fontsize=12)
        plt.legend(loc=1, fontsize=10)
        # plt.savefig(ModelDir.MEDIA / "doe_pdf.png", bbox_inches="tight", dpi=650)
        plt.show()

    if switch == 1:
        bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
        cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
        wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
        pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")
        ff= pd.read_pickle(ModelDir.DATA / "res_ref250_ff.pkl")
        base = pd.read_pickle(ModelDir.DATA / "res_base250.pkl")
        colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']

        x = list(ff['avg'])
        mn, mx = [315.5, 318.5] #lehet állítani kell
        mini = 315.96
        maxi = 317.99
        sigma = (maxi-mini)/12

        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Full Factorial", color=colors[0], linestyle='-', linewidth=2.5)
        lsl = np.min(x)
        usl = np.max(x)
        msl = np.mean(x)
        sigma = np.std(x)
        a = np.round(msl - sigma, 2)
        b = np.round(msl + sigma, 2)
        pff = kde.integrate_box_1d(a, b)

        x = list(cc['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Central Composite", color=colors[1], linestyle='--', linewidth=2)
        pcc = kde.integrate_box_1d(a, b)

        x = list(wc['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Fractional Factorial", color=colors[2], linestyle='-.', linewidth=2)
        pwc = kde.integrate_box_1d(a, b)

        x = list(bb['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Box-Behnken", color=colors[3], linestyle='--', linewidth=2)
        pbb = kde.integrate_box_1d(a, b)

        x = list(pb['avg'])
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(x)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="Plackett-Burman", color=colors[5], linestyle=':', linewidth=2.5)
        ppb = kde.integrate_box_1d(a, b)

        # plt.axvline(317.02, linestyle='--', linewidth=3, color=colors[8], label='Original (317.02 Nm')

        plt.axvline(mini, linestyle='--', linewidth=2, ymax=0.0, color=colors[8], label='Min (' + str(mini) + ' Nm)')
        plt.axvline(maxi, linestyle='--', linewidth=2, ymax=0.0, color=colors[8], label='Max (' + str(maxi) + ' Nm)')

        plt.axvline(a, linestyle=':', linewidth=2, ymax=0.31, color=colors[8])
        plt.axvline(b, linestyle=':', linewidth=2, ymax=0.31, color=colors[8])

        print([pff * 100, pcc * 100, pwc * 100, pbb * 100, ppb * 100])

        x = list(ff['avg'])
        kde = st.gaussian_kde(x)
        plt.fill_between(np.linspace(a, b, 300), 0, kde.pdf(np.linspace(a, b, 300)), color="gainsboro")

        plt.xlabel('Maximum of AOT [Nm]', fontsize=12)
        plt.ylabel('Probability Density [u.]', fontsize=12)
        plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
        plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
        plt.minorticks_on()
        plt.ylim(0, 1.5)
        plt.xticks(np.arange(mn, mx + 0.5, 0.5), fontsize=12)
        plt.yticks(np.arange(0, 1.65, 0.15), fontsize=12)
        plt.legend(loc=1, fontsize=10)
        plt.savefig(ModelDir.MEDIA / "doe2.png", bbox_inches="tight", dpi=650)
        plt.show()

if switch == 12:
    bb = pd.read_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    cc = pd.read_pickle(ModelDir.DATA / "res_ref250_cc.pkl")
    wc = pd.read_pickle(ModelDir.DATA / "res_ref250_wc.pkl")
    pb = pd.read_pickle(ModelDir.DATA / "res_ref250_pb.pkl")
    ff = pd.read_pickle(ModelDir.DATA / "res_ref250_ff.pkl")
    base = pd.read_pickle(ModelDir.DATA / "res_base250.pkl")
    colors = ["#B90276", '#50237F', '#005691', "#008ECF", '#00A8B0', '#78BE20', "#006249", '#525F6B', '#000']

    mini = 35.55
    maxi = 36.92
    meani = 36.25

    x = list(ff['amp'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Full Factorial", color=colors[0])

    x = list(cc['amp'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Central Composite", color=colors[1])

    x = list(wc['amp'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Fractional Factorial", color=colors[2])

    x = list(bb['amp'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Box-Behnken", color=colors[3])

    x = list(pb['amp'])
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round(((max(x) - min(x)) / bin_width))
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(x, density=False, bins=bins, label="Plackett-Burman", color=colors[5])

    plt.axvline(base['amp'].loc[0], linestyle='--', linewidth=3, color=colors[8], label='Original (36.25 Nm)')
    plt.axvline(mini, linestyle='--', linewidth=2, ymax=0.2, color=colors[8], label='Min (' + str(mini) + ' Nm)')
    plt.axvline(maxi, linestyle='--', linewidth=2, ymax=0.2, color=colors[8], label='Max (' + str(maxi) + ' Nm)')

    plt.xlabel('Amplitude of the torque ripple [Nm]', fontsize=12)
    plt.ylabel('Frequency [u.]', fontsize=12)
    plt.grid(visible=True, which="major", color="#666666", linestyle="-", linewidth=0.8)
    plt.grid(visible=True, which="minor", color="#999999", linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.xticks(np.arange(35.5, 37.25, 0.25), fontsize=12)
    plt.yticks(np.arange(0, 450, 50), fontsize=12)
    plt.legend(fontsize=10)
    plt.savefig(ModelDir.MEDIA / "doe3.png", bbox_inches="tight", dpi=650)
    plt.show()

