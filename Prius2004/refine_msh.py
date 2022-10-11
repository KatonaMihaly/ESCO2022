from operator import add

import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from numpy import linspace
import json

from scipy.interpolate import interpolate

from digital_twin_distiller import ModelDir

ModelDir.set_base(__file__)

switch = 3
if switch == 0:

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 1.0
    range_b1 = 0.1
    nsteps_b = 7

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_250.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        a = a + 1
        b = b + 61

    for i in range(int(ran/nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res1 = {"rotorangle": rota,
           "tavg": tavg,
           "tmin": tmin,
           "tmax": tmax,
           "earheight": ear,
           "meshing": mesh}
    res1 = pd.DataFrame(res1)

    # dif = []
    # y0 = 0
    # y1 = 0
    # y2 = 54
    # while y0 < 26:
    #     for x in range(y1, y2):
    #         a = (res["tavg"])[0+x]
    #         b = (res["tavg"])[9+x]
    #         c = b - a
    #         dif.append(c)
    #     for z in range(9):
    #         dif.append(np.NaN)
    #     y0 = y0 + 1
    #     y1 = y1 + 63
    #     y2 = y2 + 63
    #
    # hist = []
    # for x in range(len(dif)):
    #     if dif[x] <= -0.1:
    #         hist.append(10)
    #     elif dif[x] > -0.1 and dif[x] <= -0.09:
    #         hist.append(9)
    #     elif dif[x] > -0.09 and dif[x] <= -0.08:
    #         hist.append(8)
    #     elif dif[x] > -0.08 and dif[x] <= -0.07:
    #         hist.append(7)
    #     elif dif[x] > -0.07 and dif[x] <= -0.06:
    #         hist.append(6)
    #     elif dif[x] > -0.06 and dif[x] <= -0.05:
    #         hist.append(5)
    #     elif dif[x] > -0.05 and dif[x] <= -0.04:
    #         hist.append(4)
    #     elif dif[x] > -0.04 and dif[x] <= -0.03:
    #         hist.append(3)
    #     elif dif[x] > -0.03 and dif[x] <= -0.02:
    #         hist.append(2)
    #     elif dif[x] > -0.02 and dif[x] <= -0.01:
    #         hist.append(1)
    #     elif dif[x] > -0.01 and dif[x] <= 0.00:
    #         hist.append(0)
    #     elif dif[x] > 0.00 and dif[x] <= 0.01:
    #         hist.append(-1)
    #     elif dif[x] > 0.01 and dif[x] <= 0.02:
    #         hist.append(-2)
    #     else:
    #         hist.append(np.nan)
    #
    # res = {"rotorangle": rota,
    #        "tavg": tavg,
    #        "tmin": tmin,
    #        "tmax": tmax,
    #        "earheight": ear,
    #        "meshing": mesh,
    #        "dif": dif,
    #        "hist": hist}
    # res = pd.DataFrame(res)
    #
    # rip = []
    # y0 = 0
    # y1 = 0
    # y2 = 9
    # while y0 < 26:
    #     for x in range(y1, y2):
    #         a = (res["tavg"])[0 + x]
    #         b = (res["tavg"])[54 + x]
    #         c = b - a
    #         rip.append(c)
    #     for z in range(54):
    #         rip.append(np.NaN)
    #     y0 = y0 + 1
    #     y1 = y1 + 63
    #     y2 = y2 + 63
    #
    # res = {"rotorangle": rota,
    #        "tavg": tavg,
    #        "tmin": tmin,
    #        "tmax": tmax,
    #        "earheight": ear,
    #        "meshing": mesh,
    #        "dif": dif,
    #        "hist": hist,
    #        "rip": rip}
    # res = pd.DataFrame(res)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(res)
    # res.to_pickle(ModelDir.DATA / "df_msh250.pkl")

if switch == 1:

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.050
    range_b1 = 0.025
    nsteps_b = 2

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_250_v2.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        a = a + 1
        b = b + 61

    for i in range(int(ran/nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res2 = {"rotorangle": rota,
           "tavg": tavg,
           "tmin": tmin,
           "tmax": tmax,
           "earheight": ear,
           "meshing": mesh}
    res2 = pd.DataFrame(res2)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res2)

if switch == 3:

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 1.0
    range_b1 = 0.1
    nsteps_b = 7

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_250.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []
    tempmin = [[] for i in range(ran)]
    tempmax = [[] for i in range(ran)]

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        tempmax[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        tempmin[a] = (tmin[a] - tavg[a]) / tavg[a] * -100
        if tempmax[a] >= tempmin[a]:
            wavcent[a] = tempmax[a]
        else:
            wavcent[a] = tempmin[a]
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res1 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res1 = pd.DataFrame(res1)

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.050
    range_b1 = 0.025
    nsteps_b = 2

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_250_v2.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []
    tempmin = [[] for i in range(ran)]
    tempmax = [[] for i in range(ran)]

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        tempmax[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        tempmin[a] = (tmin[a] - tavg[a]) / tavg[a] * -100
        if tempmax[a] >= tempmin[a]:
            wavcent[a] = tempmax[a]
        else:
            wavcent[a] = tempmin[a]
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res2 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res2 = pd.DataFrame(res2)

    def Insert_row(row_number, df, row_value):
        # Starting value of upper half
        start_upper = 0

        # End value of upper half
        end_upper = row_number

        # Start value of lower half
        start_lower = row_number

        # End value of lower half
        end_lower = df.shape[0]

        # Create a list of upper_half index
        upper_half = [*range(start_upper, end_upper, 1)]

        # Create a list of lower_half index
        lower_half = [*range(start_lower, end_lower, 1)]

        # Increment the value of lower half by 1
        lower_half = [x.__add__(1) for x in lower_half]

        # Combine the two lists
        index_ = upper_half + lower_half

        # Update the index of the dataframe
        df.index = index_

        # Insert a row at the end
        df.loc[row_number] = row_value

        # Sort the index labels
        df = df.sort_index()

        # return the dataframe
        return df

    res = res1.copy()

    a = 63
    b = 81
    c = 0
    d = 18
    x = 0
    while x < 26:
        for i, j in zip(range(a, b), range(c, d)):
            row_number = i
            row_value = list(res2.iloc[j])
            res = Insert_row(row_number, res, row_value)
        a = a + 81
        b = b + 81
        c = c + 18
        d = d + 18
        x = x + 1

    ###################################################################################################################

    A = []
    i = 0
    a = 0.5
    while i < 26:
        for j in range(9):
            A.append(round(a, 2))
        i = i + 1
        a = a + 0.1

    ML = [1.000, 0.850, 0.700, 0.550, 0.400, 0.250, 0.100, 0.050, 0.025]
    M = []
    for i in range(26):
        for j in range(9):
            M.append(ML[j])

    rot = [[] for i in range(26 * 9)]
    avg = [[] for i in range(26 * 9)]
    a = 0
    i = 0
    while i < 26 * 9:
        for j in range(9):
            rot[i].append((res["rotorangle"])[a + j])
            avg[i].append((res["tavg"])[a + j])
        a = a + 9
        i = i + 1
    print(avg)

    maxt = []
    for i in range(26 * 9):
        maxt.append(np.max(avg[i]))

    doe = {'A': A,
           'M': M,
           'rot': rot,
           'avg': avg,
           'maxt': maxt}
    doe = pd.DataFrame(doe)
    doe.to_pickle(ModelDir.DATA / "df_doe250.pkl")

    ###############Statistical functions###################

    rot = list(np.arange(116, 152, 4))
    maxt = []
    mint = []
    avgt = []
    temp = [[] for i in range(9)]
    temp1 = [[] for i in range(26)]

    for i in range(9):
        for j in range(26 * 9):
            temp[i].append((avg[j])[i])
    for i in range(9):
        maxt.append(np.max(temp[i]))
        mint.append(np.min(temp[i]))
        avgt.append(np.mean(temp[i]))

    env = {"rot": rot,
           "mint": mint,
           "maxt": maxt,
           "avgt": avgt}
    env = pd.DataFrame(env)
    env.to_pickle(ModelDir.DATA / "df_env250.pkl")
    ###################################################################################################################

    rip0 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[9 + x]
            c = b - a
            rip0.append(c)
        for z in range(72):
            rip0.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip1 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[18 + x]
            c = b - a
            rip1.append(c)
        for z in range(72):
            rip1.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip2 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[27 + x]
            c = b - a
            rip2.append(c)
        for z in range(72):
            rip2.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip3 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[36 + x]
            c = b - a
            rip3.append(c)
        for z in range(72):
            rip3.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip4 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[45 + x]
            c = b - a
            rip4.append(c)
        for z in range(72):
            rip4.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip5 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[54 + x]
            c = b - a
            rip5.append(c)
        for z in range(72):
            rip5.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81
    rip6 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[63 + x]
            c = b - a
            rip6.append(c)
        for z in range(72):
            rip6.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip7 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[72 + x]
            c = b - a
            rip7.append(c)
        for z in range(72):
            rip7.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip0 = [x for x in rip0 if str(x) != 'nan']
    rip1 = [x for x in rip1 if str(x) != 'nan']
    rip2 = [x for x in rip2 if str(x) != 'nan']
    rip3 = [x for x in rip3 if str(x) != 'nan']
    rip4 = [x for x in rip4 if str(x) != 'nan']
    rip5 = [x for x in rip5 if str(x) != 'nan']
    rip6 = [x for x in rip6 if str(x) != 'nan']
    rip7 = [x for x in rip7 if str(x) != 'nan']

    rip = {'rip0': rip0,
           'rip1': rip1,
           'rip2': rip2,
           'rip3': rip3,
           'rip4': rip4,
           'rip5': rip5,
           'rip6': rip6,
           'rip7': rip7,
           }

    rip=pd.DataFrame(rip)
    rip.to_pickle(ModelDir.DATA / "df_rip250.pkl")
    res.to_pickle(ModelDir.DATA / "df_msh250.pkl")
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(rip)

    ripsum = []
    for x in rip0:
        ripsum.append(x)
    for x in rip1:
        ripsum.append(x)
    for x in rip2:
        ripsum.append(x)
    for x in rip3:
        ripsum.append(x)
    for x in rip4:
        ripsum.append(x)
    for x in rip5:
        ripsum.append(x)
    for x in rip6:
        ripsum.append(x)
    for x in rip7:
        ripsum.append(x)
    x250 = ripsum

    ###################################################################################################################
    temp_a = [[] for i in range(26 * 9)]
    temp_twav = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b + 0, b + 81, 9):
            temp_tmax = res['tmax'].loc[a + 5]
            temp_index = a + 5
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['per'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b = []
    temp_c = []
    temp_d = []
    temp_e = []
    twavmax2 = []
    twavmin2 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax2.append(round(temp_c[i], 2))
        twavmin2.append(round(temp_d[i], 2))

    temp_a = [[] for i in range(26 * 9)]
    temp_twav = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b + 0, b + 81, 9):
            temp_tmax = res['tmax'].loc[a + 6]
            temp_index = a + 6
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['per'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b = []
    temp_c = []
    temp_d = []
    temp_e = []
    twavmax1 = []
    twavmin1 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax1.append(round(temp_c[i], 2))
        twavmin1.append(round(temp_d[i], 2))
    twavmax = [np.mean(x) for x in zip(twavmax1, twavmax2)]
    twavmin = [np.mean(x) for x in zip(twavmin1, twavmin2)]

    wav = {"A": temp_b,
           "max": twavmax,
           "min": twavmin}
    wav = pd.DataFrame(wav)

    wav.to_pickle(ModelDir.DATA / "df_wav250.pkl")
    ###################################################################################################################
    rot = list(np.arange(116, 152, 4))
    per = [[] for i in range(9)]
    b = 0
    for b in range(9):
        for a in range(0, 26 * 9 * 9, 9):
            per[b].append(res['per'].loc[a + b])

    permin = []
    permax = []
    permean = []

    for b in range(9):
        permin.append(min(per[b]))
        permax.append(max(per[b]))
        permean.append(np.mean(per[b]))

    wav = {"A": rot,
           "max": permax,
           "min": permin,
           "mean": permean,
           "opt": res['per'].loc[1296:1304]}
    wav = pd.DataFrame(wav)
    wav.to_pickle(ModelDir.DATA / "df_wave250.pkl")

if switch == 4:
    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 1.0
    range_b1 = 0.1
    nsteps_b = 7

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_100.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []
    tempmin = [[] for i in range(ran)]
    tempmax = [[] for i in range(ran)]

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        tempmax[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        tempmin[a] = (tmin[a] - tavg[a]) / tavg[a] * -100
        if tempmax[a] >= tempmin[a]:
            wavcent[a] = tempmax[a]
        else:
            wavcent[a] = tempmin[a]
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res1 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res1 = pd.DataFrame(res1)

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.050
    range_b1 = 0.025
    nsteps_b = 2

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_100_v2.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []
    tempmin = [[] for i in range(ran)]
    tempmax = [[] for i in range(ran)]

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        tempmax[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        tempmin[a] = (tmin[a] - tavg[a]) / tavg[a] * -100
        if tempmax[a] >= tempmin[a]:
            wavcent[a] = tempmax[a]
        else:
            wavcent[a] = tempmin[a]
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res2 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res2 = pd.DataFrame(res2)

    def Insert_row(row_number, df, row_value):
        # Starting value of upper half
        start_upper = 0

        # End value of upper half
        end_upper = row_number

        # Start value of lower half
        start_lower = row_number

        # End value of lower half
        end_lower = df.shape[0]

        # Create a list of upper_half index
        upper_half = [*range(start_upper, end_upper, 1)]

        # Create a list of lower_half index
        lower_half = [*range(start_lower, end_lower, 1)]

        # Increment the value of lower half by 1
        lower_half = [x.__add__(1) for x in lower_half]

        # Combine the two lists
        index_ = upper_half + lower_half

        # Update the index of the dataframe
        df.index = index_

        # Insert a row at the end
        df.loc[row_number] = row_value

        # Sort the index labels
        df = df.sort_index()

        # return the dataframe
        return df

    res = res1.copy()

    a = 63
    b = 81
    c = 0
    d = 18
    x = 0
    while x < 26:
        for i, j in zip(range(a, b), range(c, d)):
            row_number = i
            row_value = list(res2.iloc[j])
            res = Insert_row(row_number, res, row_value)
        a = a + 81
        b = b + 81
        c = c + 18
        d = d + 18
        x = x + 1

    ###################################################################################################################

    A = []
    i = 0
    a = 0.5
    while i < 26:
        for j in range(9):
            A.append(round(a, 2))
        i = i + 1
        a = a + 0.1

    ML = [1.000, 0.850, 0.700, 0.550, 0.400, 0.250, 0.100, 0.050, 0.025]
    M = []
    for i in range(26):
        for j in range(9):
            M.append(ML[j])

    rot = [[] for i in range(26 * 9)]
    avg = [[] for i in range(26 * 9)]
    a = 0
    i = 0
    while i < 26 * 9:
        for j in range(9):
            rot[i].append((res["rotorangle"])[a + j])
            avg[i].append((res["tavg"])[a + j])
        a = a + 9
        i = i + 1

    maxt = []
    for i in range(26 * 9):
        maxt.append(np.max(avg[i]))

    doe = {'A': A,
           'M': M,
           'rot': rot,
           'avg': avg,
           'maxt': maxt}
    doe = pd.DataFrame(doe)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res)

    doe.to_pickle(ModelDir.DATA / "df_doe100.pkl")

    ###############Statistical functions###################

    rot = list(np.arange(116, 152, 4))
    maxt = []
    mint = []
    avgt = []
    temp = [[] for i in range(9)]
    temp1 = [[] for i in range(26)]

    for i in range(9):
        for j in range(26 * 9):
            temp[i].append((avg[j])[i])
    for i in range(9):
        maxt.append(np.max(temp[i]))
        mint.append(np.min(temp[i]))
        avgt.append(np.mean(temp[i]))

    env = {"rot": rot,
           "mint": mint,
           "maxt": maxt,
           "avgt": avgt}
    env = pd.DataFrame(env)
    env.to_pickle(ModelDir.DATA / "df_env100.pkl")
    ###################################################################################################################

    rip0 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[9 + x]
            c = b - a
            rip0.append(c)
        for z in range(72):
            rip0.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip1 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[18 + x]
            c = b - a
            rip1.append(c)
        for z in range(72):
            rip1.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip2 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[27 + x]
            c = b - a
            rip2.append(c)
        for z in range(72):
            rip2.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip3 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[36 + x]
            c = b - a
            rip3.append(c)
        for z in range(72):
            rip3.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip4 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[45 + x]
            c = b - a
            rip4.append(c)
        for z in range(72):
            rip4.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip5 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[54 + x]
            c = b - a
            rip5.append(c)
        for z in range(72):
            rip5.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81
    rip6 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[63 + x]
            c = b - a
            rip6.append(c)
        for z in range(72):
            rip6.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip7 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[72 + x]
            c = b - a
            rip7.append(c)
        for z in range(72):
            rip7.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip0 = [x for x in rip0 if str(x) != 'nan']
    rip1 = [x for x in rip1 if str(x) != 'nan']
    rip2 = [x for x in rip2 if str(x) != 'nan']
    rip3 = [x for x in rip3 if str(x) != 'nan']
    rip4 = [x for x in rip4 if str(x) != 'nan']
    rip5 = [x for x in rip5 if str(x) != 'nan']
    rip6 = [x for x in rip6 if str(x) != 'nan']
    rip7 = [x for x in rip7 if str(x) != 'nan']

    rip = {'rip0': rip0,
           'rip1': rip1,
           'rip2': rip2,
           'rip3': rip3,
           'rip4': rip4,
           'rip5': rip5,
           'rip6': rip6,
           'rip7': rip7,
           }

    rip = pd.DataFrame(rip)
    rip.to_pickle(ModelDir.DATA / "df_rip100.pkl")
    res.to_pickle(ModelDir.DATA / "df_msh100.pkl")

    ripsum = []
    for x in rip0:
        ripsum.append(x)
    for x in rip1:
        ripsum.append(x)
    for x in rip2:
        ripsum.append(x)
    for x in rip3:
        ripsum.append(x)
    for x in rip4:
        ripsum.append(x)
    for x in rip5:
        ripsum.append(x)
    for x in rip6:
        ripsum.append(x)
    for x in rip7:
        ripsum.append(x)
    x100 = ripsum

    ###################################################################################################################
    temp_a = [[] for i in range(26 * 9)]
    temp_twav = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b + 0, b + 81, 9):
            temp_tmax = res['tmax'].loc[a + 3]
            temp_index = a + 3
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['per'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b = []
    temp_c = []
    temp_d = []
    temp_e = []
    twavmax2 = []
    twavmin2 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax2.append(round(temp_c[i], 2))
        twavmin2.append(round(temp_d[i], 2))

    temp_a = [[] for i in range(26 * 9)]
    temp_twav = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b + 0, b + 81, 9):
            temp_tmax = res['tmax'].loc[a + 4]
            temp_index = a + 4
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['per'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b = []
    temp_c = []
    temp_d = []
    temp_e = []
    twavmax1 = []
    twavmin1 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax1.append(round(temp_c[i], 2))
        twavmin1.append(round(temp_d[i], 2))
    twavmax = [np.mean(x) for x in zip(twavmax1, twavmax2)]
    twavmin = [np.mean(x) for x in zip(twavmin1, twavmin2)]

    wav = {"A": temp_b,
           "max": twavmax,
           "min": twavmin}
    wav = pd.DataFrame(wav)

    wav.to_pickle(ModelDir.DATA / "df_wav100.pkl")
    ###################################################################################################################
    rot = list(np.arange(116, 152, 4))
    per = [[] for i in range(9)]
    b = 0
    for b in range(9):
        for a in range(0, 26*9*9, 9):
            per[b].append(res['per'].loc[a + b])

    permin  = []
    permax  = []
    permean = []

    for b in range(9):
        permin.append(min(per[b]))
        permax.append(max(per[b]))
        permean.append(np.mean(per[b]))

    wav = {"A": rot,
           "max": permax,
           "min": permin,
           "mean": permean,
           "opt": res['per'].loc[1296:1304]}
    wav = pd.DataFrame(wav)
    wav.to_pickle(ModelDir.DATA / "df_wave100.pkl")

if switch == 5:
    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 1.0
    range_b1 = 0.1
    nsteps_b = 7

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_150.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        wavcent[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res1 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res1 = pd.DataFrame(res1)

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.050
    range_b1 = 0.025
    nsteps_b = 2

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_150_v2.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        wavcent[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res2 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res2 = pd.DataFrame(res2)

    def Insert_row(row_number, df, row_value):
        # Starting value of upper half
        start_upper = 0

        # End value of upper half
        end_upper = row_number

        # Start value of lower half
        start_lower = row_number

        # End value of lower half
        end_lower = df.shape[0]

        # Create a list of upper_half index
        upper_half = [*range(start_upper, end_upper, 1)]

        # Create a list of lower_half index
        lower_half = [*range(start_lower, end_lower, 1)]

        # Increment the value of lower half by 1
        lower_half = [x.__add__(1) for x in lower_half]

        # Combine the two lists
        index_ = upper_half + lower_half

        # Update the index of the dataframe
        df.index = index_

        # Insert a row at the end
        df.loc[row_number] = row_value

        # Sort the index labels
        df = df.sort_index()

        # return the dataframe
        return df

    res = res1.copy()

    a = 63
    b = 81
    c = 0
    d = 18
    x = 0
    while x < 26:
        for i, j in zip(range(a, b), range(c, d)):
            row_number = i
            row_value = list(res2.iloc[j])
            res = Insert_row(row_number, res, row_value)
        a = a + 81
        b = b + 81
        c = c + 18
        d = d + 18
        x = x + 1

        ###################################################################################################################

    A = []
    i = 0
    a = 0.5
    while i < 26:
        for j in range(9):
            A.append(round(a, 2))
        i = i + 1
        a = a + 0.1

    ML = [1.000, 0.850, 0.700, 0.550, 0.400, 0.250, 0.100, 0.050, 0.025]
    M = []
    for i in range(26):
        for j in range(9):
            M.append(ML[j])

    rot = [[] for i in range(26 * 9)]
    avg = [[] for i in range(26 * 9)]
    a = 0
    i = 0
    while i < 26 * 9:
        for j in range(9):
            rot[i].append((res["rotorangle"])[a + j])
            avg[i].append((res["tavg"])[a + j])
        a = a + 9
        i = i + 1

    maxt = []
    for i in range(26 * 9):
        maxt.append(np.max(avg[i]))

    doe = {'A': A,
           'M': M,
           'rot': rot,
           'avg': avg,
           'maxt': maxt}
    doe = pd.DataFrame(doe)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
         print(res)

    doe.to_pickle(ModelDir.DATA / "df_doe150.pkl")

    ###############Statistical functions###################

    rot = list(np.arange(116, 152, 4))
    maxt = []
    mint = []
    avgt = []
    temp = [[] for i in range(9)]
    temp1 = [[] for i in range(26)]

    for i in range(9):
        for j in range(26 * 9):
            temp[i].append((avg[j])[i])
    for i in range(9):
        maxt.append(np.max(temp[i]))
        mint.append(np.min(temp[i]))
        avgt.append(np.mean(temp[i]))

    env = {"rot": rot,
           "mint": mint,
           "maxt": maxt,
           "avgt": avgt}
    env = pd.DataFrame(env)
    env.to_pickle(ModelDir.DATA / "df_env150.pkl")
    ###################################################################################################################

    rip0 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[9 + x]
            c = b - a
            rip0.append(c)
        for z in range(72):
            rip0.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip1 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[18 + x]
            c = b - a
            rip1.append(c)
        for z in range(72):
            rip1.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip2 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[27 + x]
            c = b - a
            rip2.append(c)
        for z in range(72):
            rip2.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip3 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[36 + x]
            c = b - a
            rip3.append(c)
        for z in range(72):
            rip3.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip4 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[45 + x]
            c = b - a
            rip4.append(c)
        for z in range(72):
            rip4.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip5 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[54 + x]
            c = b - a
            rip5.append(c)
        for z in range(72):
            rip5.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81
    rip6 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[63 + x]
            c = b - a
            rip6.append(c)
        for z in range(72):
            rip6.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip7 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[72 + x]
            c = b - a
            rip7.append(c)
        for z in range(72):
            rip7.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip0 = [x for x in rip0 if str(x) != 'nan']
    rip1 = [x for x in rip1 if str(x) != 'nan']
    rip2 = [x for x in rip2 if str(x) != 'nan']
    rip3 = [x for x in rip3 if str(x) != 'nan']
    rip4 = [x for x in rip4 if str(x) != 'nan']
    rip5 = [x for x in rip5 if str(x) != 'nan']
    rip6 = [x for x in rip6 if str(x) != 'nan']
    rip7 = [x for x in rip7 if str(x) != 'nan']

    rip = {'rip0': rip0,
           'rip1': rip1,
           'rip2': rip2,
           'rip3': rip3,
           'rip4': rip4,
           'rip5': rip5,
           'rip6': rip6,
           'rip7': rip7,
           }

    rip = pd.DataFrame(rip)
    rip.to_pickle(ModelDir.DATA / "df_rip150.pkl")
    res.to_pickle(ModelDir.DATA / "df_msh150.pkl")

    ripsum = []
    for x in rip0:
        ripsum.append(x)
    for x in rip1:
        ripsum.append(x)
    for x in rip2:
        ripsum.append(x)
    for x in rip3:
        ripsum.append(x)
    for x in rip4:
        ripsum.append(x)
    for x in rip5:
        ripsum.append(x)
    for x in rip6:
        ripsum.append(x)
    for x in rip7:
        ripsum.append(x)
    x150 = ripsum

    ###################################################################################################################
    temp_a      = [[] for i in range(26 * 9)]
    temp_twav   = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b+0, b+81, 9):
            temp_tmax   = res['tmax'].loc[a+5]
            temp_index  = a+5
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['twav'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b  = []
    temp_c  = []
    temp_d  = []
    temp_e  = []
    twavmax2 = []
    twavmin2 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax2.append(round(temp_c[i] / temp_e[i] * 100 / 2, 2))
        twavmin2.append(round(temp_d[i] / temp_e[i] * 100 / 2, 2))

    temp_a = [[] for i in range(26 * 9)]
    temp_twav = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b + 0, b + 81, 9):
            temp_tmax = res['tmax'].loc[a + 4]
            temp_index = a + 4
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['twav'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b = []
    temp_c = []
    temp_d = []
    temp_e = []
    twavmax1 = []
    twavmin1 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax1.append(round(temp_c[i] / temp_e[i] * 100 / 2, 2))
        twavmin1.append(round(temp_d[i] / temp_e[i] * 100 / 2, 2))
    twavmax = [np.mean(x) for x in zip(twavmax1, twavmax2)]
    twavmin = [np.mean(x) for x in zip(twavmin1, twavmin2)]

    wav = {"A": temp_b,
           "max": twavmax,
           "min": twavmin}
    wav = pd.DataFrame(wav)

    wav.to_pickle(ModelDir.DATA / "df_wav150.pkl")
    ###################################################################################################################
    rot = list(np.arange(116, 152, 4))
    per = [[] for i in range(9)]
    b = 0
    for b in range(9):
        for a in range(0, 26 * 9 * 9, 9):
            per[b].append(res['per'].loc[a + b])

    permin = []
    permax = []
    permean = []

    for b in range(9):
        permin.append(min(per[b]))
        permax.append(max(per[b]))
        permean.append(np.mean(per[b]))

    wav = {"A": rot,
           "max": permax,
           "min": permin,
           "mean": permean,
           "opt": res['per'].loc[1296:1304]}
    wav = pd.DataFrame(wav)
    wav.to_pickle(ModelDir.DATA / "df_wave150.pkl")

if switch == 6:
    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 1.0
    range_b1 = 0.1
    nsteps_b = 7

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_200.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        wavcent[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res1 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res1 = pd.DataFrame(res1)

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.050
    range_b1 = 0.025
    nsteps_b = 2

    range_c0 = 0
    range_c1 = -60
    nsteps_c = 61

    range_d0 = 29
    range_d1 = 37
    nsteps_d = 9

    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)
    range_c = linspace(range_c0, range_c1, nsteps_c)
    range_d = linspace(range_d0, range_d1, nsteps_d)

    f = open(ModelDir.DATA / f'avg_msh_200_v2.json')
    inp = json.load(f)

    msh = []
    for i in range(len(inp["Torque"])):
        msh.append((inp["Torque"])[i])

    ran = int(len(inp["Torque"]) / nsteps_c)
    trip = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tavg = [[] for i in range(ran)]
    twav = [[] for i in range(ran)]
    wavcent = [[] for i in range(ran)]
    rota = []
    ear = []
    mesh = []

    a = 0
    b = 0
    while a < (ran):
        trip[a] = [msh[i] for i in range(b + 0, b + 61)]
        trip[a] = [round(num, 1) for num in trip[a]]
        tmin[a] = min(trip[a])
        tmax[a] = max(trip[a])
        tavg[a] = np.average(trip[a])
        tavg[a] = round(tavg[a], 3)
        twav[a] = tmax[a] - tmin[a]
        wavcent[a] = (tmax[a] - tavg[a]) / tavg[a] * 100
        a = a + 1
        b = b + 61

    for i in range(int(ran / nsteps_d)):
        for j in range(nsteps_d):
            rota.append(range_d[j] * 4)

    for i in range(nsteps_a):
        for j in range(int(ran / nsteps_a)):
            ear.append(round(range_a[i], 1))

    for i in range(nsteps_a):
        for j in range(nsteps_b):
            for k in range(int(nsteps_d)):
                mesh.append(range_b[j])

    res2 = {"rotorangle": rota,
            "tavg": tavg,
            "tmin": tmin,
            "tmax": tmax,
            "earheight": ear,
            "meshing": mesh,
            "twav": twav,
            "per": wavcent}
    res2 = pd.DataFrame(res2)

    def Insert_row(row_number, df, row_value):
        # Starting value of upper half
        start_upper = 0

        # End value of upper half
        end_upper = row_number

        # Start value of lower half
        start_lower = row_number

        # End value of lower half
        end_lower = df.shape[0]

        # Create a list of upper_half index
        upper_half = [*range(start_upper, end_upper, 1)]

        # Create a list of lower_half index
        lower_half = [*range(start_lower, end_lower, 1)]

        # Increment the value of lower half by 1
        lower_half = [x.__add__(1) for x in lower_half]

        # Combine the two lists
        index_ = upper_half + lower_half

        # Update the index of the dataframe
        df.index = index_

        # Insert a row at the end
        df.loc[row_number] = row_value

        # Sort the index labels
        df = df.sort_index()

        # return the dataframe
        return df

    res = res1.copy()

    a = 63
    b = 81
    c = 0
    d = 18
    x = 0
    while x < 26:
        for i, j in zip(range(a, b), range(c, d)):
            row_number = i
            row_value = list(res2.iloc[j])
            res = Insert_row(row_number, res, row_value)
        a = a + 81
        b = b + 81
        c = c + 18
        d = d + 18
        x = x + 1

        ###################################################################################################################

    A = []
    i = 0
    a = 0.5
    while i < 26:
        for j in range(9):
            A.append(round(a, 2))
        i = i + 1
        a = a + 0.1

    ML = [1.000, 0.850, 0.700, 0.550, 0.400, 0.250, 0.100, 0.050, 0.025]
    M = []
    for i in range(26):
        for j in range(9):
            M.append(ML[j])

    rot = [[] for i in range(26 * 9)]
    avg = [[] for i in range(26 * 9)]
    a = 0
    i = 0
    while i < 26 * 9:
        for j in range(9):
            rot[i].append((res["rotorangle"])[a + j])
            avg[i].append((res["tavg"])[a + j])
        a = a + 9
        i = i + 1

    maxt = []
    for i in range(26 * 9):
        maxt.append(np.max(avg[i]))

    doe = {'A': A,
           'M': M,
           'rot': rot,
           'avg': avg,
           'maxt': maxt}
    doe = pd.DataFrame(doe)

    doe.to_pickle(ModelDir.DATA / "df_doe200.pkl")

    ###############Statistical functions###################

    rot = list(np.arange(116, 152, 4))
    maxt = []
    mint = []
    avgt = []
    temp = [[] for i in range(9)]
    temp1 = [[] for i in range(26)]

    for i in range(9):
        for j in range(26 * 9):
            temp[i].append((avg[j])[i])
    for i in range(9):
        maxt.append(np.max(temp[i]))
        mint.append(np.min(temp[i]))
        avgt.append(np.mean(temp[i]))

    env = {"rot": rot,
           "mint": mint,
           "maxt": maxt,
           "avgt": avgt}
    env = pd.DataFrame(env)
    env.to_pickle(ModelDir.DATA / "df_env200.pkl")
    ###################################################################################################################

    rip0 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[9 + x]
            c = b - a
            rip0.append(c)
        for z in range(72):
            rip0.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip1 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[18 + x]
            c = b - a
            rip1.append(c)
        for z in range(72):
            rip1.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip2 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[27 + x]
            c = b - a
            rip2.append(c)
        for z in range(72):
            rip2.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip3 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[36 + x]
            c = b - a
            rip3.append(c)
        for z in range(72):
            rip3.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip4 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[45 + x]
            c = b - a
            rip4.append(c)
        for z in range(72):
            rip4.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip5 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[54 + x]
            c = b - a
            rip5.append(c)
        for z in range(72):
            rip5.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81
    rip6 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[63 + x]
            c = b - a
            rip6.append(c)
        for z in range(72):
            rip6.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip7 = []
    y0 = 0
    y1 = 0
    y2 = 9
    while y0 < 26:
        for x in range(y1, y2):
            a = (res["tavg"])[0 + x]
            b = (res["tavg"])[72 + x]
            c = b - a
            rip7.append(c)
        for z in range(72):
            rip7.append(np.NaN)
        y0 = y0 + 1
        y1 = y1 + 81
        y2 = y2 + 81

    rip0 = [x for x in rip0 if str(x) != 'nan']
    rip1 = [x for x in rip1 if str(x) != 'nan']
    rip2 = [x for x in rip2 if str(x) != 'nan']
    rip3 = [x for x in rip3 if str(x) != 'nan']
    rip4 = [x for x in rip4 if str(x) != 'nan']
    rip5 = [x for x in rip5 if str(x) != 'nan']
    rip6 = [x for x in rip6 if str(x) != 'nan']
    rip7 = [x for x in rip7 if str(x) != 'nan']

    rip = {'rip0': rip0,
           'rip1': rip1,
           'rip2': rip2,
           'rip3': rip3,
           'rip4': rip4,
           'rip5': rip5,
           'rip6': rip6,
           'rip7': rip7,
           }

    rip = pd.DataFrame(rip)
    rip.to_pickle(ModelDir.DATA / "df_rip200.pkl")
    res.to_pickle(ModelDir.DATA / "df_msh200.pkl")

    ripsum = []
    for x in rip0:
        ripsum.append(x)
    for x in rip1:
        ripsum.append(x)
    for x in rip2:
        ripsum.append(x)
    for x in rip3:
        ripsum.append(x)
    for x in rip4:
        ripsum.append(x)
    for x in rip5:
        ripsum.append(x)
    for x in rip6:
        ripsum.append(x)
    for x in rip7:
        ripsum.append(x)
    x200 = ripsum

    ###################################################################################################################
    temp_a = [[] for i in range(26 * 9)]
    temp_twav = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b + 0, b + 81, 9):
            temp_tmax = res['tmax'].loc[a + 5]
            temp_index = a + 5
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['twav'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b = []
    temp_c = []
    temp_d = []
    temp_e = []
    twavmax2 = []
    twavmin2 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax2.append(round(temp_c[i] / temp_e[i] * 100 / 2, 2))
        twavmin2.append(round(temp_d[i] / temp_e[i] * 100 / 2, 2))

    temp_a = [[] for i in range(26 * 9)]
    temp_twav = [[] for i in range(26 * 9)]
    temp_maxi = [[] for i in range(26 * 9)]
    b = 0
    for i in range(26):
        for a in range(b + 0, b + 81, 9):
            temp_tmax = res['tmax'].loc[a + 6]
            temp_index = a + 6
            temp_a[i].append(res['earheight'].loc[temp_index])
            temp_twav[i].append(res['twav'].loc[temp_index])
            temp_maxi[i].append(res['tavg'].loc[temp_index])
        b = b + 81

    temp_b = []
    temp_c = []
    temp_d = []
    temp_e = []
    twavmax1 = []
    twavmin1 = []
    for i in range(26):
        temp_b.append(max((temp_a[i])))
        temp_c.append(round(max((temp_twav[i])), 3))
        temp_d.append(round(min((temp_twav[i])), 3))
        temp_e.append(round(max((temp_maxi[i])), 3))
    for i in range(len(temp_c)):
        twavmax1.append(round(temp_c[i] / temp_e[i] * 100 / 2, 2))
        twavmin1.append(round(temp_d[i] / temp_e[i] * 100 / 2, 2))
    twavmax = np.multiply([np.mean(x) for x in zip(twavmax1, twavmax2)], 0.99)
    twavmin = np.multiply([np.mean(x) for x in zip(twavmin1, twavmin2)], 0.99)

    wav = {"A": temp_b,
           "max": twavmax,
           "min": twavmin}
    wav = pd.DataFrame(wav)

    wav.to_pickle(ModelDir.DATA / "df_wav200.pkl")
    ###################################################################################################################
    rot = list(np.arange(116, 152, 4))
    per = [[] for i in range(9)]
    b = 0
    for b in range(9):
        for a in range(0, 26 * 9 * 9, 9):
            per[b].append(res['per'].loc[a + b])

    permin = []
    permax = []
    permean = []

    for b in range(9):
        permin.append(min(per[b]))
        permax.append(max(per[b]))
        permean.append(np.mean(per[b]))

    wav = {"A": rot,
           "max": permax,
           "min": permin,
           "mean": permean,
           "opt": res['per'].loc[1296:1304]}
    wav = pd.DataFrame(wav)
    wav.to_pickle(ModelDir.DATA / "df_wave200.pkl")
