import numpy as np
import pandas as pd
from numpy import linspace
import json
import matplotlib.pyplot as plt


from digital_twin_distiller import ModelDir

ModelDir.set_base(__file__)

switch = 1
if switch == 0:

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.0
    range_b1 = -60.0
    nsteps_b = 61

    iterlist = [[], [], []]
    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)

    for a in range(nsteps_a):
        for i in range(46):
            range_c0 = 0 + i
            range_c1 = 15 + i
            nsteps_c = nsteps_b

            range_c = linspace(range_c0, range_c1, nsteps_c)

            for j in range(nsteps_c):
                iterlist[0].append(round(range_c[j], 3))
                iterlist[1].append(round(range_b[j], 3))
                iterlist[2].append(round(range_a[a], 3))

    f = open(ModelDir.DATA / f'avgear_100.json')
    avg = json.load(f)

    ran = 46 * nsteps_a
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tdif = [[] for i in range(ran)]
    t = [[] for i in range(ran)]
    r = [[] for i in range(ran)]
    l = []
    e = []
    a = 0
    b = 0
    while a < (ran):
        t[a] = [(avg["Torque"])[i] for i in range(b + 0, b + 61)]
        t[a] = [round(num, 1) for num in t[a]]
        tmin[a] = min(t[a])
        tmax[a] = max(t[a])
        tdif[a] = tmax[a] - tmin[a]
        a = a + 1
        b = b + 61

    for i in range(ran):
        for j in range(61):
            r[i].append(j)
    for i in range(ran):
            l.append(i%46)

    for j in range(26):
        for i in range(46):
            e.append(0.5+j/10)

    tavg = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    #twav = [[] for i in range(ran)]
    rota = [[] for i in range(46)]
    for i in range(ran):
        tavg[i] = np.mean(t[i])
    #print(len(tavg))

        tmax[i] = max(t[i])
        tmin[i] = min(t[i])
        #twav[i] = (tmax[i] - tavg[i]) / tavg[i]
    for i in range(46):
        rota[i] = 4*i
    for i in range(25):
        for j in range(46):
            rota.append(rota[j])
    #print(len(rota))
    for i in range(ran):
        rota[i] = round(rota[i], 3)
        tavg[i] = round(tavg[i], 3)
        tmax[i] = round(tmax[i], 3)
        tmin[i] = round(tmin[i], 3)
        #twav[i] = round(twav[i], 3)

    x = [[] for i in range(nsteps_a)]
    y = [[] for i in range(nsteps_a)]
    z = [[] for i in range(nsteps_a)]
    w = [[] for i in range(nsteps_a)]
    for i in range(26):
        for j in range(46):
            x[i].append(rota[j])
    a=0
    b=0
    while a < 26:
        for c in range(46):
            y[a].append(tavg[c + b])
            z[a].append(tmin[c + b])
            w[a].append(tmax[c + b])
        a=a+1
        b=b+46
    res = {"rotorangle": x,
           "tavg": y,
           "tmin": z,
           "tmax": w}


    ripple = {'earheight': e,
              "loadangle": l,
              "rotorangle": r,
              "torque": t,
              "dif": tdif,
              "maxavg": tavg}

    ripple = pd.DataFrame(ripple)
    res = pd.DataFrame(res)
    res.to_pickle(ModelDir.DATA / "df_avgear100.pkl")

    tmax = [[] for i in range(26)]
    lmax = [[] for i in range(26)]
    for i in range(26):
        tmax[i] = max(y[i])
        lmax[i] = y[i].index(tmax[i])

    ear = list(linspace(0.5, 3.0, 26))
    ear = [round(num, 1) for num in ear]

    dff = ripple.loc[(ripple['earheight'] == ear[0]) & (ripple['loadangle'] == lmax[0])]
    for i in range(26):
        df = ripple.loc[(ripple['earheight'] == ear[i]) & (ripple['loadangle'] == lmax[0])]
        dff = pd.concat([dff, df], ignore_index=True)
    ripple = dff.iloc[1:, :]
    ripple.to_pickle(ModelDir.DATA / "df_ripple100.pkl")

    list100 = [[] for x in range(9)]
    for b, c in zip(range(29, 38), range(9)):
        for a in range(0, 26):
            list100[c].append(((res['tavg'])[a])[b])
    for i in range(9):
        list100[i] = np.mean(list100[i])
    print(list100)

if switch == 1:

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.0
    range_b1 = -60.0
    nsteps_b = 61

    iterlist = [[], [], []]
    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)

    for a in range(nsteps_a):
        for i in range(46):
            range_c0 = 0 + i
            range_c1 = 15 + i
            nsteps_c = nsteps_b

            range_c = linspace(range_c0, range_c1, nsteps_c)

            for j in range(nsteps_c):
                iterlist[0].append(round(range_c[j], 3))
                iterlist[1].append(round(range_b[j], 3))
                iterlist[2].append(round(range_a[a], 3))

    f = open(ModelDir.DATA / f'avgear_250.json')
    avg = json.load(f)

    ran = 46 * nsteps_a
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tdif = [[] for i in range(ran)]
    t = [[] for i in range(ran)]
    r = [[] for i in range(ran)]
    l = []
    e = []
    a = 0
    b = 0
    while a < (ran):
        t[a] = [(avg["Torque"])[i] for i in range(b + 0, b + 61)]
        t[a] = [round(num, 1) for num in t[a]]
        tmin[a] = min(t[a])
        tmax[a] = max(t[a])
        tdif[a] = tmax[a] - tmin[a]
        a = a + 1
        b = b + 61

    for i in range(ran):
        for j in range(61):
            r[i].append(j)
    for i in range(ran):
            l.append(i%46)

    for j in range(26):
        for i in range(46):
            e.append(0.5+j/10)

    tavg = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    #twav = [[] for i in range(ran)]
    rota = [[] for i in range(46)]
    for i in range(ran):
        tavg[i] = np.mean(t[i])
    #print(len(tavg))

        tmax[i] = max(t[i])
        tmin[i] = min(t[i])
        #twav[i] = (tmax[i] - tavg[i]) / tavg[i]
    for i in range(46):
        rota[i] = 4*i
    for i in range(25):
        for j in range(46):
            rota.append(rota[j])
    #print(len(rota))
    for i in range(ran):
        rota[i] = round(rota[i], 3)
        tavg[i] = round(tavg[i], 3)
        tmax[i] = round(tmax[i], 3)
        tmin[i] = round(tmin[i], 3)
        #twav[i] = round(twav[i], 3)

    x = [[] for i in range(nsteps_a)]
    y = [[] for i in range(nsteps_a)]
    z = [[] for i in range(nsteps_a)]
    w = [[] for i in range(nsteps_a)]
    for i in range(26):
        for j in range(46):
            x[i].append(rota[j])
    a = 0
    b = 0
    while a < 26:
        for c in range(46):
            y[a].append(tavg[c + b])
            z[a].append(tmin[c + b])
            w[a].append(tmax[c + b])
        a = a + 1
        b = b + 46
    res = {"rotorangle": x,
           "tavg": y,
           "tmin": z,
           "tmax": w}

    ripple = {'earheight': e,
              "loadangle": l,
              "rotorangle": r,
              "torque": t,
              "dif": tdif,
              "maxavg": tavg}

    ripple = pd.DataFrame(ripple)
    res = pd.DataFrame(res)
    res.to_pickle(ModelDir.DATA / "df_avgear250.pkl")

    tmax = [[] for i in range(26)]
    lmax = [[] for i in range(26)]
    for i in range(26):
        tmax[i] = max(y[i])
        lmax[i] = y[i].index(tmax[i])

    ear = list(linspace(0.5, 3.0, 26))
    ear = [round(num, 1) for num in ear]

    dff = ripple.loc[(ripple['earheight'] == ear[0]) & (ripple['loadangle'] == lmax[0])]
    for i in range(26):
        df = ripple.loc[(ripple['earheight'] == ear[i]) & (ripple['loadangle'] == lmax[0])]
        dff = pd.concat([dff, df], ignore_index=True)
    ripple = dff.iloc[1:, :]
    ripple.to_pickle(ModelDir.DATA / "df_ripple250.pkl")

    list250 = [[] for x in range(9)]
    for b, c in zip(range(29, 38), range(9)):
        for a in range(0, 26):
            list250[c].append(((res['tavg'])[a])[b])
    for i in range(9):
        list250[i] = np.mean(list250[i])
    print(list250)

if switch == 2:

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.0
    range_b1 = -60.0
    nsteps_b = 61

    iterlist = [[], [], []]
    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)

    for a in range(nsteps_a):
        for i in range(46):
            range_c0 = 0 + i
            range_c1 = 15 + i
            nsteps_c = nsteps_b

            range_c = linspace(range_c0, range_c1, nsteps_c)

            for j in range(nsteps_c):
                iterlist[0].append(round(range_c[j], 3))
                iterlist[1].append(round(range_b[j], 3))
                iterlist[2].append(round(range_a[a], 3))

    f = open(ModelDir.DATA / f'avgear_150.json')
    avg = json.load(f)

    ran = 46 * nsteps_a
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tdif = [[] for i in range(ran)]
    t = [[] for i in range(ran)]
    r = [[] for i in range(ran)]
    l = []
    e = []
    a = 0
    b = 0
    while a < (ran):
        t[a] = [(avg["Torque"])[i] for i in range(b + 0, b + 61)]
        t[a] = [round(num, 1) for num in t[a]]
        tmin[a] = min(t[a])
        tmax[a] = max(t[a])
        tdif[a] = tmax[a] - tmin[a]
        a = a + 1
        b = b + 61

    for i in range(ran):
        for j in range(61):
            r[i].append(j)
    for i in range(ran):
            l.append(i%46)

    for j in range(26):
        for i in range(46):
            e.append(0.5+j/10)

    tavg = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    #twav = [[] for i in range(ran)]
    rota = [[] for i in range(46)]
    for i in range(ran):
        tavg[i] = np.mean(t[i])
    #print(len(tavg))

        tmax[i] = max(t[i])
        tmin[i] = min(t[i])
        #twav[i] = (tmax[i] - tavg[i]) / tavg[i]
    for i in range(46):
        rota[i] = 4*i
    for i in range(25):
        for j in range(46):
            rota.append(rota[j])
    #print(len(rota))
    for i in range(ran):
        rota[i] = round(rota[i], 3)
        tavg[i] = round(tavg[i], 3)
        tmax[i] = round(tmax[i], 3)
        tmin[i] = round(tmin[i], 3)
        #twav[i] = round(twav[i], 3)

    x = [[] for i in range(nsteps_a)]
    y = [[] for i in range(nsteps_a)]
    z = [[] for i in range(nsteps_a)]
    w = [[] for i in range(nsteps_a)]
    for i in range(26):
        for j in range(46):
            x[i].append(rota[j])
    a = 0
    b = 0
    while a < 26:
        for c in range(46):
            y[a].append(tavg[c + b])
            z[a].append(tmin[c + b])
            w[a].append(tmax[c + b])
        a = a + 1
        b = b + 46
    res = {"rotorangle": x,
           "tavg": y,
           "tmin": z,
           "tmax": w}

    ripple = {'earheight': e,
              "loadangle": l,
              "rotorangle": r,
              "torque": t,
              "dif": tdif,
              "maxavg": tavg}

    ripple = pd.DataFrame(ripple)
    res = pd.DataFrame(res)
    res.to_pickle(ModelDir.DATA / "df_avgear150.pkl")

    tmax = [[] for i in range(26)]
    lmax = [[] for i in range(26)]
    for i in range(26):
        tmax[i] = max(y[i])
        lmax[i] = y[i].index(tmax[i])

    ear = list(linspace(0.5, 3.0, 26))
    ear = [round(num, 1) for num in ear]

    dff = ripple.loc[(ripple['earheight'] == ear[0]) & (ripple['loadangle'] == lmax[0])]
    for i in range(26):
        df = ripple.loc[(ripple['earheight'] == ear[i]) & (ripple['loadangle'] == lmax[0])]
        dff = pd.concat([dff, df], ignore_index=True)
    ripple = dff.iloc[1:, :]
    ripple.to_pickle(ModelDir.DATA / "df_ripple150.pkl")

    list150 = [[] for x in range(9)]
    for b, c in zip(range(29, 38), range(9)):
        for a in range(0, 26):
            list150[c].append(((res['tavg'])[a])[b])
    for i in range(9):
        list150[i] = np.mean(list150[i])
    print(list150)

if switch == 3:

    range_a0 = 0.5
    range_a1 = 3.0
    nsteps_a = 26

    range_b0 = 0.0
    range_b1 = -60.0
    nsteps_b = 61

    iterlist = [[], [], []]
    range_a = linspace(range_a0, range_a1, nsteps_a)
    range_b = linspace(range_b0, range_b1, nsteps_b)

    for a in range(nsteps_a):
        for i in range(46):
            range_c0 = 0 + i
            range_c1 = 15 + i
            nsteps_c = nsteps_b

            range_c = linspace(range_c0, range_c1, nsteps_c)

            for j in range(nsteps_c):
                iterlist[0].append(round(range_c[j], 3))
                iterlist[1].append(round(range_b[j], 3))
                iterlist[2].append(round(range_a[a], 3))

    f = open(ModelDir.DATA / f'avgear_200.json')
    avg = json.load(f)

    ran = 46 * nsteps_a
    tmin = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tdif = [[] for i in range(ran)]
    t = [[] for i in range(ran)]
    r = [[] for i in range(ran)]
    l = []
    e = []
    a = 0
    b = 0
    while a < (ran):
        t[a] = [(avg["Torque"])[i] for i in range(b + 0, b + 61)]
        t[a] = [round(num, 1) for num in t[a]]
        tmin[a] = min(t[a])
        tmax[a] = max(t[a])
        tdif[a] = tmax[a] - tmin[a]
        a = a + 1
        b = b + 61

    for i in range(ran):
        for j in range(61):
            r[i].append(j)
    for i in range(ran):
            l.append(i%46)

    for j in range(26):
        for i in range(46):
            e.append(0.5+j/10)

    tavg = [[] for i in range(ran)]
    tmax = [[] for i in range(ran)]
    tmin = [[] for i in range(ran)]
    #twav = [[] for i in range(ran)]
    rota = [[] for i in range(46)]
    for i in range(ran):
        tavg[i] = np.mean(t[i])
    #print(len(tavg))

        tmax[i] = max(t[i])
        tmin[i] = min(t[i])
        #twav[i] = (tmax[i] - tavg[i]) / tavg[i]
    for i in range(46):
        rota[i] = 4*i
    for i in range(25):
        for j in range(46):
            rota.append(rota[j])
    #print(len(rota))
    for i in range(ran):
        rota[i] = round(rota[i], 3)
        tavg[i] = round(tavg[i], 3)
        tmax[i] = round(tmax[i], 3)
        tmin[i] = round(tmin[i], 3)
        #twav[i] = round(twav[i], 3)

    x = [[] for i in range(nsteps_a)]
    y = [[] for i in range(nsteps_a)]
    z = [[] for i in range(nsteps_a)]
    w = [[] for i in range(nsteps_a)]
    for i in range(26):
        for j in range(46):
            x[i].append(rota[j])
    a = 0
    b = 0
    while a < 26:
        for c in range(46):
            y[a].append(tavg[c + b])
            z[a].append(tmin[c + b])
            w[a].append(tmax[c + b])
        a = a + 1
        b = b + 46
    res = {"rotorangle": x,
           "tavg": y,
           "tmin": z,
           "tmax": w}

    ripple = {'earheight': e,
              "loadangle": l,
              "rotorangle": r,
              "torque": t,
              "dif": tdif,
              "maxavg": tavg}

    ripple = pd.DataFrame(ripple)
    res = pd.DataFrame(res)
    res.to_pickle(ModelDir.DATA / "df_avgear200.pkl")

    tmax = [[] for i in range(26)]
    lmax = [[] for i in range(26)]
    for i in range(26):
        tmax[i] = max(y[i])
        lmax[i] = y[i].index(tmax[i])

    ear = list(linspace(0.5, 3.0, 26))
    ear = [round(num, 1) for num in ear]

    dff = ripple.loc[(ripple['earheight'] == ear[0]) & (ripple['loadangle'] == lmax[0])]
    for i in range(26):
        df = ripple.loc[(ripple['earheight'] == ear[i]) & (ripple['loadangle'] == lmax[0])]
        dff = pd.concat([dff, df], ignore_index=True)
    ripple = dff.iloc[1:, :]
    ripple.to_pickle(ModelDir.DATA / "df_ripple200.pkl")

    list200 = [[] for x in range(9)]
    for b, c in zip(range(29, 38), range(9)):
        for a in range(0, 26):
            list200[c].append(((res['tavg'])[a])[b])
    for i in range(9):
        list200[i] = np.mean(list200[i])
    print(list200)