import numpy as np
import pandas as pd
from numpy import linspace
import json

from digital_twin_distiller import ModelDir

ModelDir.set_base(__file__)

switch = 4
if switch == -1:
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_bb.pkl")

    f = open(ModelDir.DATA / f'res_rbst250_bb.json')
    res = json.load(f)
    res = [np.round(i, 3) for i in res['Torque']]
    iterlist['torque'] = res

    base = pd.DataFrame(columns=['rot', 'rip', 'min', 'max', 'avg', 'amp'])

    rot = list(range(61))
    base['rot'] = rot

    rip = []
    rip.append([iterlist['torque'].loc[k] for k in range(6832, 6893)])
    base['rip'] = rip[0]

    max = base['rip'].max()
    min = base['rip'].min()
    avg = np.round(base['rip'].mean(), 3)
    amp = max - avg

    base['min'] = min
    base['max'] = max
    base['avg'] = avg
    base['amp'] = amp

    base.to_pickle(ModelDir.DATA / "res_base250.pkl")
    print(base)

if switch == 0:
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_bb.pkl")

    f = open(ModelDir.DATA / f'res_rbst250_bb.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i+61
        j = j+1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i+0, i+61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref.to_pickle(ModelDir.DATA / "res_ref250_bb.pkl")
    print(ref)
    temp0 = []
    for j in range(61):
        for i in range(len(ref)):
            temp0.append((ref['rip'].loc[i])[j])

    temp1 = [[] for i in range(61)]
    k = -1
    for i in range(61):
        k = k + 1
        for j in range(k*len(ref), (k+1)*len(ref)):
            temp1[i].append(temp0[j])

    dif = {}
    emax = []
    emin = []
    eavg = []
    for i in range(len(temp1)):
        emax.append(np.max(temp1[i]))
        emin.append(np.min(temp1[i]))
        eavg.append(np.round(np.mean(temp1[i]), 3))
    dif['emax'] = emax
    dif['emin'] = emin
    dif['eavg'] = eavg
    dif = pd.DataFrame(dif)
    dif.to_pickle(ModelDir.DATA / "res_dif250_bb.pkl")

if switch == 1:
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_cc.pkl")

    f = open(ModelDir.DATA / f'res_rbst250_cc.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i+61
        j = j+1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i+0, i+61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref.to_pickle(ModelDir.DATA / "res_ref250_cc.pkl")

    temp0 = []
    for j in range(61):
        for i in range(len(ref)):
            temp0.append((ref['rip'].loc[i])[j])

    temp1 = [[] for i in range(61)]
    k = -1
    for i in range(61):
        k = k + 1
        for j in range(k*len(ref), (k+1)*len(ref)):
            temp1[i].append(temp0[j])

    dif = {}
    emax = []
    emin = []
    eavg = []
    for i in range(len(temp1)):
        emax.append(np.max(temp1[i]))
        emin.append(np.min(temp1[i]))
        eavg.append(np.round(np.mean(temp1[i]), 3))
    dif['emax'] = emax
    dif['emin'] = emin
    dif['eavg'] = eavg
    dif = pd.DataFrame(dif)
    dif.to_pickle(ModelDir.DATA / "res_dif250_cc.pkl")

if switch == 2:
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_wc.pkl")

    f = open(ModelDir.DATA / f'res_rbst250_wc.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i+61
        j = j+1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i+0, i+61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref.to_pickle(ModelDir.DATA / "res_ref250_wc.pkl")

    temp0 = []
    for j in range(61):
        for i in range(len(ref)):
            temp0.append((ref['rip'].loc[i])[j])

    temp1 = [[] for i in range(61)]
    k = -1
    for i in range(61):
        k = k + 1
        for j in range(k*len(ref), (k+1)*len(ref)):
            temp1[i].append(temp0[j])

    dif = {}
    emax = []
    emin = []
    eavg = []
    for i in range(len(temp1)):
        emax.append(np.max(temp1[i]))
        emin.append(np.min(temp1[i]))
        eavg.append(np.round(np.mean(temp1[i]), 3))
    dif['emax'] = emax
    dif['emin'] = emin
    dif['eavg'] = eavg
    dif = pd.DataFrame(dif)
    dif.to_pickle(ModelDir.DATA / "res_dif250_wc.pkl")

if switch == 3:
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_pb.pkl")

    f = open(ModelDir.DATA / f'res_rbst250_pb.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i+61
        j = j+1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i+0, i+61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref.to_pickle(ModelDir.DATA / "res_ref250_pb.pkl")


    temp0 = []
    for j in range(61):
        for i in range(len(ref)):
            temp0.append((ref['rip'].loc[i])[j])

    temp1 = [[] for i in range(61)]
    k = -1
    for i in range(61):
        k = k + 1
        for j in range(k*len(ref), (k+1)*len(ref)):
            temp1[i].append(temp0[j])

    dif = {}
    emax = []
    emin = []
    eavg = []
    for i in range(len(temp1)):
        emax.append(np.max(temp1[i]))
        emin.append(np.min(temp1[i]))
        eavg.append(np.round(np.mean(temp1[i]), 3))
    dif['emax'] = emax
    dif['emin'] = emin
    dif['eavg'] = eavg
    dif = pd.DataFrame(dif)
    dif.to_pickle(ModelDir.DATA / "res_dif250_pb.pkl")

if switch == 4:
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_ff1.pkl")

    f = open(ModelDir.DATA / f'res_rbst250_ff1.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i+61
        j = j+1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i+0, i+61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref1 = ref
    # print(ref1)
    ################################################################################################################
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_ff2.pkl")
    iterlist = iterlist.reset_index(drop=True)

    f = open(ModelDir.DATA / f'res_rbst250_ff2.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i + 61
        j = j + 1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i + 0, i + 61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref2 = ref
    # print(ref2)
    ################################################################################################################
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_ff3.pkl")
    iterlist = iterlist.reset_index(drop=True)

    f = open(ModelDir.DATA / f'res_rbst250_ff3.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i + 61
        j = j + 1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i + 0, i + 61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref3 = ref
    # print(ref3)
    ################################################################################################################
    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_ff4.pkl")
    iterlist = iterlist.reset_index(drop=True)

    f = open(ModelDir.DATA / f'res_rbst250_ff4.json')
    res = json.load(f)

    res = [np.round(i, 3) for i in res['Torque']]

    iterlist['torque'] = res

    ref = pd.DataFrame(columns=['c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y', 'c5x', 'c5y', 'rot', 'rip', 'min', 'max',
                                'avg', 'amp'])

    i = 0
    j = 0
    while i < len(iterlist):
        ref.loc[j] = iterlist.iloc[i, 0:8]
        i = i + 61
        j = j + 1

    rot = list(range(61))
    rot = np.tile(rot, (len(ref), 1))
    rot = list(rot)
    ref['rot'] = rot

    rip = []
    j = 0
    i = 0
    while i < len(iterlist):
        rip.append([iterlist['torque'].loc[k] for k in range(i + 0, i + 61)])
        i = i + 61
        j = j + 1
    ref['rip'] = rip

    min = []
    max = []
    avg = []
    amp = []
    for i in range(len(rip)):
        min.append(np.min(rip[i]))
        max.append(np.max(rip[i]))
        avg.append(np.round(np.mean(rip[i]), 3))
    ref['min'] = min
    ref['max'] = max
    ref['avg'] = avg
    amp = ref['max'] - ref['avg']
    ref['amp'] = amp

    ref4 = ref
    # print(ref4)
    refsum = pd.concat([ref1, ref2, ref3, ref4], ignore_index=True, sort=False)
    # print(refsum)
    # refsum.to_pickle(ModelDir.DATA / "res_ref250_ff.pkl")
    print(refsum['avg'].min())
    print(refsum['avg'].max())
    print(refsum['avg'].mean())
    print(np.std(list(refsum['avg'])))

    print(refsum['amp'].min())
    print(refsum['amp'].max())
    print(refsum['amp'].mean())