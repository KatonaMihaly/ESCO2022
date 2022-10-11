import json
from multiprocessing import Pool

import pandas as pd

from digital_twin_distiller.encapsulator import Encapsulator
from digital_twin_distiller.modelpaths import ModelDir
from digital_twin_distiller.simulationproject import sim

from model_robustness import RobustPriusMotor

def execute_model(model: RobustPriusMotor):
    return model(timeout=2000, cleanup=False).get("Torque", 0.0) * -8

@sim.register('default')
def default_simulation(model, modelparams, simparams, miscparams):
    return "Hello World!"

@sim.register('rbst')
def rbst(model, modelparams, simparams, miscparams):

    iterlist = pd.read_pickle(ModelDir.DATA / "df_rbst250_ff1.pkl")
    a = 0
    b = 1
    models = [model(earheight=2.1, I0=250,
                    prob2x=c2x, prob2y=c2y,
                    prob3x=c3x, prob3y=c3y,
                    prob4x=c4x, prob4y=c4y,
                    prob5x=c5x, prob5y=c5y,
                    alpha=alp, rotorangle=rot,
                    ) for c2x, c2y, c3x, c3y, c4x, c4y, c5x, c5y, alp, rot in
              zip(list(iterlist['c2x'].iloc[a:b]), list(iterlist['c2y'].iloc[a:b]),
                  list(iterlist['c3x'].iloc[a:b]), list(iterlist['c3y'].iloc[a:b]),
                  list(iterlist['c4x'].iloc[a:b]), list(iterlist['c4y'].iloc[a:b]),
                  list(iterlist['c5x'].iloc[a:b]), list(iterlist['c5y'].iloc[a:b]),
                  list(iterlist['alp'].iloc[a:b]), list(iterlist['rot'].iloc[a:b]))]
    with Pool() as pool:
        res = pool.map(execute_model, models)

    result = {'Torque': list(res)}

    with open(ModelDir.DATA / f'res_rbst250_ff1.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    return result

if __name__ == "__main__":
    ModelDir.set_base(__file__)

    # set the model for the simulation
    sim.set_model(RobustPriusMotor)

    model = Encapsulator(sim)
    model.port = 8080
    model.run()