#import sys
# sys.path.append(r'C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10')
import powerfactory
import numpy as np
import pandas as pd
import math
import random


# exec(open(r"C:\Users\jsww999\Desktop\pythonProject\CreateNetwork.py").read())

LoadShape = np.loadtxt("load_profile_1.txt", dtype=float)
PVShape = np.loadtxt("PVloadshape.txt", dtype=float)
PVRating_Array = pd.read_csv('PV Ratings\PV_Rating.csv')
PVRating_Array = PVRating_Array.values
PVRating_Array = PVRating_Array[:, 1:]
PVRating_Array = np.amin(PVRating_Array, axis=0)
# PVRating_Array = np.full(28, 35.0, dtype=float)


class EnviroNet(object):
    def __init__(self):
        self.app = powerfactory.GetApplication()
        self.app.ActivateProject("Net_28_Clean.IntPrj")
        # self.PVsystems = self.app.GetCalcRelevantObjects('*.ElmPvsys')
        # self.Terms = self.app.GetCalcRelevantObjects('TES_*.ElmTerm')
        self.PVsystems = sorted(self.app.GetCalcRelevantObjects('*.ElmPvsys'), key=lambda obj: obj.GetAttribute('loc_name'))
        self.Terms = sorted(self.app.GetCalcRelevantObjects('TES_*.ElmTerm'), key=lambda obj: obj.GetAttribute('loc_name'))
        self.duration = len(PVShape)

        self.action_space = []
        for PV in self.PVsystems:
            for index in range(0, 2):
                self.action_space.append(PV)

    def state_shape(self):
        return ((len(self.Terms)*4), )

    def state_size(self):
        return (len(self.Terms)*4)

    def action_size(self):
        return len(self.action_space)

    def initialize(self, time):
        Solar_Irrad = PVShape[time]

        Loads = self.app.GetCalcRelevantObjects('*.ElmLodlv')
        for Load in Loads:
            Load.slini = LoadShape[time]
            Load.coslini = 0.95

        for idx, PV in enumerate(self.PVsystems):
            PV.sgn = PVRating_Array[idx]
            PV.pgini = PVRating_Array[idx]*PVShape[time]
            PV.qgini = 0

        # calculate loadflow
        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        ldf.Execute()

        Current_state = np.zeros(len(self.Terms)*2, dtype=float)
        index = 0
        for Term in self.Terms:
            P = Term.GetAttribute('m:Pflow')
            Q = Term.GetAttribute('m:Qflow')
            Current_state[index] = P*1000
            Current_state[index+1] = Q*1000
            index += 2
            #print('Active power of the time%d = %.6f' % (0, P))
            #print('Reactive power of the time%d = %.6f' % (0, Q))

        CustomerVoltage = np.zeros(len(self.Terms), dtype=float)
        for index, Term in enumerate(self.Terms):
            CustomerVoltage[index] = Term.GetAttribute('m:U1')

        Current_state = np.append(Current_state, CustomerVoltage)
        Current_state = np.append(Current_state, np.full(len(self.Terms), Solar_Irrad))

        # return Current_state


    def step_env(self, time):
        Solar_Irrad = PVShape[time]

        Loads = self.app.GetCalcRelevantObjects('*.ElmLodlv')
        for Load in Loads:
            Load.slini = LoadShape[time]
            Load.coslini = 0.95

        # calculate loadflow
        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        error = ldf.Execute()
        done = False
        if error == 1:
            done = True
            return [[], done]

        CustomerVoltage = np.zeros(len(self.Terms), dtype=float)
        index = 0
        for Term in self.Terms:
            CustomerVoltage[index] = Term.GetAttribute('m:U1')
            index += 1

        Current_state = np.zeros(len(self.Terms)*2, dtype=float)
        index = 0
        for Term in self.Terms:
            P = Term.GetAttribute('m:Pflow')
            Q = Term.GetAttribute('m:Qflow')
            Current_state[index] = P*1000
            Current_state[index+1] = Q*1000
            index += 2
        Current_state = np.append(Current_state, CustomerVoltage)
        Current_state = np.append(Current_state, np.full(len(self.Terms), Solar_Irrad))
        # Current_state = np.append(CustomerVoltage, np.full(len(self.Terms), Solar_Irrad))

        return [Current_state, done]



    def step_ctr(self, time, actions, Agent, evaluate):
        Solar_Irrad = PVShape[time]
        if Solar_Irrad == 1:
            Solar_Irrad = 0.99
        TotalP_Control = np.empty(len(self.PVsystems))
        TotalQ_Control = np.empty(len(self.PVsystems))
        Total_Curtailment = np.empty(len(self.PVsystems))

        for index in range(0, len(self.PVsystems)):
            P_available = Solar_Irrad * PVRating_Array[index]

            P_Control = P_available * (0.9 + (0.1*actions[2 * index]))
            #P_Control = P_available * (actions[2 * index] + 1) / 2

            Curtailment = P_available - P_Control
            if Curtailment < 0.01:
                Curtailment = 0.0

            Total_Curtailment[index] = Curtailment
            TotalP_Control[index] = P_Control

            self.action_space[2*index].pgini = P_Control
            Q_available = math.sqrt(PVRating_Array[index] ** 2 - P_Control ** 2)

            if Solar_Irrad == 0 and PVRating_Array[index] > 4:
                Q_available = 4

            if Solar_Irrad > 0 and Q_available > 4:
                Q_available = 4

            Q_control = Q_available * actions[(2 * index) + 1]
            TotalQ_Control[index] = Q_control
            self.action_space[(2*index)+1].qgini = Q_control

        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        error = ldf.Execute()
        if error == 1:
            done = True
            Vmax_limit = 258
            Vmin_limit = 220
            Nominal_Voltage = 230
            reward = -2 * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage)) / 0.05
            return [[], reward, [], done, [], [], [], [], [], [], []]

        Terms = self.app.GetCalcRelevantObjects('*.ElmTerm')
        TermVoltage = np.zeros(len(Terms), dtype=float)
        for index, Term in enumerate(Terms):
            TermVoltage[index] = Term.GetAttribute('m:U1')

        TermVoltage[0] = np.nan
        MaxVoltage = 1000*np.nanmax(TermVoltage)
        MinVolatge = 1000*np.nanmin(TermVoltage)

        CustomerVoltage = np.zeros(len(self.Terms), dtype=float)
        for index, Term in enumerate(self.Terms):
            CustomerVoltage[index] = Term.GetAttribute('m:U1')

        Next_state = np.zeros(len(self.Terms)*2, dtype=float)
        index = 0
        for Term in self.Terms:
            P = Term.GetAttribute('m:Pflow')
            Q = Term.GetAttribute('m:Qflow')
            Next_state[index] = P*1000
            Next_state[index+1] = Q*1000
            index += 2

        Next_state = np.append(Next_state, CustomerVoltage)
        Next_state = np.append(Next_state, np.full(len(self.Terms), Solar_Irrad))
        # Next_state = np.append(CustomerVoltage, np.full(len(self.Terms), Solar_Irrad))
        Transformer_Term = self.app.GetCalcRelevantObjects('TEM_1.ElmTerm')[0]
        Loading = Transformer_Term.GetAttribute('m:Sout')

        done = False
        Vmax_limit = 258
        Vmin_limit = 207
        Nominal_Voltage = 230
        weight = 20000

        reward = 0
        Total_Output_Power = 0

        for index in range(0, len(self.PVsystems)):
            Total_Output_Power += TotalP_Control[index]
            if Solar_Irrad == 0:
                reward += 0
            else:
                Delta = weight * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage)) / (PVRating_Array[index] * Solar_Irrad)
                reward += -Delta * Total_Curtailment[index]

        reward = (reward / len(self.PVsystems)) - abs(MaxVoltage - Nominal_Voltage) - abs(MinVolatge - Nominal_Voltage)

        if (MaxVoltage > Vmax_limit) or (MinVolatge < Vmin_limit):
            reward = (-4 * weight * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage))) - (abs(MaxVoltage - Nominal_Voltage)) - abs(MinVolatge - Nominal_Voltage)

        if (Loading > 1):
            reward = (-4 * weight * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage))) - 1000* weight *(abs(Loading - 0.5))

        return [Next_state, reward, actions, done, MaxVoltage, MinVolatge, CustomerVoltage, Total_Curtailment, TotalQ_Control, TotalP_Control, Loading, Solar_Irrad]


















