# import sys
# sys.path.append(r'C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.10')
import powerfactory
import numpy as np
import math
import random

# exec(open(r"C:\Users\jsww999\Desktop\pythonProject\CreateNetwork.py").read())

LoadShape = np.loadtxt("load_profile_1.txt", dtype=float)
PVShape = np.loadtxt("PVloadshape.txt", dtype=float)


class EnviroNet(object):
    def __init__(self):
        self.app = powerfactory.GetApplication()
        self.app.ActivateProject("Net_28_Clean.IntPrj")
        self.PVsystems = self.app.GetCalcRelevantObjects('*.ElmPvsys')
        self.Terms = self.app.GetCalcRelevantObjects('TES_*.ElmTerm')
        self.duration = 5760 # len(PVShape)

        self.action_space = []
        for PV in self.PVsystems:
            for index in range(0, 3):
                self.action_space.append(PV)

    def state_shape(self):
        return ((len(self.Terms)*2), )

    def state_size(self):
        return (len(self.Terms)*2)

    def action_size(self):
        return len(self.action_space)

    def NoControl(self, time):

        Loads = self.app.GetCalcRelevantObjects('*.ElmLodlv')
        for Load in Loads:
            Load.slini = LoadShape[time]
            Load.coslini = 0.95

        for PV in self.PVsystems:
            PV.pgini = 7*PVShape[time]
            PV.qgini = 0

        # calculate loadflow
        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        ldf.Execute()

        Terms = self.app.GetCalcRelevantObjects('*.ElmTerm')
        TermVoltage = np.zeros(len(Terms), dtype=float)
        index = 0
        for Term in Terms:
            TermVoltage[index] = Term.GetAttribute('m:U1')
            index += 1

        TermVoltage[0] = 0
        MaxVoltage = np.amax(TermVoltage)
        MinVolatge = np.amin(TermVoltage)

        return MaxVoltage

    def initialize(self, time):
        Solar_Irrad = PVShape[time]

        Loads = self.app.GetCalcRelevantObjects('*.ElmLodlv')
        for Load in Loads:
            Load.slini = LoadShape[time]
            Load.coslini = 0.95

        for PV in self.PVsystems:
            PV.pgini = 7*PVShape[time]
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
        index = 0
        for Term in self.Terms:
            CustomerVoltage[index] = Term.GetAttribute('m:U1')
            index += 1

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

        '''
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
        '''

        CustomerVoltage = np.zeros(len(self.Terms), dtype=float)
        index = 0
        for Term in self.Terms:
            CustomerVoltage[index] = Term.GetAttribute('m:U1')
            index += 1

        # Current_state = np.append(Current_state, CustomerVoltage)
        # Current_state = np.append(Current_state, np.full(len(self.Terms), Solar_Irrad))
        Current_state = np.append(CustomerVoltage, np.full(len(self.Terms), Solar_Irrad))

        return [Current_state, done]



    def step_ctr(self, time, actions, evaluate, warmup):
        Solar_Irrad = PVShape[time]
        TotalP_Rating = np.empty(len(self.PVsystems))
        TotalP_Control = np.empty(len(self.PVsystems))
        TotalQ_Control = np.empty(len(self.PVsystems))
        Total_Curtailment = np.empty(len(self.PVsystems))

        Terms = self.app.GetCalcRelevantObjects('*.ElmTerm')
        TermVoltage = np.zeros(len(Terms), dtype=float)
        index = 0
        for Term in Terms:
            TermVoltage[index] = Term.GetAttribute('m:U1')
            index += 1

        TermVoltage[0] = 0
        PreviousMaxVoltage = np.amax(TermVoltage)

        Max_HC = 120
        for index in range(0, len(self.PVsystems)):
            PV_Rating = Max_HC * ((actions[3 * index]) + 1) / 2
            PV_Rating = np.clip(PV_Rating, 1, Max_HC)
            TotalP_Rating[index] = PV_Rating
        if (not evaluate) and (time < warmup):
            MeanPV_Rating = random.randrange(1, Max_HC)
            TotalP_Rating = np.clip(TotalP_Rating, 0.9 * MeanPV_Rating, 1.1 * MeanPV_Rating)
        else:
            MeanPV_Rating = np.mean(TotalP_Rating)
            TotalP_Rating = np.clip(TotalP_Rating, 0.9 * MeanPV_Rating, 1.1 * MeanPV_Rating)



        for index in range(0, len(self.PVsystems)):
            PV_Rating = TotalP_Rating[index]
            actions[3 * index] = (2*PV_Rating/Max_HC)-1
            self.action_space[(3 * index)].sgn = PV_Rating

            P_available = Solar_Irrad * PV_Rating
            P_Control = P_available * (0.9 + (0.1*actions[(3 * index) + 1]))

            Curtailment = (Max_HC*Solar_Irrad) - P_Control

            Total_Curtailment[index] = Curtailment
            TotalP_Control[index] = P_Control
            if P_Control < 0:
                P_Control = 0

            self.action_space[(3*index)+1].pgini = P_Control
            Q_available = math.sqrt(PV_Rating ** 2 - P_Control ** 2)
            if Solar_Irrad == 0 and PV_Rating > 4:
                Q_available = 4

            if Solar_Irrad > 0 and Q_available > 4:
                Q_available = 4

            Q_control = Q_available * actions[(3 * index) + 2]
            # Q_control = PV_Rating * actions[(3 * index) + 2]
            TotalQ_Control[index] = Q_control
            self.action_space[(3*index)+2].qgini = Q_control



        ldf = self.app.GetFromStudyCase('ComLdf')
        ldf.iopt_net = 1
        error = ldf.Execute()
        if error == 1:
            done = True
            Vmax_limit = 260
            Vmin_limit = 220
            Nominal_Voltage = 230
            reward = -2 * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage)) / 0.05
            return [[], reward, [], done, [], [], [], [], [], [], []]

        '''
        Next_state = np.zeros(len(self.Terms)*2, dtype=float)
        index = 0
        for Term in self.Terms:
            P = Term.GetAttribute('m:Pflow')
            Q = Term.GetAttribute('m:Qflow')
            Next_state[index] = P*1000
            Next_state[index+1] = Q*1000
            index += 2
            #print('Active power of the time%d = %.6f' % (0, P))
            #print('Reactive power of the time%d = %.6f' % (0, Q))
        '''

        Terms = self.app.GetCalcRelevantObjects('*.ElmTerm')
        TermVoltage = np.zeros(len(Terms), dtype=float)
        index = 0
        for Term in Terms:
            TermVoltage[index] = Term.GetAttribute('m:U1')
            index += 1

        TermVoltage[0] = np.nan
        MaxVoltage = 1000*np.nanmax(TermVoltage)
        MinVolatge = 1000*np.nanmin(TermVoltage)


        CustomerVoltage = np.zeros(len(self.Terms), dtype=float)
        index = 0
        for Term in self.Terms:
            CustomerVoltage[index] = Term.GetAttribute('m:U1')
            index += 1

        # Next_state = np.append(Next_state, CustomerVoltage)
        # Next_state = np.append(Next_state, np.full(len(self.Terms), Solar_Irrad))
        Next_state = np.append(CustomerVoltage, np.full(len(self.Terms), Solar_Irrad))

        Transformer_Term = self.app.GetCalcRelevantObjects('TEM_1.ElmTerm')[0]
        Loading = Transformer_Term.GetAttribute('m:Sout')

        done = False
        Vmax_limit = 260
        Vmin_limit = 220
        Nominal_Voltage = 230
        weight = 20

        reward = 0
        Total_Output_Power = 0
        Delta = weight * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage)) / (Max_HC * Solar_Irrad)
        for index in range(0, len(self.PVsystems)):
            Total_Output_Power += TotalP_Control[index]
            if Solar_Irrad == 0:
                reward += 0
            else:
                reward += -Delta * Total_Curtailment[index]

        reward = (reward / len(self.PVsystems)) - abs(MaxVoltage - Nominal_Voltage) - abs(MinVolatge - Nominal_Voltage)

        if (MaxVoltage > Vmax_limit) or (MinVolatge < Vmin_limit):
            reward = (-4 * weight * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage))) - (abs(MaxVoltage - Nominal_Voltage)) - abs(MinVolatge - Nominal_Voltage)
            done = False
            for index in range(0, len(self.PVsystems)):
                TotalP_Control[index] = 0

        if (Loading > 1):
            reward = (-4 * weight * (abs(Vmax_limit - Nominal_Voltage) + abs(Vmin_limit - Nominal_Voltage))) - 1000* weight *(abs(Loading - 0.5))
            done = False
            for index in range(0, len(self.PVsystems)):
                TotalP_Control[index] = 0

        return [Next_state, reward, actions, done, MaxVoltage, MinVolatge, TotalP_Rating, np.mean(Total_Curtailment), TotalQ_Control, TotalP_Control, Solar_Irrad]


















