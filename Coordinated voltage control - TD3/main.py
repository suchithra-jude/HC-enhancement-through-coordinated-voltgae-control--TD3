import numpy as np
import pandas as pd
import datetime
from td3 import Agent
from Environment import EnviroNet
from utils import plot_result_curve

if __name__ == '__main__':
    Date_time = np.loadtxt("Datetime.txt", dtype=str)
    env = EnviroNet()
    Time_steps = env.duration
    agent = Agent(input_dims=env.state_shape(), n_actions=env.action_size())

    load_checkpoint = True

    Episodes = 1
    reward = 0
    Vmax = 0
    Vmin = 0
    Term_Voltage = 0
    Curtailment = 0
    Q_Output = 0
    P_Output = 0
    Irradaition = 0
    TrfLoading = 0


    evaluate = False
    if load_checkpoint:
        n_steps = 0
        Time = 0
        env.initialize(0)
        agent.batch_size = 50
        while n_steps <= agent.batch_size + 6:
            observation, done = env.step_env(Time)
            action = np.random.rand(env.action_size())
            observation_, reward, correctAction, done, Vmax, Vmin, Term_Voltage, Curtailment, Q_Output, P_Output, TrfLoading, Irradaition = env.step_ctr(
                Time, action, agent, evaluate)
            agent.remember(observation, correctAction, reward, observation_, done)
            n_steps += 1
            agent.learn()
        agent.load_models()
        evaluate = True


    score_history = []
    Max_V_history = []
    Voltage_history = np.zeros([Time_steps, len(env.PVsystems)])
    PV_Output_history = np.zeros([Time_steps, len(env.PVsystems)])
    Curtailment_history = np.zeros([Time_steps, len(env.PVsystems)])
    Reward_history = np.zeros([48, Episodes])
    Reward_history_plot = np.zeros([int(Episodes*(Time_steps/48)), 2])

    figure_ResultR = 'plots/CumulativeReward.png'
    figure_ResultV = 'plots/Voltage.png'

    best_score = -30000
    avg_score = -30000
    score = 0
    env.initialize(0)

    for Time in range(0, Time_steps):
        done = False
        observation, done = env.step_env(Time)

        for Episode in range(0, Episodes):
            action = agent.choose_action(observation, evaluate)
            out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12 = env.step_ctr(Time, action, agent, evaluate)
            score += out2
            done = out4
            if not done:
                observation_ = out1
                reward = out2
                correctAction = out3
                done = out4
                Vmax = out5
                Vmin = out6
                Term_Voltage = out7
                Curtailment = out8
                Q_Output = out9
                P_Output = out10
                TrfLoading = out11
                Irradaition = out12
                agent.remember(observation, correctAction, reward, observation_, done)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_
            else:
                env.initialize(Time)

            Reward_history[Time % 48, Episode] = reward


        if (Time+1) % 48 == 0:
            Daily_reward = np.sum(Reward_history, axis=0)
            for Episode in range(0, Episodes):
                Reward_history_plot[(Episodes * (int((Time+1) / 48) - 1)) + Episode, 0] = (Time+1)/48
                Reward_history_plot[(Episodes * (int((Time+1) / 48) - 1)) + Episode, 0] = Daily_reward[Episode]

            score_history.append(score/Episodes)
            avg_score = np.mean(score_history[-5:])
            Reward_history = np.zeros([48, Episodes])
            score = 0

        if (Time+1) % 3120 == 0:
            agent.batch_size += 500

        Max_V_history.append(Vmax)
        Voltage_history[Time, :] = Term_Voltage
        PV_Output_history[Time, :] = P_Output
        Curtailment_history[Time, :] = Curtailment

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
                env.app.PrintPlain('... saving models ...')


        '''
        print('Time:', Date_time[Time], 'score: %.1f' % score,
            'avg score: %.1f' % best_score, 'maximum voltage: %.4f' % Vmax, 'PV_Rating: ', PV_Rating,
            'Curtailment: %.1f' % Curtailment, 'avg_Q: ', Q_Output, 'avg_P_Out: ', P_Output)
        '''

        env.app.PrintPlain(
            'Time: %s score: %.1f best score: %.1f maximum voltage: %.4f minimum voltage: %.4f Curtailment: %s avg_Q: %s avg_P_Out: %s Trf_Load: %.4f Irradiation: %.4f'
            % (Date_time[Time], score/Episodes, best_score, Vmax, Vmin, str(np.sum(Curtailment)),
               str(np.mean(Q_Output)), str(np.mean(P_Output)), TrfLoading, Irradaition))

    if not load_checkpoint:
        agent.save_models()
        env.app.PrintPlain('... saving models ...')
        DF = pd.DataFrame(Reward_history_plot)
        DF.to_csv("PV Ratings/Reward_history_plot.csv")

    if evaluate:
        Term_Name = []
        for Term in env.Terms:
            Term_Name.append(Term.GetAttribute('loc_name'))

        DF = pd.DataFrame(Voltage_history)
        #DF.columns = Term_Name
        DF.to_csv("PV Ratings/TD3_Voltage_history.csv")
        DF = pd.DataFrame(PV_Output_history)
        DF.to_csv("PV Ratings/TD3_PV_Output.csv")
        DF = pd.DataFrame(Curtailment_history)
        DF.to_csv("PV Ratings/TD3_curtailment_history.csv")



    x = [i + 1 for i in range(len(score_history))]
    y = [i + 1 for i in range(len(Max_V_history))]
    plot_result_curve(x, score_history, figure_ResultR, 'Cumulative reward')
    plot_result_curve(y, Max_V_history, figure_ResultV, 'Maximum Voltage')

    del env.app