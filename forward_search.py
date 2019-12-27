import numpy as np
from passengers import loadfactor_monthly
from passengers import get_sample 
import math
import numbers
from matplotlib import pyplot as plt
from passengers import fuel
#--------------------Problem Parameters
gamma=1;
capacity=130;
capacity_1=150;
capacity_2=120;
state_space=np.array([[1,1],[1,0],[0,1],[0,0]])
year0=2020
#Initial conditions and problem parameter declaration
#----------------------------Parameters that will remain fixed during time stepping

#Operating cost per passenger:

OC=np.array([[40,10.2,30],
    [40,15,40]]) # OC[0]: fixed costs independent of aircraft(like terminal/port costs)
#print(OC.shape)                         # OC[1]: fuel cost per passenger per litre. Needs to be * by total fuel volume used
                         # OC[2]: fixed costs dependent on aircraft(maintenance etc)
#Ticket Price:
ticket_price_1=90   # Flat, independent of plane
ticket_price_2=120

#-----------------------------Parameters that will vary during timestepping in MDP

#---Availability Probabilities
#Store pseudocounts of beta distribution for availability of both aircraft

#---Load Factor

[mean_values,std_values]=loadfactor_monthly() # mean_values consist of a 12x2 matrix, 12: monthly and 2: parameters m and c of linear fit y=mx+c 
                                            # where y is the predicted  mean of load factor in the desired year for that month. std is the stan
                                            # dard deviation of load factor in that month(predicted).


#-----



def best_action(counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,state):

    global OC               # Forward search recursive loop
    global gamma;
    global state_space
    global year0
    if curr_month==13:
        curr_month=1;
        curr_year+=1;

    if d==0:

        return [0,[]];

    action_star=[]
    reward_star=-float('inf');
    
    actions=action_space(state);
    if len(actions)==0: # If no action can be taken from current state, since both planes are unavailable

        reward_current=noaction_reward();
        i_action=[]
        [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)    
        for i_state in state_space:
            expected_prob=transition(counts_newplane_new,counts_oldplane_new,i_state);
            [v_dash,a_dash]=best_action(counts_newplane_new,counts_oldplane_new,mean_values,std_values,curr_month+1,curr_year,d-1,i_state)
            reward_current+=gamma*expected_prob*v_dash

        if reward_current > reward_star:

            action_star=i_action;
            reward_star=reward_current;


    else:
        load_factor=(0.01)*np.random.normal(curr_year*mean_values[curr_month-1,0]+mean_values[curr_month-1,1],std_values[curr_month-1])
        for i_action in actions:

            OC_current=OC[i_action-1,:];
            load_factor=0.01*(curr_year*mean_values[curr_month-1,0]+mean_values[curr_month-1,1])
            reward_current=reward_func(OC_current,curr_month,curr_year,load_factor)
            [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)
            
            for i_state in state_space:
                expected_prob=transition(counts_newplane_new,counts_oldplane_new,i_state);
                [v_dash,a_dash]=best_action(counts_newplane_new,counts_oldplane_new,mean_values,std_values,curr_month+1,curr_year,d-1,i_state)
                reward_current+=gamma*expected_prob*v_dash

            if reward_current > reward_star:

                action_star=i_action;
                reward_star=reward_current;
    #print(action_star)
    return [reward_star,action_star];


def action_space(state): #Possible actions from given state

    if state[0]==1:

        if state[1]==1:

            return [1,2];
        if state[1]==0:

            return [1];
    if state[0]==0:

        if state[1]==1:
            return [2];

        if state[1]==0:
            return [];

def reward_func(OC_current,curr_month,curr_year,load_factor): # Reward function: Use as it is
    

    global capacity_1
    global capacity_2
    global ticket_price_1
    global ticket_price_2
    global year0
    fuel_ppl=fuel(int(curr_month),int(curr_year-year0)) # Fuel consumed per person
    if load_factor >0.85:
        ticket_price=ticket_price_2;
    else:
        ticket_price=ticket_price_1;
    operating_cost= OC_current[0]+OC_current[1]*fuel_ppl+OC_current[2];
    if OC_current[2]==30:
        return capacity_1*(ticket_price*load_factor-operating_cost);
    else:
        return capacity_2*(ticket_price*load_factor-operating_cost);
def update_counts(i_action,counts_newplane,counts_oldplane):  # This function updates Beta dist. counters based on actions


    if i_action ==1:

        counts_newplane_new=[counts_newplane[0],counts_newplane[1]+1];
        counts_oldplane_new=[counts_oldplane[0]+2,counts_oldplane[1]];
        return [counts_newplane_new,counts_oldplane_new];
    if i_action==2:

        counts_oldplane_new=[counts_oldplane[0],counts_oldplane[1]+1];
        counts_newplane_new=[counts_newplane[0]+2,counts_newplane[1]];
        return [counts_newplane_new,counts_oldplane_new];

    if len(i_action)==0:

        counts_newplane_new=[counts_newplane[0]+2,counts_newplane[1]];
        counts_oldplane_new=[counts_oldplane[0]+2,counts_oldplane[1]];
        return [counts_newplane_new,counts_oldplane_new];


def transition(counts_newplane,counts_oldplane,state): # Calculates T(s'|s,a)

    ticker_newplane=0.01;
    ticker_oldplane=0.01;
    if counts_newplane[0]-counts_newplane[1]>4:
        ticker_newplane=0.99;
    if counts_oldplane[0]-counts_oldplane[1]>4:
        ticker_oldplane=0.99;

    

    if state[0]==1:
        if state[1]==1:
            return ticker_newplane*ticker_oldplane
        if state[1]==0:
            return ticker_newplane*(1-ticker_oldplane);
    if state[0]==0:
        if state[1]==1:
            return (1-ticker_newplane)*ticker_oldplane
        if state[1]==0:
            return (1-ticker_newplane)*(1-ticker_oldplane)
def noaction_reward(): # Reward for reaching state [0,0] and hence taking no action
    global OC
    global capacity
    return -OC[0,0]*capacity

def execute_policy(actions,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,fuel_price,state):
                                                   # This function simulates execution of the optimal policy, executed once in every timestep
    if not(isinstance(actions, numbers.Integral)): # If no action can be taken from current state, since both planes are unavailable

        reward_current=noaction_reward();
        i_action=[]
        [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)
        expected_prob=np.zeros([state_space.shape[0]]);
        count_istate=0;
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1
        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob);
        next_state=state_space[temp,:];
    else:
        i_action=actions;
        load_factor=(0.01)*np.random.normal(curr_year*mean_values[curr_month-1,0]+mean_values[curr_month-1,1],std_values[curr_month-1])
        OC_current=OC[i_action-1,:];
        reward_current=reward_func(OC_current,curr_month,curr_year,load_factor)
        [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)
        expected_prob=np.zeros([state_space.shape[0]]);

        count_istate=0;
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1;
        
        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob);
        next_state=state_space[temp,:];
    curr_month+=1;
    if curr_month==13:
        curr_month=1;
        curr_year+=1;
    return [reward_current,counts_newplane_new,counts_oldplane_new,next_state,curr_month,curr_year];


def execute_policy_2(actions,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,fuel_price,state):
    switch_a=0;                             # This function is meant to simulate for an arbitrary policy, as shown below
    possible_actions=action_space(state);
    if ((actions in possible_actions)) :  # Policy #1: 1 or nothing
        switch_a=1;
    
    elif (actions+1 in possible_actions):  # Comment out for  1 or nothing policy
        actions+=1;
        switch_a=1;                      #Policy #2: 1 when possible, if 1 not available 2.
    
    if  switch_a==0: # If no action can be taken from current state, since both planes are unavailable

        reward_current=noaction_reward();
        i_action=[]
        [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)
        expected_prob=np.zeros([state_space.shape[0]]);
        count_istate=0;
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1
        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob); # Simulate next state according to probabilities
        next_state=state_space[temp,:];
    else:
        i_action=actions;
        load_factor=(0.01)*np.random.normal(curr_year*mean_values[curr_month-1,0]+mean_values[curr_month-1,1],std_values[curr_month-1])
        OC_current=OC[i_action-1,:];
        reward_current=reward_func(OC_current,curr_month,curr_year,load_factor)
        [counts_newplane_new,counts_oldplane_new]=update_counts(i_action,counts_newplane,counts_oldplane)
        expected_prob=np.zeros([state_space.shape[0]]);
        count_istate=0;
        for i_state in state_space:
            expected_prob[count_istate]=transition(counts_newplane_new,counts_oldplane_new,i_state);
            count_istate+=1;

        temp=np.random.choice(range(state_space.shape[0]),p=expected_prob); # Simulate next state according to probabilities
        next_state=state_space[temp,:];
    curr_month+=1;
    if curr_month==13:
        curr_month=1;
        curr_year+=1;
    return [reward_current,counts_newplane_new,counts_oldplane_new,next_state,curr_month,curr_year];


def running_average(reward_array,i_sim):
    return sum(reward_array[:i_sim+1])/float((i_sim+1));

#--------------------------Simulation
import time
start_time = time.time()


d=5;            # Horizon
total_optim=0;
total_1=0;
i_sim=0;
rewards_sim_optim=np.zeros([0]);
rewards_sim_1=np.zeros([0]);


while True:
    print(i_sim)
    counts_newplane=[4,1];
    counts_oldplane=[4,1]; # Assumed initial numbers(pseudocounts)


    nMonths=10;
    state=[1,1];
    total_reward=0;
    curr_month=2
    curr_year=2020
    for i_month in range(nMonths):
        [exp_profit,policy]=best_action(counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,state);
        [reward_current,counts_newplane_new,counts_oldplane_new,next_state,next_month,next_year]=execute_policy(policy,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,fuel_price,state)
        total_reward+=reward_current;
        counts_newplane=counts_newplane_new;
        counts_oldplane=counts_oldplane_new;
        state=next_state;
        curr_month=next_month;
        curr_year=next_year;
    rewards_sim_optim=np.append(rewards_sim_optim,total_reward);

    curr_year=2020;
    curr_month=2;
    total_optim+=total_reward;
    total_reward=0;
    counts_newplane=[4,1];
    counts_oldplane=[4,1];
    state=[1,1];
    #print("------------------------")
    for i_month in range(nMonths):
        [reward_current,counts_newplane_new,counts_oldplane_new,next_state,next_month,next_year]=execute_policy_2(1,counts_newplane,counts_oldplane,mean_values,std_values,curr_month,curr_year,d,fuel_price,state)
        total_reward+=reward_current;
        counts_newplane=counts_newplane_new;
        counts_oldplane=counts_oldplane_new;
        state=next_state;
        curr_month=next_month;
        curr_year=next_year;
        
    rewards_sim_1=np.append(rewards_sim_1,total_reward);
    total_1+=total_reward
    i_sim+=1;
    if i_sim>3 and abs(running_average(rewards_sim_optim,i_sim)-running_average(rewards_sim_optim,i_sim-1))<20:
        break;
    
nSim=i_sim;
print(total_optim/nSim,total_1/nSim);
count_optim=0;
count_1=0;
print(nSim,rewards_sim_optim.shape);

for i_sim in range(nSim):
    
    if rewards_sim_optim[i_sim]>rewards_sim_1[i_sim]:

        count_optim+=1;

    else:

        count_1+=1;

print(count_optim,count_1);

running_av_optim=np.zeros([nSim]);
running_av_1=np.zeros([nSim]);

for i_sim in range(nSim):

    running_av_optim[i_sim]=running_average(rewards_sim_optim,i_sim);
    running_av_1[i_sim]=running_average(rewards_sim_1,i_sim);

nDays=30;

plt.plot(range(nSim),running_av_optim*nDays,label='Average Reward for Optimal Policy');
plt.plot(range(nSim),running_av_1*nDays,label='Average Reward for Greedy Policy');
plt.legend()
plt.xlabel('Simulation Iteration');
plt.ylabel('Running Average of Reward')
print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
