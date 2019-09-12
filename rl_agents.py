
class DQN_Agent():
    '''
    Regular Q-Learning Agent
    One deep network.
    DQN - to predict Q of a given action, value a state. i.e. Q(s,a) and Q(s', a') for loss calculation.
    '''

class DDQN_Agent():
    '''
    Double DQN Agents
    DQN - to predict Q of a given action, value a state. i.e. Q(s,a) and the  a' = argmax Q(s', a)
    Target - used to calc Q(s', a') for loss calculation.
    '''

class DDDQN_Agent():
    '''
    Dueling Double DQN Agents

    '''