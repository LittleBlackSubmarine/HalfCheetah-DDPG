from tensorflow.keras.layers import Dense, Flatten, Concatenate, Input
from tensorflow.keras.models import Model



def get_actor(obs, actions):
    state_input = Input(shape=(1,) + obs)
    fl_state_input = Flatten()(state_input)
    h1 = Dense(400, activation='relu')(fl_state_input)
    h2 = Dense(300, activation='relu')(h1)
    output = Dense(actions[0], activation='tanh')(h2)
    actor = Model(inputs=state_input, outputs=output)
    print(h1)
    print(output)


    return actor


def get_critic(obs, actions):
    state_input = Input(shape=(1,) + obs, name='obs_input')
    fl_state_input = Flatten()(state_input)
    action_input = Input(shape=actions, name='action_input')
    h1 = Dense(400, activation='relu')(fl_state_input)
    hc = Concatenate()([h1, action_input])
    h2 = Dense(300, activation='relu')(hc)
    output = Dense(1)(h2)
    critic = Model(inputs=[action_input, state_input], outputs=output)


    return critic, action_input
