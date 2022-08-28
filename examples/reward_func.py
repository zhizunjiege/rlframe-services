def func(states, actions, next_states, done):
    if done:
        return 1.0
    else:
        return 0.0
