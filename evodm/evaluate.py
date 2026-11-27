"""Greedy evaluation of a trained agent, with optional mismatched staleness (added for the
delay-history study; see CHANGES.md).

The evaluation runs the trained policy with no exploration and no learning on fresh episodes.
For the delayed-observation modes the reading can be made staler or fresher than the agent was
trained for (``stale``) while the network input keeps the shape it was trained with.
"""
import numpy as np

from evodm.learner import practice


def per_episode(agent):
    """Mean fitness of each episode, from the agent's master memory."""
    d = {}
    for row in agent.master_memory:
        d.setdefault(row[0], []).append(float(np.mean(row[-1])))
    return np.array([np.mean(d[k]) for k in sorted(d)])


def greedy_eval(agent, stale=None, episodes=100, trace=None):
    """Run a trained agent greedily (epsilon 0, no weight updates) for ``episodes`` episodes.

    stale: real staleness of the reading during evaluation (delayed-observation modes only).
           None keeps the staleness the agent was trained with.
    trace: optional list; one row (episode, action_number, action, genotype, fitness) is appended
           for every simulated step, where genotype is the true genotype before the step and the
           action is the drug given (the first step of an episode is not chosen by the agent).
    Returns an array with the mean fitness of every evaluation episode (lower is better).
    The agent's own settings are restored afterwards.
    """
    env = agent.env
    saved = dict(episodes=agent.hp.EPISODES, epsilon=agent.hp.epsilon,
                 memory=agent.master_memory, stale=getattr(env, "OBS_STALE", None))
    try:
        if stale is not None:
            if not getattr(env, "OBS_MODE", False):
                raise ValueError("stale can only be set in the delayed-observation modes")
            env.OBS_STALE = int(stale)
        agent.hp.EPISODES = int(episodes)
        agent.master_memory = []
        env.reset()  # rebuild the first observation under the chosen staleness
        original_step = env.step
        if trace is not None:
            def traced_step():
                before = int(np.argmax(np.array(env.state_vector).ravel()))
                original_step()
                trace.append((env.episode_number, env.action_number, int(env.action), before,
                              float(np.mean(env.fitness))))
            env.step = traced_step
        try:
            practice(agent, pre_trained=True)
        finally:
            if trace is not None:
                env.step = original_step
        return per_episode(agent)
    finally:
        agent.hp.EPISODES = saved["episodes"]
        agent.hp.epsilon = saved["epsilon"]
        agent.master_memory = saved["memory"]
        if saved["stale"] is not None:
            env.OBS_STALE = saved["stale"]
        env.reset()


def reference_eval(agent, kind, episodes=100):
    """Evaluate a reference policy ('naive' = random drugs, 'dp' = exact optimum) on fresh episodes."""
    saved_episodes, saved_memory = agent.hp.EPISODES, agent.master_memory
    try:
        agent.hp.EPISODES = int(episodes)
        agent.master_memory = []
        if kind == "naive":
            practice(agent, naive=True)
        elif kind == "dp":
            practice(agent, dp_solution=True, discount_rate=agent.hp.DISCOUNT)
        else:
            raise ValueError("kind must be 'naive' or 'dp'")
        return per_episode(agent)
    finally:
        agent.hp.EPISODES = saved_episodes
        agent.master_memory = saved_memory
