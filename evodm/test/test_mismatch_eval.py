"""Tests for the mismatched-staleness observation and the greedy evaluation."""
import numpy as np
import pytest
from evodm import evol_env, define_mira_landscapes, evol_deepmind
from evodm.evaluate import greedy_eval, reference_eval

N, K = 4, 15


def make_env(mode, w):
    return evol_env(N=N, num_drugs=K, drugs=define_mira_landscapes(), train_input=mode,
                    obs_delay=w, add_noise=False, normalize_drugs=False)


def onehot(a):
    v = np.zeros(K)
    v[a - 1] = 1.0
    return v


def play(env, actions):
    for a in actions:
        env.action = a
        env.step()


def window(env):
    return env.build_obs()[16:].reshape(-1, K)


def test_default_staleness_equals_window_length():
    env = make_env("delayed_history", 3)
    assert env.OBS_STALE == env.OBS_DELAY == 3


def test_staler_than_trained_drops_oldest_drugs():
    env = make_env("delayed_history", 2)          # network built for a window of 2
    play(env, [3, 5, 7, 9])                       # t = 4 actions taken
    env.OBS_STALE = 3                             # reading is 3 steps old, window holds only 2
    obs = env.build_obs()
    assert np.allclose(obs[:16], env.state_hist[1])        # stale state s_1
    w = window(env)
    assert w.shape == (2, K)
    assert np.allclose(w[0], onehot(7)) and np.allclose(w[1], onehot(9))  # a_3, a_4; a_2 (=5) dropped
    assert obs.shape == env.ENVIRONMENT_SHAPE


def test_fresher_than_trained_pads_with_zeros():
    env = make_env("delayed_history", 3)
    play(env, [3, 5, 7, 9])
    env.OBS_STALE = 1                             # reading is only 1 step old
    obs = env.build_obs()
    assert np.allclose(obs[:16], env.state_hist[3])        # stale state s_3
    w = window(env)
    assert w.shape == (3, K)
    assert np.allclose(w[0], 0) and np.allclose(w[1], 0) and np.allclose(w[2], onehot(9))


def test_zero_staleness_has_empty_window():
    env = make_env("delayed_history", 2)
    play(env, [3, 5])
    env.OBS_STALE = 0
    obs = env.build_obs()
    assert np.allclose(obs[:16], np.array(env.state_vector).ravel())
    assert np.allclose(window(env), 0)


def test_decoy_window_has_one_random_drug_per_real_slot():
    env = make_env("delayed_decoy", 3)
    play(env, [3, 5, 7])
    env.OBS_STALE = 2
    w = window(env)
    assert w.sum() == 2 and (w.sum(axis=1) <= 1).all()


def test_state_mode_shape_is_unchanged_by_staleness():
    env = make_env("delayed_state", 2)
    play(env, [3, 5, 7])
    for k in range(5):
        env.OBS_STALE = k
        assert env.build_obs().shape == (16,)


@pytest.fixture(scope="module")
def trained():
    return evol_deepmind(episodes=3, N=N, num_drugs=K, mira=True,
                         train_input="delayed_history", obs_delay=2, train_freq=1)


def test_greedy_eval_runs_and_restores_agent(trained):
    agent = trained[2]
    n_before, eps_before, mem_before = agent.hp.EPISODES, agent.hp.epsilon, len(agent.master_memory)
    fit = greedy_eval(agent, stale=3, episodes=2)
    assert fit.shape == (2,) and np.isfinite(fit).all() and (fit >= 0).all()
    assert agent.hp.EPISODES == n_before and agent.hp.epsilon == eps_before
    assert len(agent.master_memory) == mem_before and agent.env.OBS_STALE == 2


def test_greedy_eval_does_not_change_weights(trained):
    agent = trained[2]
    before = [w.copy() for w in agent.model.get_weights()]
    greedy_eval(agent, stale=1, episodes=2)
    assert all(np.array_equal(a, b) for a, b in zip(before, agent.model.get_weights()))


def test_stale_rejected_outside_delayed_modes():
    out = evol_deepmind(episodes=2, N=N, num_drugs=K, mira=True, train_input="state_vector", train_freq=1)
    with pytest.raises(ValueError):
        greedy_eval(out[2], stale=2, episodes=1)


def test_reference_policies_evaluate(trained):
    naive = reference_eval(trained[3], "naive", episodes=2)
    dp = reference_eval(trained[4], "dp", episodes=2)
    assert naive.shape == (2,) and dp.shape == (2,)
    assert dp.mean() <= naive.mean() + 0.3  # exact optimum is not worse than random beyond noise


def test_trace_records_every_step(trained):
    agent = trained[2]
    rows = []
    greedy_eval(agent, stale=2, episodes=2, trace=rows)
    assert len(rows) == 2 * 21                      # the unchosen first step plus 20 chosen steps
    assert all(1 <= r[2] <= K and 0 <= r[3] < 16 for r in rows)
    assert callable(agent.env.step) and agent.env.step.__name__ == "step"  # original step restored
