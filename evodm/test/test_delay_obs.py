"""Tests for the delayed-observation modes (delayed_state / delayed_history / delayed_decoy)."""
import numpy as np
import pytest
from evodm import evol_env, define_mira_landscapes, evol_deepmind

N, K = 4, 15  # 16 genotypes, 15 drugs


def make_env(mode, d):
    return evol_env(N=N, num_drugs=K, drugs=define_mira_landscapes(), train_input=mode,
                    obs_delay=d, add_noise=False, normalize_drugs=False)


def onehot(a):
    v = np.zeros(K)
    v[a - 1] = 1.0
    return v


def play(env, actions):
    for a in actions:
        env.action = a
        env.step()


def test_shapes():
    assert make_env("delayed_state", 3).ENVIRONMENT_SHAPE == (16,)
    assert make_env("delayed_history", 3).ENVIRONMENT_SHAPE == (16 + 3 * K,)
    assert make_env("delayed_decoy", 3).ENVIRONMENT_SHAPE == (16 + 3 * K,)
    for mode in ("delayed_state", "delayed_history", "delayed_decoy"):
        env = make_env(mode, 3)
        assert env.cur_obs.shape == env.ENVIRONMENT_SHAPE


def test_zero_delay_is_true_state():
    env = make_env("delayed_state", 0)
    play(env, [1, 5, 7])
    assert np.allclose(env.cur_obs, np.array(env.state_vector).ravel())


def test_stale_state_and_history_values():
    env = make_env("delayed_history", 2)
    play(env, [1, 5, 7])  # t = 3, stale state is s_1, window = a_2, a_3 = 5, 7
    obs = env.cur_obs
    assert np.allclose(obs[:16], env.state_hist[1])
    assert np.allclose(obs[16:16 + K], onehot(5))
    assert np.allclose(obs[16 + K:], onehot(7))


def test_left_padding_early_in_episode():
    env = make_env("delayed_history", 3)
    play(env, [4])  # t = 1 < d, stale state is s_0, window = a_1 only, right-aligned
    obs = env.cur_obs
    assert np.allclose(obs[:16], env.state_hist[0])
    assert np.allclose(obs[16:16 + 2 * K], 0.0)
    assert np.allclose(obs[16 + 2 * K:], onehot(4))


def test_decoy_has_same_size_and_no_true_history():
    np.random.seed(0)
    import random
    random.seed(0)
    env = make_env("delayed_decoy", 2)
    play(env, [1, 5, 7])
    obs = env.cur_obs
    assert obs.shape == (16 + 2 * K,)
    assert np.allclose(obs[:16], env.state_hist[1])
    assert obs[16:].sum() == 2.0  # two one-hot slots, but drugs are random


def test_replay_tuple_alignment_and_reset():
    env = make_env("delayed_history", 2)
    env.action = 3
    prev_obs = env.cur_obs.copy()
    env.step()
    s_old, a, r, s_new = env.sensor
    assert np.allclose(s_old, prev_obs) and a == 3
    assert np.allclose(s_new, env.cur_obs)
    env.reset()
    assert len(env.action_hist) == 0 and len(env.state_hist) == 1
    assert np.allclose(env.cur_obs[:16], np.array(env.state_vector).ravel())


@pytest.mark.parametrize("mode", ["delayed_state", "delayed_history", "delayed_decoy"])
def test_agent_runs_end_to_end(mode):
    out = evol_deepmind(episodes=2, N=N, num_drugs=K, mira=True, train_input=mode,
                        obs_delay=2, train_freq=1)
    assert len(out[2].master_memory) > 0
