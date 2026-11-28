"""The optional convolutional front end for the delayed-observation modes."""
import numpy as np
import pytest
from evodm import evol_deepmind


@pytest.mark.parametrize("mode,width", [("delayed_state", 16), ("delayed_history", 16 + 2 * 15)])
def test_conv_network_builds_and_trains(mode, width):
    res = evol_deepmind(episodes=3, N=4, num_drugs=15, mira=True, train_input=mode, obs_delay=2,
                        train_freq=1, network="conv")
    agent = res[2]
    assert agent.model.layers[0].__class__.__name__ == "Reshape"
    assert agent.model.input_shape == (None, width)
    assert agent.model.output_shape == (None, 15)
    assert len(agent.get_qs()) == 15


def test_default_network_is_dense():
    agent = evol_deepmind(episodes=2, N=4, num_drugs=15, mira=True, train_input="delayed_history",
                          obs_delay=2, train_freq=1)[2]
    assert agent.model.layers[0].__class__.__name__ == "Dense"
