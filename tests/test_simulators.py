import numpy as np

from app.simulators import linearly_decaying_epsilon


def test_decaying_epsilon():
    steps = 10
    epsilon_min = 0.1
    result = [
        linearly_decaying_epsilon(
            decay_period=10,
            step=s,
            warmup_steps=0,
            epsilon_min=epsilon_min,
        ) for s in range(steps)
    ]

    expected = np.flip(np.linspace(epsilon_min, 1, num=10))
    assert np.isclose(result, expected, atol=0.1, rtol=0).all()


def test_decaying_lower_bound():
    lower_bound = 0.4
    result = linearly_decaying_epsilon(
        decay_period=10,
        step=20,
        warmup_steps=0,
        epsilon_min=lower_bound,
    )
    assert result == lower_bound

    # test last decaying step
    result = linearly_decaying_epsilon(
        decay_period=10,
        step=10,
        warmup_steps=0,
        epsilon_min=lower_bound,
    )
    assert result == lower_bound


def test_decaying_warmup():
    warmup_steps = 10
    decay_period = 20  # any value greater than warmup
    epsilon_min = 0.1
    for s in range(warmup_steps):
        eps = linearly_decaying_epsilon(
            decay_period=decay_period,
            step=s,
            warmup_steps=warmup_steps,
            epsilon_min=epsilon_min,
        )
        assert eps == 1

    # assert decaying after warmup steps
    eps = linearly_decaying_epsilon(
        decay_period=decay_period,
        step=warmup_steps+1,
        warmup_steps=warmup_steps,
        epsilon_min=epsilon_min,
    )
    assert eps != 1
