from aeroguard.data.training_schedule import deterministic_state_mask


def test_state_dropout_schedule_is_deterministic_and_preserves_masked_mode():
    first = [
        deterministic_state_mask(1.0, 0.5, seed=17, schedule_index=index)
        for index in range(100)
    ]
    second = [
        deterministic_state_mask(1.0, 0.5, seed=17, schedule_index=index)
        for index in range(100)
    ]

    assert first == second
    assert set(first) == {0.0, 1.0}
    assert all(
        deterministic_state_mask(0.0, 0.5, seed=17, schedule_index=index) == 0.0
        for index in range(100)
    )


def test_zero_state_dropout_keeps_every_paired_sample_enabled():
    assert all(
        deterministic_state_mask(1.0, 0.0, seed=17, schedule_index=index) == 1.0
        for index in range(100)
    )
