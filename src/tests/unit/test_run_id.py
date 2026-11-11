from src.run_id import new_run_id


def test_run_ids_are_unique():
    assert new_run_id() != new_run_id()