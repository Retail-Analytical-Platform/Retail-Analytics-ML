import pytest

import main


def test_validate_exits_zero():
    assert main.main(["--stage", "validate", "--sample"]) == 0


def test_invalid_stage_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main.main(["--stage", "nope"])
    assert exc.value.code != 0

 