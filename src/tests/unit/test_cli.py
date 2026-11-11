import pytest

import main


def test_validate_exits_zero():
<<<<<<< HEAD
    assert main.main(["--stage", "validate", "--sample"]) == 0
=======
    assert main.main(["--stage", "validate"]) == 0
>>>>>>> 0e6b1b4 (Add CI workflow, configuration files, and initial tests)


def test_invalid_stage_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main.main(["--stage", "nope"])
<<<<<<< HEAD
    assert exc.value.code != 0

 
=======
    assert exc.value.code != 0
>>>>>>> 0e6b1b4 (Add CI workflow, configuration files, and initial tests)
