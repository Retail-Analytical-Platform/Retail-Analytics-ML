import pytest

import main


def test_validate_exits_zero():
<<<<<<< HEAD
<<<<<<< HEAD
    assert main.main(["--stage", "validate", "--sample"]) == 0
=======
    assert main.main(["--stage", "validate"]) == 0
>>>>>>> 0e6b1b4 (Add CI workflow, configuration files, and initial tests)
=======
    assert main.main(["--stage", "validate", "--sample"]) == 0
>>>>>>> 1fb9dd0 (Implement data ingestion functions and enhance validation with sample option)


def test_invalid_stage_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main.main(["--stage", "nope"])
<<<<<<< HEAD
<<<<<<< HEAD
    assert exc.value.code != 0

 
=======
    assert exc.value.code != 0
>>>>>>> 0e6b1b4 (Add CI workflow, configuration files, and initial tests)
=======
    assert exc.value.code != 0

 
>>>>>>> 1fb9dd0 (Implement data ingestion functions and enhance validation with sample option)
