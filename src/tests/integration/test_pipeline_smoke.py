import main


def test_all_stages_with_sample_exits_zero():
    assert main.main(["--stage", "all", "--sample"]) == 0