from app import main


def test_main_output(capsys):
    """Verify that ``main`` prints the welcome messages."""

    main()
    captured = capsys.readouterr()

    assert "Welcome to OmniBeing Project!" in captured.out
    assert "The future is being built here." in captured.out
    assert "Created by behicof" in captured.out
