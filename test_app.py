from app import main


def test_main_output(capsys):
    """Verify that ``main`` prints the trading system demo messages."""

    main()
    captured = capsys.readouterr()

    assert "OmniBeing Trading System - Complete Integration Demo" in captured.out
    assert "Created by behicof" in captured.out
    assert "Trading System Demo Complete!" in captured.out
