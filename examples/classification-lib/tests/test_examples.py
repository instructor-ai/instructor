from schema import Examples


def test_examples_creation():
    """Test that examples can be created successfully."""
    examples = Examples(
        examples_positive=["positive1", "positive2"],
        examples_negative=["negative1", "negative2"],
    )
    assert len(examples.examples_positive) == 2
    assert len(examples.examples_negative) == 2
    assert "positive1" in examples.examples_positive
    assert "negative1" in examples.examples_negative


def test_empty_examples():
    """Test that examples can be created with empty lists."""
    examples = Examples(examples_positive=[], examples_negative=[])
    assert len(examples.examples_positive) == 0
    assert len(examples.examples_negative) == 0


def test_examples_with_duplicates():
    """Test that examples can contain duplicate entries."""
    examples = Examples(
        examples_positive=["duplicate", "duplicate"], examples_negative=["negative"]
    )
    assert len(examples.examples_positive) == 2
    # Both duplicates are preserved
    assert examples.examples_positive.count("duplicate") == 2
