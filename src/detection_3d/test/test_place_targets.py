import pytest
from detection_3d.place_targets import validate_place_targets, get_place_target


class TestValidatePlaceTargets:

    def test_empty_list_returns_empty(self):
        result = validate_place_targets([])
        assert result == []

    def test_single_target(self):
        result = validate_place_targets([1.0, 2.0, 3.0])
        assert result == [(1.0, 2.0, 3.0)]

    def test_multiple_targets(self):
        result = validate_place_targets([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert result == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

    def test_length_not_multiple_of_3_raises(self):
        with pytest.raises(ValueError, match="multiple of 3"):
            validate_place_targets([1.0, 2.0])

    def test_length_4_raises(self):
        with pytest.raises(ValueError, match="multiple of 3"):
            validate_place_targets([1.0, 2.0, 3.0, 4.0])

    def test_integers_converted_to_floats(self):
        result = validate_place_targets([1, 2, 3])
        assert result == [(1.0, 2.0, 3.0)]


class TestGetPlaceTarget:

    def test_index_0_returns_first(self):
        targets = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        assert get_place_target(targets, 0) == (1.0, 2.0, 3.0)

    def test_index_1_returns_second(self):
        targets = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        assert get_place_target(targets, 1) == (4.0, 5.0, 6.0)

    def test_empty_raises(self):
        with pytest.raises(IndexError, match="empty"):
            get_place_target([], 0)

    def test_negative_index_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            get_place_target([(1.0, 2.0, 3.0)], -1)

    def test_index_too_large_raises(self):
        with pytest.raises(IndexError, match="out of range"):
            get_place_target([(1.0, 2.0, 3.0)], 1)
