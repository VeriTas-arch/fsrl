import unittest
from unittest.mock import Mock

from fsrl.infra.validation_session import reuse_validation, validation_session


class ValidationSessionTests(unittest.TestCase):
    def test_reuses_pure_result_only_inside_one_session(self):
        operation = Mock(side_effect=lambda value: {"value": value})
        validate = reuse_validation(operation)

        outside_first = validate(3)
        outside_second = validate(3)
        self.assertEqual(operation.call_count, 2)
        self.assertIsNot(outside_first, outside_second)

        with validation_session():
            inside_first = validate(3)
            inside_second = validate(3)
            other_argument = validate(4)
        self.assertEqual(operation.call_count, 4)
        self.assertIs(inside_first, inside_second)
        self.assertEqual(other_argument, {"value": 4})

        validate(3)
        self.assertEqual(operation.call_count, 5)

    def test_nested_session_uses_outer_cache_and_does_not_cache_failures(self):
        operation = Mock(side_effect=[RuntimeError("invalid"), "valid"])
        validate = reuse_validation(operation)

        with validation_session():
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                validate()
            with validation_session():
                self.assertEqual(validate(), "valid")
                self.assertEqual(validate(), "valid")
        self.assertEqual(operation.call_count, 2)
