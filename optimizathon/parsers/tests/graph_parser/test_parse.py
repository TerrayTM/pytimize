import numpy as np

from ... import GraphParser
from unittest import TestCase, main

class TestParse(TestCase):
    def test_parse_type_one(self) -> None:
        result = GraphParser.parse("ab cd ef 12 34 56")
        expected = [("a", "b"), ("c", "d"), ("e", "f"), ("1", "2"), ("3", "4"), ("5", "6")]

        self.assertEqual(result, expected, "Should parse type one correctly.")
    
    def test_parse_type_two(self) -> None:
        result = GraphParser.parse("123-456 abc-5 5-5 a-bc x-yz")
        expected = [("123", "456"), ("abc", "5"), ("5", "5"), ("a", "bc"), ("x", "yz")]

        self.assertEqual(result, expected, "Should parse type two correctly.")

    def test_invalid_format(self) -> None:
        with self.assertRaises(ValueError, msg="Should throw error if invalid format."):
            GraphParser.parse("abc")
        
        with self.assertRaises(ValueError, msg="Should throw error if invalid format."):
            GraphParser.parse("abc=efg h")

        with self.assertRaises(ValueError, msg="Should throw error if invalid format."):
            GraphParser.parse("")
        
        with self.assertRaises(ValueError, msg="Should throw error if invalid format."):
            GraphParser.parse("!@")

if __name__ == "__main__":
    main()
