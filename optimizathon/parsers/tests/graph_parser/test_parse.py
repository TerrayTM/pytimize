import numpy as np

from ... import GraphParser
from unittest import TestCase, main

class TestParse(TestCase):
    def test_parse_type_one(self) -> None:
        result = GraphParser.parse("ab cd ef 12 34 56")
        expected = [("a", "b", 0), ("c", "d", 0), ("e", "f", 0), ("1", "2", 0), ("3", "4", 0), ("5", "6", 0)]

        self.assertEqual(result, expected, "Should parse type one correctly.")
    
    def test_parse_type_two(self) -> None:
        result = GraphParser.parse("123-456 abc-5 5-5 a-bc x-yz")
        expected = [("123", "456", 0), ("abc", "5", 0), ("5", "5", 0), ("a", "bc", 0), ("x", "yz", 0)]

        self.assertEqual(result, expected, "Should parse type two correctly.")

    def test_parse_type_three(self) -> None:
        result = GraphParser.parse("ab:1 cd:2.5 ef:0.25 gh:555 ij:25.255")
        expected = [("a", "b", 1), ("c", "d", 2.5), ("e", "f", 0.25), ("g", "h", 555), ("i", "j", 25.255)]

        self.assertEqual(result, expected, "Should parse type two correctly.")

    def test_parse_type_four(self) -> None:
        result = GraphParser.parse("123-456:123 abc-5:2.5 5-5:0.123 a-bc:456.456")
        expected = [("123", "456", 123), ("abc", "5", 2.5), ("5", "5", 0.123), ("a", "bc", 456.456)]

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
