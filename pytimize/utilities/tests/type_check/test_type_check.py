from ..._type_check import typecheck
from unittest import TestCase, main
from typing import List, Set, Tuple, Union

class TestTypeCheck(TestCase):
    def test_check_primitive_types(self) -> None:
        @typecheck
        def method(a: int, b: str, c: float, d: bool) -> bool:
            return True

        self.assertTrue(method(1, "a", 2.5, True), "Should check primitive types.")
        self.assertTrue(method(5, "b", 3, False), "Should check primitive types.")
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method(1, True, 2.5, True)
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method(1, "b", "w", True)
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method(1, "b", 2.5, 5)
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method("w", 15, False, True)
    
    def test_check_collection_types(self) -> None:
        @typecheck
        def method_one(a: List[int], b: Set[int]) -> bool:
            return True

        self.assertTrue(method_one([1, 2, 3], {1, 2, 3}), "Should check collection types.")
        self.assertTrue(method_one([], set()), "Should check collection types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one(True, 5)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one([1, 2, "a"], {1, 2, 3})

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one([1, 2, 6], {1, 2.5, 3})
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one([1, True, "w"], {2.7, False, 3})

        @typecheck
        def method_two(a: List[str], b: Set[str]) -> bool:
            return True

        self.assertTrue(method_two(["a", "b"], {"a", "b"}), "Should check collection types.")
        self.assertTrue(method_two([], set()), "Should check collection types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(None, None)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(["a", "b", None], {"a", "b"})

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(["e", "d", "c"], {"b", False, "a"})
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(["w", True, "w"], {None, False, 3})

        @typecheck
        def method_three(a: List[bool], b: Set[bool]) -> bool:
            return True

        self.assertTrue(method_three([True, False], {True}), "Should check collection types.")
        self.assertTrue(method_three([], set()), "Should check collection types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three(None, 5)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three([False, False, 5], {True, False})

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three([True], {"b", False, "a"})
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three(["w", True, "w"], {None, False, 3})

        @typecheck
        def method_four(a: List[float], b: Set[float]) -> bool:
            return True

        self.assertTrue(method_four([5, 5.35], {True, 3.5}), "Should check collection types.")
        self.assertTrue(method_four([], set()), "Should check collection types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four(5, 2.5)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four([1, None], {1, 2})

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four([5, 6, 7], {5, None})
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four([5, 7, "a"], {None})

    def test_check_tuple_types(self) -> None:
        @typecheck
        def method_one(a: Tuple[int, int]) -> bool:
            return True
        
        @typecheck
        def method_two(a: Tuple[int, str]) -> bool:
            return True

        @typecheck
        def method_three(a: Tuple[str, float]) -> bool:
            return True

        @typecheck
        def method_four(a: Tuple[str, bool]) -> bool:
            return True

        @typecheck
        def method_five(a: Tuple[str, bool, int, float]) -> bool:
            return True

        @typecheck
        def method_six(a: Tuple[str, str, int, int]) -> bool:
            return True
        
        @typecheck
        def method_seven(a: Tuple[int]) -> bool:
            return True

    def test_check_compound_types(self) -> None:
        @typecheck
        def method_one(a: List[List[Set[str]]]) -> bool:
            return True
        
        @typecheck
        def method_two(a: Tuple[List[Tuple[int, float]], Tuple[str, Set[bool]]]) -> bool:
            return True

        @typecheck
        def method_three(a: List[Set[List[Tuple[int, int]]]]) -> bool:
            return True

        @typecheck
        def method_four(a: Tuple[List[str], str, Set[str], Tuple[int, int]]) -> bool:
            return True

    def test_check_empty(self) -> None:
        @typecheck
        def method() -> bool:
            return True
        
        self.assertTrue(method(), "Should not crash for methods without parameters.")
    
    def test_incomplete_annotation(self) -> None:
        @typecheck
        def method_tuple(a: Tuple) -> bool:
            return True

    def test_check_generic_type(self) -> None:
        @typecheck
        def method_list(a: List) -> bool:
            return True

        @typecheck
        def method_set(a: Set) -> bool:
            return True

        @typecheck
        def method_union(a: Union) -> bool:
            return True

        @typecheck
        def method_mixed(a: List, b: int) -> bool:
            return True

    def test_no_annotation(self) -> None:
        @typecheck
        def method(a) -> bool:
            return True

        with self.assertRaises(NotImplementedError, msg="Should throw error if parameter is not annotated."):
            method(5)

if __name__ == "__main__":
    main()
