from ... import typecheck
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

        self.assertTrue(method_one((5, 10)), "Should check tuple types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one(8)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one(())
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one(("a", 5))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one((1, 2, 3))

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one((2.5, 2))
        
        @typecheck
        def method_two(a: Tuple[int, str]) -> bool:
            return True

        self.assertTrue(method_two((5, "a")), "Should check tuple types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(8)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(())
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(("a", 5))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two((1, 2, 3, 4, 5))

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two((2.5, 2.5))

        @typecheck
        def method_three(a: Tuple[str, float]) -> bool:
            return True

        self.assertTrue(method_three(("a", 2.5)), "Should check tuple types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three(8)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three(())
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three(("a", "b"))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three((1, 2, 3, 4, 5))

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three((2.5, 2.5))

        @typecheck
        def method_four(a: Tuple[str, bool, int, float]) -> bool:
            return True

        self.assertTrue(method_four(("a", True, 5, 2.5)), "Should check tuple types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four(8)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four(("a", True, 5))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four(("a", "b"))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four(("a", True, 5, 5, 6))

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_four((2.5, 2.5))

        @typecheck
        def method_five(a: Tuple[str, str, int, int]) -> bool:
            return True

        self.assertTrue(method_five(("a", "a", 5, 5)), "Should check tuple types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_five(8)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_five(("a", True, 5))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_five(("a", "b"))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_five(("a", "a", 5, 5, 6))

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_five((2.5, 2.5))
        
    def test_check_nested_types(self) -> None:
        @typecheck
        def method_one(a: List[List[Set[str]]]) -> bool:
            return True

        self.assertTrue(method_one([[{"a", "b", "c"}]]), "Should check nested types.")
        self.assertTrue(method_one([]), "Should check nested types.")
        self.assertTrue(method_one([[]]), "Should check nested types.")
        self.assertTrue(method_one([[set()]]), "Should check nested types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one([[5]])
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one(5)

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one([5])
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one([[{"a", "b", 2}]])

        @typecheck
        def method_two(a: Tuple[List[Tuple[int, float]], Tuple[str, Set[bool]]]) -> bool:
            return True

        self.assertTrue(method_two(([(5, 2.5)], ("a", {True}))), "Should check nested types.")
        self.assertTrue(method_two(([(5, 6)], ("a", set()))), "Should check nested types.")
        self.assertTrue(method_two(([], ("a", set()))), "Should check nested types.")
        self.assertTrue(method_two(([(1, 2), (5, 6), (5, 6)], ("a", set()))), "Should check nested types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(())
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(([5], (5, set())))

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(([("a", 7)], ("a", set())))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(([(5, 7)], ("a", (2, 2))))

        @typecheck
        def method_three(a: List[List[List[Tuple[int, int]]]]) -> bool:
            return True

        self.assertTrue(method_three([[[(1, 2), (1, 2)], [(1, 2), (6, 2)]], [[(1, 2), (1, 2)], [(1, 2), (6, 2)]]]), "Should check nested types.")
        self.assertTrue(method_three([]), "Should check nested types.")
        self.assertTrue(method_three([[]]), "Should check nested types.")
        self.assertTrue(method_three([[[]]]), "Should check nested types.")
        self.assertTrue(method_three([[[(1, 2)]]]), "Should check nested types.")
        self.assertTrue(method_three([[[(1, 2)], [(1, 2)]], [[(1, 2)], [(1, 2)]]]), "Should check nested types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two(())

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two([5])

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two([[5]])
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two([[[5]]])

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two([[[5]]])

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two([[[(1, 5), (2, 3), ("a", 6)]]])

        @typecheck
        def method_four(a: Tuple[List[str], str, Set[str], Tuple[int, int]]) -> bool:
            return True

        self.assertTrue(method_four((["a", "a", "b"], "a", {"a", "b"}, (8, 9))), "Should check nested types.")
        self.assertTrue(method_four(([], "b", set(), (0, 0))), "Should check nested types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two((["w"], 5, {"b"}, (1, 2)))
        
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two((["w"], "a", {"b", "6", 8}, (1, 2)))

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two((["w", "a", "v"], "a", {"b"}, (2.5, 2)))

    def test_check_empty(self) -> None:
        @typecheck
        def method() -> bool:
            return True
        
        self.assertTrue(method(), "Should not crash for methods without parameters.")

    def test_incomplete_annotation(self) -> None:
        @typecheck
        def method_tuple(a: Tuple) -> bool:
            return True
            
        with self.assertRaises(NotImplementedError, msg="Should throw error if type annotation is incomplete."):
            method_tuple(())

    def test_check_generic_type(self) -> None:
        @typecheck
        def method_list(a: List) -> bool:
            return True

        self.assertTrue(method_list(["a", 5, 2.5, ()]), "Should check generic types.")

        @typecheck
        def method_set(a: Set) -> bool:
            return True

        self.assertTrue(method_set({1, 2, "a"}), "Should check generic types.")

        @typecheck
        def method_union(a: Union) -> bool:
            return True

        self.assertTrue(method_union([]), "Should check generic types.")

        @typecheck
        def method_mixed(a: List, b: int) -> bool:
            return True

        self.assertTrue(method_mixed([1, 2, "a", set()], 6), "Should check generic types.")

    import unittest
    @unittest.skip("Work in progress...")
    def test_check_union(self) -> None:
        @typecheck
        def method_one(a: Union[str, int]) -> bool:
            return True

        self.assertTrue(method_one("a"), "Should check union types.")
        self.assertTrue(method_one(5), "Should check union types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one([])

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_one(2.5)

        @typecheck
        def method_two(a: Union[str, Union[int, str, float]]) -> bool:
            return True

        self.assertTrue(method_two(8), "Should check union types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_two([])
        
        @typecheck
        def method_three(a: Tuple[Union[str, Tuple[int, Union[str, float]]], Union[int, Tuple[Union[int, str], Union[List[int], Set[int]]]]]) -> bool:
            return True

        self.assertTrue(method_three(("a", 5)), "Should check union types.")
        self.assertTrue(method_three(((5, 5), 5)), "Should check union types.")
        self.assertTrue(method_three(((5, "a"), 5)), "Should check union types.")
        self.assertTrue(method_three(("a", (5, []))), "Should check union types.")
        self.assertTrue(method_three((("a", "w"), 5)), "Should check union types.")
        self.assertTrue(method_three(("a", (5, [2, 5, 10]))), "Should check union types.")
        self.assertTrue(method_three(("a", ("a", {2, 5, 10}))), "Should check union types.")
        self.assertTrue(method_three((("a", 7), ("a", {2, 5, 10}))), "Should check union types.")
        self.assertTrue(method_three((("a", "w"), ("a", {2, 5, 10}))), "Should check union types.")
        self.assertTrue(method_three((("a", "w"), (5, set()))), "Should check union types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method_three((7, 7))

        #TODO FINISH TEST CASES
        # @typecheck
        # def method_three(a: Union[str, List[Union[str, Tuple[Union[str, int], int]]]]) -> bool:
        #     return True

        # self.assertTrue(method_three(8), "Should check union types.")

        # with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
        #     method_three([])

    def test_check_class_method(self) -> None:
        class Test:
            @typecheck
            def method(self) -> bool:
                return True

        test = Test()

        self.assertTrue(test.method(), "Should not crash on class method.")

    def test_check_alias(self) -> None:
        A = List[float]
        B = int
        C = Union[List[str], Tuple[int, int]]

        @typecheck
        def method(a: A, b: B, c: C) -> bool:
            return True

        self.assertTrue(method([1, 2, 3], 5, ["a"]), "Should check alias types.")
        self.assertTrue(method([0.5], 1, (1, 2)), "Should check alias types.")

        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method([1, 2, 3], "a", [])
    
        with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
            method([], "b", (1, 2, 3))

    def test_no_annotation(self) -> None:
        @typecheck
        def method(a) -> bool:
            return True

        with self.assertRaises(NotImplementedError, msg="Should throw error if parameter is not annotated."):
            method(5)

if __name__ == "__main__":
    main()
