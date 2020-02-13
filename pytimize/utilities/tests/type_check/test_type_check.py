from ..._type_check import typecheck
from unittest import TestCase, main
from typing import List, Set, Tuple

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

        # with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
        #     method_one(True, 5)

        # with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
        #     method_one([1, 2, "a"], {1, 2, 3})

        # with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
        #     method_one([1, 2, 6], {1, False, 3})
        
        # with self.assertRaises(TypeError, msg="Should throw error if invalid type."):
        #     method_one([1, True, "w"], {2.7, False, 3})

    def test_check_tuple_types(self) -> None:
        pass


    # def test_check_compound_types(self) -> None:
    #     @typecheck
    #     def method(a, b, c):
    #         return 1

    def test_check_empty(self) -> None:
        @typecheck
        def method() -> None:
            return True
        
        self.assertTrue(method(), "Should not crash for functions without parameters.")

    def test_no_annotation(self) -> None:
        pass

if __name__ == "__main__":
    main()
