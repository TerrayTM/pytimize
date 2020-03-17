import numpy as np

from math import isclose
from typing import Union

class Comparator:
    @staticmethod        
    def is_close_to_zero(value: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Checks if the given value is close to zero. Use this function over 
        `math.isclose` for comparing to 0.

        Parameters
        ----------
        value : Union[float, np.ndarray]
            The value to be tested.

        Returns
        -------
        result : Union[bool, np.ndarray]
            Whether or not the value is close to zero. Returns an array of boolean
            values if an array is passed in as `value`.
        
        """
        return abs(value) < 1.0e-10

    @staticmethod
    def is_close_compare(value: Union[float, np.ndarray], comparison: str, test: float) -> bool:
        """
        Compares two numbers or an array with a number. Takes into account 
        of floating point rounding error. If an array is passed as `value`, 
        elementwise comparision is performed.
        
        Parameters
        ----------
        value : Union[float, np.ndarray]
            The left side value in the comparision.

        comparison : str
            The comparing operator. Must be either `<=`, `<`, `>` or `>=`.

        test : float
            The right side value in the comparision.

        Returns
        -------
        result : bool
            The result of the comparision.
        
        """
        comparator = Comparator.is_close_to_zero if test == 0 else lambda x: np.allclose(x, test)
        result = None

        if comparison == ">=":            
            result = np.logical_or(comparator(value), value >= test)
        elif comparison == ">":
            result = np.logical_and(np.logical_not(comparator(value)), value > test)
        elif comparison == "<=":
            result = np.logical_or(comparator(value), value <= test)
        elif comparison == "<":
            result = np.logical_and(np.logical_not(comparator(value)), value < test)
        else:
            raise ValueError("Invalid comparison operator.")

        return result.all() if isinstance(result, np.ndarray) else result

    @staticmethod
    def is_integer(value: Union[float, np.ndarray]) -> bool:
        """
        Checks if the given number or array is integral. Takes into account of
        floating point rounding error.
        
        Parameters
        ----------
        value : Union[float, np.ndarray]
            The value to be tested.

        Returns
        -------
        result : bool
            Whether or not the value is an integer.
        
        """
        return np.allclose(value, np.round(value))

    @staticmethod
    def is_negative(value: Union[float, np.ndarray]) -> bool:
        """
        Checks if the given number or array is negative. Takes into account of
        floating point rounding error.
        
        Parameters
        ----------
        value : Union[float, np.ndarray]
            The value to be tested.

        Returns
        -------
        result : bool
            Whether or not the value is negative.
        
        """
        return Comparator.is_close_compare(value, "<", 0)
    
    @staticmethod
    def is_positive(value: Union[float, np.ndarray]) -> bool:
        """
        Checks if the given number or array is positive. Takes into account of
        floating point rounding error.
        
        Parameters
        ----------
        value : Union[float, np.ndarray]
            The value to be tested.

        Returns
        -------
        result : bool
            Whether or not the value is positive.
        
        """
        return Comparator.is_close_compare(value, ">", 0)
