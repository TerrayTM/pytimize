import re
import numpy as np

class Parser:
  def parse(self, expression: str, size: int = None) -> np.ndarray:
    if self.__validate_expression(expression):
      term = re.compile(r"(\+|-)?(\d+(?:\.\d+)?)x_(\d+)")
      data = {}
      length = 0

      for sign, coefficient, index in term.findall(expression):
        index = int(index) - 1
        
        if index in data.keys():
          raise Exception()
        else:
          data[index] = float(("-" if sign == "-" else "") + coefficient)

          if index > length:
            length = index

      result = np.full((1, length + 1 if size is None else size), 0, np.float)

      for key, value in data.items():
        result[0, key] = value
      
      return result
    else:
      raise Exception()



  def parse_multiple(self, expressions: list, size: int = None):
    for expression in expressions:
      if self.__validate_expression(expression):
        self.compile(expression, size)
      else:
        raise Exception()



  def __validate_expression(self, candidate):
    validation = re.compile(r"^-?\d+(?:\.\d+)?x_\d+(?:[+|-]\d+(?:\.\d+)?x_\d+)*$")

    return validation.match(candidate) is not None
