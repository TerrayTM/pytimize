import re

from typing import List, Tuple

class GraphParser:
  # Format: ab cd de
  # Format: a-b c-d s-s
  @staticmethod
  def parse(expression: str) -> List[Tuple[str, str]]:
    parse_type = GraphParser.get_expression_type(expression)
    result = []
    term = None

    if parse_type == -1:
      raise ValueError("Invalid expression format.")

    if parse_type == 0:
      term = re.compile(r"(\w)(\w)")
    elif parse_type == 1:
      term = re.compile(r"(\w+)-(\w+)")
    
    for a, b in term.findall(expression):
      result.append((a, b))
    
    return result


  @staticmethod
  def get_expression_type(candidate: str) -> int:
    validation_one = re.compile(r"^(?:\w\w)(?:\s\w\w)*$")
    validation_two = re.compile(r"^(?:\w+-\w+)(?:\s\w+-\w+)*$")
    result = -1

    if validation_one.match(candidate) is not None:
      result = 0
    elif validation_two.match(candidate) is not None:
      result = 1

    return result
