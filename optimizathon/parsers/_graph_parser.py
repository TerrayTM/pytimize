import re

from typing import List, Tuple

class GraphParser:
  # Format: ab cd de
  # Format: ab:5 sd:12.12 sjf:45
  # Format: a-b:465 b-c:485
  # Format: a-b c-d s-s
  @staticmethod
  def parse(expression: str) -> List[Tuple[str, str, float]]:
    parse_type = GraphParser.get_expression_type(expression)
    result = []
    term = None

    if parse_type == -1:
      raise ValueError("Invalid expression format.")

    if parse_type == 0:
      term = re.compile(r"(\w)(\w)")
    elif parse_type == 1:
      term = re.compile(r"(\w+)-(\w+)")
    elif parse_type == 2:
      term = re.compile(r"(\w)(\w):(\d+(?:\.\d+)?)")
    elif parse_type == 3:
      term = re.compile(r"(\w+)-(\w+):(\d+(?:\.\d+)?)")
    
    for info in term.findall(expression):
      result.append((info[0], info[1], float(info[2]) if parse_type > 1 else 0))
    
    return result



  @staticmethod
  def get_expression_type(candidate: str) -> int:
    validation_one = re.compile(r"^(?:\w\w)(?:\s\w\w)*$")
    validation_two = re.compile(r"^(?:\w+-\w+)(?:\s\w+-\w+)*$")
    validation_three = re.compile(r"^(?:\w\w:\d+(?:\.\d+)?)(?:\s\w\w:\d+(?:\.\d+)?)*$")
    validation_four = re.compile(r"^(?:\w+-\w+:\d+(?:\.\d+)?)(?:\s\w+-\w+:\d+(?:\.\d+)?)*$")
    result = -1

    if validation_one.match(candidate) is not None:
      result = 0
    elif validation_two.match(candidate) is not None:
      result = 1
    elif validation_three.match(candidate) is not None:
      result = 2
    elif validation_four.match(candidate) is not None:
      result = 3

    return result



  @staticmethod
  def format_graph(graph: List[Tuple[str, str, float]]) -> str:
    pass