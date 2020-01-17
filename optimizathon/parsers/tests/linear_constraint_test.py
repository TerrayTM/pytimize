import sys
import numpy as np

sys.path.append("../optimization")
sys.path.append("../optimization/enums")

from linear_constraint import Parser
from unittest import TestCase, main

class TestInit(TestCase):
    def test_parse(self):
        constraint = Parse

le = LinearEquation()
res = le.compile('1.25x_5+0.25x_4+3x_9')
print(res)
'''

1x_5+2x_4+3x_5

2x+6y+4z

'''

if __name__ == "__main__":
    main()
