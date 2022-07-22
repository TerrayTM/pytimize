SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


class SymbolParser:
    @staticmethod
    def subscript(string):
        return string.translate(SUBSCRIPT)
