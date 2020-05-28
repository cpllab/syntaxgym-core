from pyparsing import *
import numpy as np



# Relative and absolute tolerance thresholds for surprisal equality
EQUALITY_RTOL = 1e-5
EQUALITY_ATOL = 1e-3


class Prediction(object):

    #######
    # Define a grammar for prediction formulae.

    # References a surprisal region
    lpar = Suppress("(")
    rpar = Suppress(")")
    region = lpar + Word(nums) + Suppress(";%") + Word(alphanums + "_-") + Suppress("%") + rpar
    literal_float = pyparsing_common.number

    class Region(object):
        def __init__(self, tokens):
            self.region_number = tokens[0]
            self.condition_name = tokens[1]

        def __str__(self):
            return "(%s;%%%s%%)" % (self.region_number, self.condition_name)

        def __repr__(self):
            return "Region(%s,%s)" % (self.condition_name, self.region_number)

        def __call__(self, surprisal_dict):
            if self.region_number == "*":
                return sum(value for (condition, region), value in surprisal_dict.items()
                           if condition == self.condition_name)

            return surprisal_dict[self.condition_name, int(self.region_number)]

    class LiteralFloat(object):
        def __init__(self, tokens):
            self.value = float(tokens[0])

        def __str__(self):
            return "%f" % (self.value,)

        def __repr__(self):
            return "LiteralFloat(%f)" % (self.value,)

        def __call__(self, surprisal_dict):
            return self.value

    class BinaryOp(object):
        operators = None
        def __init__(self, tokens):
            self.operator = tokens[0][1]
            if self.operators is not None and self.operator not in self.operators:
                raise ValueError("Invalid %s operator %s" % (self.__class__.__name__,
                                                             self.operator))
            self.operands = [tokens[0][0], tokens[0][2]]

        def __str__(self):
            return "%s %s %s" % (self.operands[0], self.operator, self.operands[1])

        def __repr__(self):
            return "%s(%s)(%s)" % (self.__class__.__name__, self.operator, ",".join(map(repr, self.operands)))

        def __call__(self, surprisal_dict):
            op_vals = [op(surprisal_dict) for op in self.operands]
            return self._evaluate(op_vals, surprisal_dict)

        def _evaluate(self, evaluated_operands, surprisal_dict):
            raise NotImplementedError()

    class BoolOp(BinaryOp):
        operators = ["&", "|"]
        def _evaluate(self, op_vals, surprisal_dict):
            if self.operator == "&":
                return op_vals[0] and op_vals[1]
            elif self.operator == "|":
                return op_vals[0] or op_vals[1]

    class FloatOp(BinaryOp):
        operators = ["-", "+"]
        def _evaluate(self, op_vals, surprisal_dict):
            if self.operator == "-":
                return op_vals[0] - op_vals[1]
            elif self.operator == "+":
                return op_vals[0] + op_vals[1]

    class ComparatorOp(BinaryOp):
        operators = ["<", ">", "="]
        def _evaluate(self, op_vals, surprisal_dict):
            if self.operator == "<":
                return op_vals[0] < op_vals[1]
            elif self.operator == ">":
                return op_vals[0] > op_vals[1]
            elif self.operator == "=":
                return np.isclose(op_vals[0], op_vals[1],
                                  rtol=EQUALITY_RTOL,
                                  atol=EQUALITY_ATOL)

    atom = region.setParseAction(Region) | literal_float.setParseAction(LiteralFloat)

    stack = []
    prediction_expr = infixNotation(
        atom,
        [
            ("+", 2, opAssoc.LEFT, FloatOp),
            ("-", 2, opAssoc.LEFT, FloatOp),
            ("<", 2, opAssoc.LEFT, ComparatorOp),
            (">", 2, opAssoc.LEFT, ComparatorOp),
            ("=", 2, opAssoc.LEFT, ComparatorOp),
            ("&", 2, opAssoc.LEFT, BoolOp),
        ],
        lpar=lpar, rpar=rpar
    )


    def __init__(self, idx, formula):
        if isinstance(formula, str):
            try:
                formula = self.prediction_expr.parseString(formula, parseAll=True)[0]
            except ParseException as e:
                raise ValueError("Invalid formula expression %r" % (formula,)) from e

        self.idx = idx
        self.formula = formula

    def __call__(self, item):
        """
        Evaluate the prediction on the given item.
        """
        # Prepare relevant surprisal dict
        surps = {(c["condition_name"], r["region_number"]): r["metric_value"]["sum"]
                 for c in item["conditions"]
                 for r in c["regions"]}
        return self.formula(surps)

    @classmethod
    def from_dict(cls, pred_dict, idx):
        if not pred_dict["type"] == "formula":
            raise ValueError("Unknown prediction type %s" % (pred_dict["type"],))

        return cls(formula=pred_dict["formula"], idx=idx)

    def as_dict(self):
        return dict(type="formula", formula=str(self.formula))

    def __str__(self):
        return "Prediction(%s)" % (self.formula,)
    __repr__ = __str__

    def __hash__(self):
        return hash(self.formula)

    def __eq__(self, other):
        return isinstance(other, Prediction) and hash(self) == hash(other)

