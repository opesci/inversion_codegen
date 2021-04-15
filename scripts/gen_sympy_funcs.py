from sympy.functions import __all__ as funcs

print("import sympy\n")
print("from devito.finite_differences.differentiable\
 import (DifferentiableOp, diffify)\n\n")

for fn in funcs:
    try:
        strc = """class %s(DifferentiableOp, sympy.%s):
    __sympy_class__ = sympy.%s
    __new__ = DifferentiableOp.__new__\n\n""" % (fn, fn, fn)
        exec(strc)
        print(strc)
    except:
        # Some are not classes such as sqrt
        print("""def %s(x):
    diffify(sympy.%s(x))\n\n""" % (fn, fn))
