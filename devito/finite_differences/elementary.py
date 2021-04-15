import sympy

from devito.finite_differences.differentiable import (DifferentiableFunction, diffify)


class factorial(DifferentiableFunction, sympy.factorial):
    __sympy_class__ = sympy.factorial
    __new__ = DifferentiableFunction.__new__


class factorial2(DifferentiableFunction, sympy.factorial2):
    __sympy_class__ = sympy.factorial2
    __new__ = DifferentiableFunction.__new__


class rf(DifferentiableFunction, sympy.rf):
    __sympy_class__ = sympy.rf
    __new__ = DifferentiableFunction.__new__


class ff(DifferentiableFunction, sympy.ff):
    __sympy_class__ = sympy.ff
    __new__ = DifferentiableFunction.__new__


class binomial(DifferentiableFunction, sympy.binomial):
    __sympy_class__ = sympy.binomial
    __new__ = DifferentiableFunction.__new__


class RisingFactorial(DifferentiableFunction, sympy.RisingFactorial):
    __sympy_class__ = sympy.RisingFactorial
    __new__ = DifferentiableFunction.__new__


class FallingFactorial(DifferentiableFunction, sympy.FallingFactorial):
    __sympy_class__ = sympy.FallingFactorial
    __new__ = DifferentiableFunction.__new__


class subfactorial(DifferentiableFunction, sympy.subfactorial):
    __sympy_class__ = sympy.subfactorial
    __new__ = DifferentiableFunction.__new__


class carmichael(DifferentiableFunction, sympy.carmichael):
    __sympy_class__ = sympy.carmichael
    __new__ = DifferentiableFunction.__new__


class fibonacci(DifferentiableFunction, sympy.fibonacci):
    __sympy_class__ = sympy.fibonacci
    __new__ = DifferentiableFunction.__new__


class lucas(DifferentiableFunction, sympy.lucas):
    __sympy_class__ = sympy.lucas
    __new__ = DifferentiableFunction.__new__


class motzkin(DifferentiableFunction, sympy.motzkin):
    __sympy_class__ = sympy.motzkin
    __new__ = DifferentiableFunction.__new__


class tribonacci(DifferentiableFunction, sympy.tribonacci):
    __sympy_class__ = sympy.tribonacci
    __new__ = DifferentiableFunction.__new__


class harmonic(DifferentiableFunction, sympy.harmonic):
    __sympy_class__ = sympy.harmonic
    __new__ = DifferentiableFunction.__new__


class bernoulli(DifferentiableFunction, sympy.bernoulli):
    __sympy_class__ = sympy.bernoulli
    __new__ = DifferentiableFunction.__new__


class bell(DifferentiableFunction, sympy.bell):
    __sympy_class__ = sympy.bell
    __new__ = DifferentiableFunction.__new__


class euler(DifferentiableFunction, sympy.euler):
    __sympy_class__ = sympy.euler
    __new__ = DifferentiableFunction.__new__


class catalan(DifferentiableFunction, sympy.catalan):
    __sympy_class__ = sympy.catalan
    __new__ = DifferentiableFunction.__new__


class genocchi(DifferentiableFunction, sympy.genocchi):
    __sympy_class__ = sympy.genocchi
    __new__ = DifferentiableFunction.__new__


class partition(DifferentiableFunction, sympy.partition):
    __sympy_class__ = sympy.partition
    __new__ = DifferentiableFunction.__new__


def sqrt(x):
    return diffify(sympy.sqrt(x))


def root(x):
    return diffify(sympy.root(x))


class Min(sympy.Min):

    @property
    def evaluate(self):
        return self.func(*[getattr(a, 'evaluate', a) for a in self.args])


class Max(sympy.Max):

    @property
    def evaluate(self):
        return self.func(*[getattr(a, 'evaluate', a) for a in self.args])


def Id(x):
    return diffify(sympy.Id(x))


def real_root(x):
    return diffify(sympy.real_root(x))


def cbrt(x):
    return diffify(sympy.cbrt(x))


class re(DifferentiableFunction, sympy.re):
    __sympy_class__ = sympy.re
    __new__ = DifferentiableFunction.__new__


class im(DifferentiableFunction, sympy.im):
    __sympy_class__ = sympy.im
    __new__ = DifferentiableFunction.__new__


class sign(DifferentiableFunction, sympy.sign):
    __sympy_class__ = sympy.sign
    __new__ = DifferentiableFunction.__new__


class Abs(DifferentiableFunction, sympy.Abs):
    __sympy_class__ = sympy.Abs
    __new__ = DifferentiableFunction.__new__


class conjugate(DifferentiableFunction, sympy.conjugate):
    __sympy_class__ = sympy.conjugate
    __new__ = DifferentiableFunction.__new__


class arg(DifferentiableFunction, sympy.arg):
    __sympy_class__ = sympy.arg
    __new__ = DifferentiableFunction.__new__


class polar_lift(DifferentiableFunction, sympy.polar_lift):
    __sympy_class__ = sympy.polar_lift
    __new__ = DifferentiableFunction.__new__


class periodic_argument(DifferentiableFunction, sympy.periodic_argument):
    __sympy_class__ = sympy.periodic_argument
    __new__ = DifferentiableFunction.__new__


def unbranched_argument(x):
    return diffify(sympy.unbranched_argument(x))


class principal_branch(DifferentiableFunction, sympy.principal_branch):
    __sympy_class__ = sympy.principal_branch
    __new__ = DifferentiableFunction.__new__


class transpose(DifferentiableFunction, sympy.transpose):
    __sympy_class__ = sympy.transpose
    __new__ = DifferentiableFunction.__new__


class adjoint(DifferentiableFunction, sympy.adjoint):
    __sympy_class__ = sympy.adjoint
    __new__ = DifferentiableFunction.__new__


def polarify(x):
    return diffify(sympy.polarify(x))


def unpolarify(x):
    return diffify(sympy.unpolarify(x))


class sin(DifferentiableFunction, sympy.sin):
    __sympy_class__ = sympy.sin
    __new__ = DifferentiableFunction.__new__


class cos(DifferentiableFunction, sympy.cos):
    __sympy_class__ = sympy.cos
    __new__ = DifferentiableFunction.__new__


class tan(DifferentiableFunction, sympy.tan):
    __sympy_class__ = sympy.tan
    __new__ = DifferentiableFunction.__new__


class sec(DifferentiableFunction, sympy.sec):
    __sympy_class__ = sympy.sec
    __new__ = DifferentiableFunction.__new__


class csc(DifferentiableFunction, sympy.csc):
    __sympy_class__ = sympy.csc
    __new__ = DifferentiableFunction.__new__


class cot(DifferentiableFunction, sympy.cot):
    __sympy_class__ = sympy.cot
    __new__ = DifferentiableFunction.__new__


class sinc(DifferentiableFunction, sympy.sinc):
    __sympy_class__ = sympy.sinc
    __new__ = DifferentiableFunction.__new__


class asin(DifferentiableFunction, sympy.asin):
    __sympy_class__ = sympy.asin
    __new__ = DifferentiableFunction.__new__


class acos(DifferentiableFunction, sympy.acos):
    __sympy_class__ = sympy.acos
    __new__ = DifferentiableFunction.__new__


class atan(DifferentiableFunction, sympy.atan):
    __sympy_class__ = sympy.atan
    __new__ = DifferentiableFunction.__new__


class asec(DifferentiableFunction, sympy.asec):
    __sympy_class__ = sympy.asec
    __new__ = DifferentiableFunction.__new__


class acsc(DifferentiableFunction, sympy.acsc):
    __sympy_class__ = sympy.acsc
    __new__ = DifferentiableFunction.__new__


class acot(DifferentiableFunction, sympy.acot):
    __sympy_class__ = sympy.acot
    __new__ = DifferentiableFunction.__new__


class atan2(DifferentiableFunction, sympy.atan2):
    __sympy_class__ = sympy.atan2
    __new__ = DifferentiableFunction.__new__


class exp_polar(DifferentiableFunction, sympy.exp_polar):
    __sympy_class__ = sympy.exp_polar
    __new__ = DifferentiableFunction.__new__


class exp(DifferentiableFunction, sympy.exp):
    __sympy_class__ = sympy.exp
    __new__ = DifferentiableFunction.__new__


class ln(DifferentiableFunction, sympy.ln):
    __sympy_class__ = sympy.ln
    __new__ = DifferentiableFunction.__new__


class log(DifferentiableFunction, sympy.log):
    __sympy_class__ = sympy.log
    __new__ = DifferentiableFunction.__new__


class LambertW(DifferentiableFunction, sympy.LambertW):
    __sympy_class__ = sympy.LambertW
    __new__ = DifferentiableFunction.__new__


class sinh(DifferentiableFunction, sympy.sinh):
    __sympy_class__ = sympy.sinh
    __new__ = DifferentiableFunction.__new__


class cosh(DifferentiableFunction, sympy.cosh):
    __sympy_class__ = sympy.cosh
    __new__ = DifferentiableFunction.__new__


class tanh(DifferentiableFunction, sympy.tanh):
    __sympy_class__ = sympy.tanh
    __new__ = DifferentiableFunction.__new__


class coth(DifferentiableFunction, sympy.coth):
    __sympy_class__ = sympy.coth
    __new__ = DifferentiableFunction.__new__


class sech(DifferentiableFunction, sympy.sech):
    __sympy_class__ = sympy.sech
    __new__ = DifferentiableFunction.__new__


class csch(DifferentiableFunction, sympy.csch):
    __sympy_class__ = sympy.csch
    __new__ = DifferentiableFunction.__new__


class asinh(DifferentiableFunction, sympy.asinh):
    __sympy_class__ = sympy.asinh
    __new__ = DifferentiableFunction.__new__


class acosh(DifferentiableFunction, sympy.acosh):
    __sympy_class__ = sympy.acosh
    __new__ = DifferentiableFunction.__new__


class atanh(DifferentiableFunction, sympy.atanh):
    __sympy_class__ = sympy.atanh
    __new__ = DifferentiableFunction.__new__


class acoth(DifferentiableFunction, sympy.acoth):
    __sympy_class__ = sympy.acoth
    __new__ = DifferentiableFunction.__new__


class asech(DifferentiableFunction, sympy.asech):
    __sympy_class__ = sympy.asech
    __new__ = DifferentiableFunction.__new__


class acsch(DifferentiableFunction, sympy.acsch):
    __sympy_class__ = sympy.acsch
    __new__ = DifferentiableFunction.__new__


class floor(DifferentiableFunction, sympy.floor):
    __sympy_class__ = sympy.floor
    __new__ = DifferentiableFunction.__new__


class ceiling(DifferentiableFunction, sympy.ceiling):
    __sympy_class__ = sympy.ceiling
    __new__ = DifferentiableFunction.__new__


class frac(DifferentiableFunction, sympy.frac):
    __sympy_class__ = sympy.frac
    __new__ = DifferentiableFunction.__new__


class Piecewise(DifferentiableFunction, sympy.Piecewise):
    __sympy_class__ = sympy.Piecewise
    __new__ = DifferentiableFunction.__new__


def piecewise_fold(x):
    return diffify(sympy.piecewise_fold(x))


class erf(DifferentiableFunction, sympy.erf):
    __sympy_class__ = sympy.erf
    __new__ = DifferentiableFunction.__new__


class erfc(DifferentiableFunction, sympy.erfc):
    __sympy_class__ = sympy.erfc
    __new__ = DifferentiableFunction.__new__


class erfi(DifferentiableFunction, sympy.erfi):
    __sympy_class__ = sympy.erfi
    __new__ = DifferentiableFunction.__new__


class erf2(DifferentiableFunction, sympy.erf2):
    __sympy_class__ = sympy.erf2
    __new__ = DifferentiableFunction.__new__


class erfinv(DifferentiableFunction, sympy.erfinv):
    __sympy_class__ = sympy.erfinv
    __new__ = DifferentiableFunction.__new__


class erfcinv(DifferentiableFunction, sympy.erfcinv):
    __sympy_class__ = sympy.erfcinv
    __new__ = DifferentiableFunction.__new__


class erf2inv(DifferentiableFunction, sympy.erf2inv):
    __sympy_class__ = sympy.erf2inv
    __new__ = DifferentiableFunction.__new__


class Ei(DifferentiableFunction, sympy.Ei):
    __sympy_class__ = sympy.Ei
    __new__ = DifferentiableFunction.__new__


class expint(DifferentiableFunction, sympy.expint):
    __sympy_class__ = sympy.expint
    __new__ = DifferentiableFunction.__new__


def E1(x):
    return diffify(sympy.E1(x))


class li(DifferentiableFunction, sympy.li):
    __sympy_class__ = sympy.li
    __new__ = DifferentiableFunction.__new__


class Li(DifferentiableFunction, sympy.Li):
    __sympy_class__ = sympy.Li
    __new__ = DifferentiableFunction.__new__


class Si(DifferentiableFunction, sympy.Si):
    __sympy_class__ = sympy.Si
    __new__ = DifferentiableFunction.__new__


class Ci(DifferentiableFunction, sympy.Ci):
    __sympy_class__ = sympy.Ci
    __new__ = DifferentiableFunction.__new__


class Shi(DifferentiableFunction, sympy.Shi):
    __sympy_class__ = sympy.Shi
    __new__ = DifferentiableFunction.__new__


class Chi(DifferentiableFunction, sympy.Chi):
    __sympy_class__ = sympy.Chi
    __new__ = DifferentiableFunction.__new__


class fresnels(DifferentiableFunction, sympy.fresnels):
    __sympy_class__ = sympy.fresnels
    __new__ = DifferentiableFunction.__new__


class fresnelc(DifferentiableFunction, sympy.fresnelc):
    __sympy_class__ = sympy.fresnelc
    __new__ = DifferentiableFunction.__new__


class gamma(DifferentiableFunction, sympy.gamma):
    __sympy_class__ = sympy.gamma
    __new__ = DifferentiableFunction.__new__


class lowergamma(DifferentiableFunction, sympy.lowergamma):
    __sympy_class__ = sympy.lowergamma
    __new__ = DifferentiableFunction.__new__


class uppergamma(DifferentiableFunction, sympy.uppergamma):
    __sympy_class__ = sympy.uppergamma
    __new__ = DifferentiableFunction.__new__


class polygamma(DifferentiableFunction, sympy.polygamma):
    __sympy_class__ = sympy.polygamma
    __new__ = DifferentiableFunction.__new__


class loggamma(DifferentiableFunction, sympy.loggamma):
    __sympy_class__ = sympy.loggamma
    __new__ = DifferentiableFunction.__new__


class digamma(DifferentiableFunction, sympy.digamma):
    __sympy_class__ = sympy.digamma
    __new__ = DifferentiableFunction.__new__


class trigamma(DifferentiableFunction, sympy.trigamma):
    __sympy_class__ = sympy.trigamma
    __new__ = DifferentiableFunction.__new__


class multigamma(DifferentiableFunction, sympy.multigamma):
    __sympy_class__ = sympy.multigamma
    __new__ = DifferentiableFunction.__new__


class dirichlet_eta(DifferentiableFunction, sympy.dirichlet_eta):
    __sympy_class__ = sympy.dirichlet_eta
    __new__ = DifferentiableFunction.__new__


class zeta(DifferentiableFunction, sympy.zeta):
    __sympy_class__ = sympy.zeta
    __new__ = DifferentiableFunction.__new__


class lerchphi(DifferentiableFunction, sympy.lerchphi):
    __sympy_class__ = sympy.lerchphi
    __new__ = DifferentiableFunction.__new__


class polylog(DifferentiableFunction, sympy.polylog):
    __sympy_class__ = sympy.polylog
    __new__ = DifferentiableFunction.__new__


class stieltjes(DifferentiableFunction, sympy.stieltjes):
    __sympy_class__ = sympy.stieltjes
    __new__ = DifferentiableFunction.__new__


class riemann_xi(DifferentiableFunction, sympy.riemann_xi):
    __sympy_class__ = sympy.riemann_xi
    __new__ = DifferentiableFunction.__new__


def Eijk(x):
    return diffify(sympy.Eijk(x))


class LeviCivita(DifferentiableFunction, sympy.LeviCivita):
    __sympy_class__ = sympy.LeviCivita
    __new__ = DifferentiableFunction.__new__


class KroneckerDelta(DifferentiableFunction, sympy.KroneckerDelta):
    __sympy_class__ = sympy.KroneckerDelta
    __new__ = DifferentiableFunction.__new__


class SingularityFunction(DifferentiableFunction, sympy.SingularityFunction):
    __sympy_class__ = sympy.SingularityFunction
    __new__ = DifferentiableFunction.__new__


class DiracDelta(DifferentiableFunction, sympy.DiracDelta):
    __sympy_class__ = sympy.DiracDelta
    __new__ = DifferentiableFunction.__new__


class Heaviside(DifferentiableFunction, sympy.Heaviside):
    __sympy_class__ = sympy.Heaviside
    __new__ = DifferentiableFunction.__new__


def bspline_basis(x):
    return diffify(sympy.bspline_basis(x))


def bspline_basis_set(x):
    return diffify(sympy.bspline_basis_set(x))


def interpolating_spline(x):
    return diffify(sympy.interpolating_spline(x))


class besselj(DifferentiableFunction, sympy.besselj):
    __sympy_class__ = sympy.besselj
    __new__ = DifferentiableFunction.__new__


class bessely(DifferentiableFunction, sympy.bessely):
    __sympy_class__ = sympy.bessely
    __new__ = DifferentiableFunction.__new__


class besseli(DifferentiableFunction, sympy.besseli):
    __sympy_class__ = sympy.besseli
    __new__ = DifferentiableFunction.__new__


class besselk(DifferentiableFunction, sympy.besselk):
    __sympy_class__ = sympy.besselk
    __new__ = DifferentiableFunction.__new__


class hankel1(DifferentiableFunction, sympy.hankel1):
    __sympy_class__ = sympy.hankel1
    __new__ = DifferentiableFunction.__new__


class hankel2(DifferentiableFunction, sympy.hankel2):
    __sympy_class__ = sympy.hankel2
    __new__ = DifferentiableFunction.__new__


class jn(DifferentiableFunction, sympy.jn):
    __sympy_class__ = sympy.jn
    __new__ = DifferentiableFunction.__new__


class yn(DifferentiableFunction, sympy.yn):
    __sympy_class__ = sympy.yn
    __new__ = DifferentiableFunction.__new__


def jn_zeros(x):
    return diffify(sympy.jn_zeros(x))


class hn1(DifferentiableFunction, sympy.hn1):
    __sympy_class__ = sympy.hn1
    __new__ = DifferentiableFunction.__new__


class hn2(DifferentiableFunction, sympy.hn2):
    __sympy_class__ = sympy.hn2
    __new__ = DifferentiableFunction.__new__


class airyai(DifferentiableFunction, sympy.airyai):
    __sympy_class__ = sympy.airyai
    __new__ = DifferentiableFunction.__new__


class airybi(DifferentiableFunction, sympy.airybi):
    __sympy_class__ = sympy.airybi
    __new__ = DifferentiableFunction.__new__


class airyaiprime(DifferentiableFunction, sympy.airyaiprime):
    __sympy_class__ = sympy.airyaiprime
    __new__ = DifferentiableFunction.__new__


class airybiprime(DifferentiableFunction, sympy.airybiprime):
    __sympy_class__ = sympy.airybiprime
    __new__ = DifferentiableFunction.__new__


class marcumq(DifferentiableFunction, sympy.marcumq):
    __sympy_class__ = sympy.marcumq
    __new__ = DifferentiableFunction.__new__


class hyper(DifferentiableFunction, sympy.hyper):
    __sympy_class__ = sympy.hyper
    __new__ = DifferentiableFunction.__new__


class meijerg(DifferentiableFunction, sympy.meijerg):
    __sympy_class__ = sympy.meijerg
    __new__ = DifferentiableFunction.__new__


class appellf1(DifferentiableFunction, sympy.appellf1):
    __sympy_class__ = sympy.appellf1
    __new__ = DifferentiableFunction.__new__


class legendre(DifferentiableFunction, sympy.legendre):
    __sympy_class__ = sympy.legendre
    __new__ = DifferentiableFunction.__new__


class assoc_legendre(DifferentiableFunction, sympy.assoc_legendre):
    __sympy_class__ = sympy.assoc_legendre
    __new__ = DifferentiableFunction.__new__


class hermite(DifferentiableFunction, sympy.hermite):
    __sympy_class__ = sympy.hermite
    __new__ = DifferentiableFunction.__new__


class chebyshevt(DifferentiableFunction, sympy.chebyshevt):
    __sympy_class__ = sympy.chebyshevt
    __new__ = DifferentiableFunction.__new__


class chebyshevu(DifferentiableFunction, sympy.chebyshevu):
    __sympy_class__ = sympy.chebyshevu
    __new__ = DifferentiableFunction.__new__


class chebyshevu_root(DifferentiableFunction, sympy.chebyshevu_root):
    __sympy_class__ = sympy.chebyshevu_root
    __new__ = DifferentiableFunction.__new__


class chebyshevt_root(DifferentiableFunction, sympy.chebyshevt_root):
    __sympy_class__ = sympy.chebyshevt_root
    __new__ = DifferentiableFunction.__new__


class laguerre(DifferentiableFunction, sympy.laguerre):
    __sympy_class__ = sympy.laguerre
    __new__ = DifferentiableFunction.__new__


class assoc_laguerre(DifferentiableFunction, sympy.assoc_laguerre):
    __sympy_class__ = sympy.assoc_laguerre
    __new__ = DifferentiableFunction.__new__


class gegenbauer(DifferentiableFunction, sympy.gegenbauer):
    __sympy_class__ = sympy.gegenbauer
    __new__ = DifferentiableFunction.__new__


class jacobi(DifferentiableFunction, sympy.jacobi):
    __sympy_class__ = sympy.jacobi
    __new__ = DifferentiableFunction.__new__


def jacobi_normalized(x):
    return diffify(sympy.jacobi_normalized(x))


class Ynm(DifferentiableFunction, sympy.Ynm):
    __sympy_class__ = sympy.Ynm
    __new__ = DifferentiableFunction.__new__


def Ynm_c(x):
    return diffify(sympy.Ynm_c(x))


class Znm(DifferentiableFunction, sympy.Znm):
    __sympy_class__ = sympy.Znm
    __new__ = DifferentiableFunction.__new__


class elliptic_k(DifferentiableFunction, sympy.elliptic_k):
    __sympy_class__ = sympy.elliptic_k
    __new__ = DifferentiableFunction.__new__


class elliptic_f(DifferentiableFunction, sympy.elliptic_f):
    __sympy_class__ = sympy.elliptic_f
    __new__ = DifferentiableFunction.__new__


class elliptic_e(DifferentiableFunction, sympy.elliptic_e):
    __sympy_class__ = sympy.elliptic_e
    __new__ = DifferentiableFunction.__new__


class elliptic_pi(DifferentiableFunction, sympy.elliptic_pi):
    __sympy_class__ = sympy.elliptic_pi
    __new__ = DifferentiableFunction.__new__


class beta(DifferentiableFunction, sympy.beta):
    __sympy_class__ = sympy.beta
    __new__ = DifferentiableFunction.__new__


class betainc(DifferentiableFunction, sympy.betainc):
    __sympy_class__ = sympy.betainc
    __new__ = DifferentiableFunction.__new__


class betainc_regularized(DifferentiableFunction, sympy.betainc_regularized):
    __sympy_class__ = sympy.betainc_regularized
    __new__ = DifferentiableFunction.__new__


class mathieus(DifferentiableFunction, sympy.mathieus):
    __sympy_class__ = sympy.mathieus
    __new__ = DifferentiableFunction.__new__


class mathieuc(DifferentiableFunction, sympy.mathieuc):
    __sympy_class__ = sympy.mathieuc
    __new__ = DifferentiableFunction.__new__


class mathieusprime(DifferentiableFunction, sympy.mathieusprime):
    __sympy_class__ = sympy.mathieusprime
    __new__ = DifferentiableFunction.__new__


class mathieucprime(DifferentiableFunction, sympy.mathieucprime):
    __sympy_class__ = sympy.mathieucprime
    __new__ = DifferentiableFunction.__new__
