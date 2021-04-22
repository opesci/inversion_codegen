import sympy

from devito.finite_differences.differentiable import DifferentiableFunction, diffify


class factorial(DifferentiableFunction, sympy.factorial):
    __sympy_class__ = sympy.factorial


class factorial2(DifferentiableFunction, sympy.factorial2):
    __sympy_class__ = sympy.factorial2


class rf(DifferentiableFunction, sympy.rf):
    __sympy_class__ = sympy.rf


class ff(DifferentiableFunction, sympy.ff):
    __sympy_class__ = sympy.ff


class binomial(DifferentiableFunction, sympy.binomial):
    __sympy_class__ = sympy.binomial


class RisingFactorial(DifferentiableFunction, sympy.RisingFactorial):
    __sympy_class__ = sympy.RisingFactorial


class FallingFactorial(DifferentiableFunction, sympy.FallingFactorial):
    __sympy_class__ = sympy.FallingFactorial


class subfactorial(DifferentiableFunction, sympy.subfactorial):
    __sympy_class__ = sympy.subfactorial


class carmichael(DifferentiableFunction, sympy.carmichael):
    __sympy_class__ = sympy.carmichael


class fibonacci(DifferentiableFunction, sympy.fibonacci):
    __sympy_class__ = sympy.fibonacci


class lucas(DifferentiableFunction, sympy.lucas):
    __sympy_class__ = sympy.lucas


class tribonacci(DifferentiableFunction, sympy.tribonacci):
    __sympy_class__ = sympy.tribonacci


class harmonic(DifferentiableFunction, sympy.harmonic):
    __sympy_class__ = sympy.harmonic


class bernoulli(DifferentiableFunction, sympy.bernoulli):
    __sympy_class__ = sympy.bernoulli


class bell(DifferentiableFunction, sympy.bell):
    __sympy_class__ = sympy.bell


class euler(DifferentiableFunction, sympy.euler):
    __sympy_class__ = sympy.euler


class catalan(DifferentiableFunction, sympy.catalan):
    __sympy_class__ = sympy.catalan


class genocchi(DifferentiableFunction, sympy.genocchi):
    __sympy_class__ = sympy.genocchi


class partition(DifferentiableFunction, sympy.partition):
    __sympy_class__ = sympy.partition


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


class im(DifferentiableFunction, sympy.im):
    __sympy_class__ = sympy.im


class sign(DifferentiableFunction, sympy.sign):
    __sympy_class__ = sympy.sign


class Abs(DifferentiableFunction, sympy.Abs):
    __sympy_class__ = sympy.Abs


class conjugate(DifferentiableFunction, sympy.conjugate):
    __sympy_class__ = sympy.conjugate


class arg(DifferentiableFunction, sympy.arg):
    __sympy_class__ = sympy.arg


class polar_lift(DifferentiableFunction, sympy.polar_lift):
    __sympy_class__ = sympy.polar_lift


class periodic_argument(DifferentiableFunction, sympy.periodic_argument):
    __sympy_class__ = sympy.periodic_argument


def unbranched_argument(x):
    return diffify(sympy.unbranched_argument(x))


class principal_branch(DifferentiableFunction, sympy.principal_branch):
    __sympy_class__ = sympy.principal_branch


class transpose(DifferentiableFunction, sympy.transpose):
    __sympy_class__ = sympy.transpose


class adjoint(DifferentiableFunction, sympy.adjoint):
    __sympy_class__ = sympy.adjoint


def polarify(x):
    return diffify(sympy.polarify(x))


def unpolarify(x):
    return diffify(sympy.unpolarify(x))


class sin(DifferentiableFunction, sympy.sin):
    __sympy_class__ = sympy.sin


class cos(DifferentiableFunction, sympy.cos):
    __sympy_class__ = sympy.cos


class tan(DifferentiableFunction, sympy.tan):
    __sympy_class__ = sympy.tan


class sec(DifferentiableFunction, sympy.sec):
    __sympy_class__ = sympy.sec


class csc(DifferentiableFunction, sympy.csc):
    __sympy_class__ = sympy.csc


class cot(DifferentiableFunction, sympy.cot):
    __sympy_class__ = sympy.cot


class sinc(DifferentiableFunction, sympy.sinc):
    __sympy_class__ = sympy.sinc


class asin(DifferentiableFunction, sympy.asin):
    __sympy_class__ = sympy.asin


class acos(DifferentiableFunction, sympy.acos):
    __sympy_class__ = sympy.acos


class atan(DifferentiableFunction, sympy.atan):
    __sympy_class__ = sympy.atan


class asec(DifferentiableFunction, sympy.asec):
    __sympy_class__ = sympy.asec


class acsc(DifferentiableFunction, sympy.acsc):
    __sympy_class__ = sympy.acsc


class acot(DifferentiableFunction, sympy.acot):
    __sympy_class__ = sympy.acot


class atan2(DifferentiableFunction, sympy.atan2):
    __sympy_class__ = sympy.atan2


class exp_polar(DifferentiableFunction, sympy.exp_polar):
    __sympy_class__ = sympy.exp_polar


class exp(DifferentiableFunction, sympy.exp):
    __sympy_class__ = sympy.exp


class ln(DifferentiableFunction, sympy.ln):
    __sympy_class__ = sympy.ln


class log(DifferentiableFunction, sympy.log):
    __sympy_class__ = sympy.log


class LambertW(DifferentiableFunction, sympy.LambertW):
    __sympy_class__ = sympy.LambertW


class sinh(DifferentiableFunction, sympy.sinh):
    __sympy_class__ = sympy.sinh


class cosh(DifferentiableFunction, sympy.cosh):
    __sympy_class__ = sympy.cosh


class tanh(DifferentiableFunction, sympy.tanh):
    __sympy_class__ = sympy.tanh


class coth(DifferentiableFunction, sympy.coth):
    __sympy_class__ = sympy.coth


class sech(DifferentiableFunction, sympy.sech):
    __sympy_class__ = sympy.sech


class csch(DifferentiableFunction, sympy.csch):
    __sympy_class__ = sympy.csch


class asinh(DifferentiableFunction, sympy.asinh):
    __sympy_class__ = sympy.asinh


class acosh(DifferentiableFunction, sympy.acosh):
    __sympy_class__ = sympy.acosh


class atanh(DifferentiableFunction, sympy.atanh):
    __sympy_class__ = sympy.atanh


class acoth(DifferentiableFunction, sympy.acoth):
    __sympy_class__ = sympy.acoth


class asech(DifferentiableFunction, sympy.asech):
    __sympy_class__ = sympy.asech


class acsch(DifferentiableFunction, sympy.acsch):
    __sympy_class__ = sympy.acsch


class floor(DifferentiableFunction, sympy.floor):
    __sympy_class__ = sympy.floor


class ceiling(DifferentiableFunction, sympy.ceiling):
    __sympy_class__ = sympy.ceiling


class frac(DifferentiableFunction, sympy.frac):
    __sympy_class__ = sympy.frac


class Piecewise(DifferentiableFunction, sympy.Piecewise):
    __sympy_class__ = sympy.Piecewise


def piecewise_fold(x):
    return diffify(sympy.piecewise_fold(x))


class erf(DifferentiableFunction, sympy.erf):
    __sympy_class__ = sympy.erf


class erfc(DifferentiableFunction, sympy.erfc):
    __sympy_class__ = sympy.erfc


class erfi(DifferentiableFunction, sympy.erfi):
    __sympy_class__ = sympy.erfi


class erf2(DifferentiableFunction, sympy.erf2):
    __sympy_class__ = sympy.erf2


class erfinv(DifferentiableFunction, sympy.erfinv):
    __sympy_class__ = sympy.erfinv


class erfcinv(DifferentiableFunction, sympy.erfcinv):
    __sympy_class__ = sympy.erfcinv


class erf2inv(DifferentiableFunction, sympy.erf2inv):
    __sympy_class__ = sympy.erf2inv


class Ei(DifferentiableFunction, sympy.Ei):
    __sympy_class__ = sympy.Ei


class expint(DifferentiableFunction, sympy.expint):
    __sympy_class__ = sympy.expint


def E1(x):
    return diffify(sympy.E1(x))


class li(DifferentiableFunction, sympy.li):
    __sympy_class__ = sympy.li


class Li(DifferentiableFunction, sympy.Li):
    __sympy_class__ = sympy.Li


class Si(DifferentiableFunction, sympy.Si):
    __sympy_class__ = sympy.Si


class Ci(DifferentiableFunction, sympy.Ci):
    __sympy_class__ = sympy.Ci


class Shi(DifferentiableFunction, sympy.Shi):
    __sympy_class__ = sympy.Shi


class Chi(DifferentiableFunction, sympy.Chi):
    __sympy_class__ = sympy.Chi


class fresnels(DifferentiableFunction, sympy.fresnels):
    __sympy_class__ = sympy.fresnels


class fresnelc(DifferentiableFunction, sympy.fresnelc):
    __sympy_class__ = sympy.fresnelc


class gamma(DifferentiableFunction, sympy.gamma):
    __sympy_class__ = sympy.gamma


class lowergamma(DifferentiableFunction, sympy.lowergamma):
    __sympy_class__ = sympy.lowergamma


class uppergamma(DifferentiableFunction, sympy.uppergamma):
    __sympy_class__ = sympy.uppergamma


class polygamma(DifferentiableFunction, sympy.polygamma):
    __sympy_class__ = sympy.polygamma


class loggamma(DifferentiableFunction, sympy.loggamma):
    __sympy_class__ = sympy.loggamma


class digamma(DifferentiableFunction, sympy.digamma):
    __sympy_class__ = sympy.digamma


class trigamma(DifferentiableFunction, sympy.trigamma):
    __sympy_class__ = sympy.trigamma


class multigamma(DifferentiableFunction, sympy.multigamma):
    __sympy_class__ = sympy.multigamma


class dirichlet_eta(DifferentiableFunction, sympy.dirichlet_eta):
    __sympy_class__ = sympy.dirichlet_eta


class zeta(DifferentiableFunction, sympy.zeta):
    __sympy_class__ = sympy.zeta


class lerchphi(DifferentiableFunction, sympy.lerchphi):
    __sympy_class__ = sympy.lerchphi


class polylog(DifferentiableFunction, sympy.polylog):
    __sympy_class__ = sympy.polylog


class stieltjes(DifferentiableFunction, sympy.stieltjes):
    __sympy_class__ = sympy.stieltjes


def Eijk(x):
    return diffify(sympy.Eijk(x))


class LeviCivita(DifferentiableFunction, sympy.LeviCivita):
    __sympy_class__ = sympy.LeviCivita


class KroneckerDelta(DifferentiableFunction, sympy.KroneckerDelta):
    __sympy_class__ = sympy.KroneckerDelta


class SingularityFunction(DifferentiableFunction, sympy.SingularityFunction):
    __sympy_class__ = sympy.SingularityFunction


class DiracDelta(DifferentiableFunction, sympy.DiracDelta):
    __sympy_class__ = sympy.DiracDelta


class Heaviside(DifferentiableFunction, sympy.Heaviside):
    __sympy_class__ = sympy.Heaviside


def bspline_basis(x):
    return diffify(sympy.bspline_basis(x))


def bspline_basis_set(x):
    return diffify(sympy.bspline_basis_set(x))


def interpolating_spline(x):
    return diffify(sympy.interpolating_spline(x))


class besselj(DifferentiableFunction, sympy.besselj):
    __sympy_class__ = sympy.besselj


class bessely(DifferentiableFunction, sympy.bessely):
    __sympy_class__ = sympy.bessely


class besseli(DifferentiableFunction, sympy.besseli):
    __sympy_class__ = sympy.besseli


class besselk(DifferentiableFunction, sympy.besselk):
    __sympy_class__ = sympy.besselk


class hankel1(DifferentiableFunction, sympy.hankel1):
    __sympy_class__ = sympy.hankel1


class hankel2(DifferentiableFunction, sympy.hankel2):
    __sympy_class__ = sympy.hankel2


class jn(DifferentiableFunction, sympy.jn):
    __sympy_class__ = sympy.jn


class yn(DifferentiableFunction, sympy.yn):
    __sympy_class__ = sympy.yn


def jn_zeros(x):
    return diffify(sympy.jn_zeros(x))


class hn1(DifferentiableFunction, sympy.hn1):
    __sympy_class__ = sympy.hn1


class hn2(DifferentiableFunction, sympy.hn2):
    __sympy_class__ = sympy.hn2


class airyai(DifferentiableFunction, sympy.airyai):
    __sympy_class__ = sympy.airyai


class airybi(DifferentiableFunction, sympy.airybi):
    __sympy_class__ = sympy.airybi


class airyaiprime(DifferentiableFunction, sympy.airyaiprime):
    __sympy_class__ = sympy.airyaiprime


class airybiprime(DifferentiableFunction, sympy.airybiprime):
    __sympy_class__ = sympy.airybiprime


class marcumq(DifferentiableFunction, sympy.marcumq):
    __sympy_class__ = sympy.marcumq


class hyper(DifferentiableFunction, sympy.hyper):
    __sympy_class__ = sympy.hyper


class meijerg(DifferentiableFunction, sympy.meijerg):
    __sympy_class__ = sympy.meijerg


class appellf1(DifferentiableFunction, sympy.appellf1):
    __sympy_class__ = sympy.appellf1


class legendre(DifferentiableFunction, sympy.legendre):
    __sympy_class__ = sympy.legendre


class assoc_legendre(DifferentiableFunction, sympy.assoc_legendre):
    __sympy_class__ = sympy.assoc_legendre


class hermite(DifferentiableFunction, sympy.hermite):
    __sympy_class__ = sympy.hermite


class chebyshevt(DifferentiableFunction, sympy.chebyshevt):
    __sympy_class__ = sympy.chebyshevt


class chebyshevu(DifferentiableFunction, sympy.chebyshevu):
    __sympy_class__ = sympy.chebyshevu


class chebyshevu_root(DifferentiableFunction, sympy.chebyshevu_root):
    __sympy_class__ = sympy.chebyshevu_root


class chebyshevt_root(DifferentiableFunction, sympy.chebyshevt_root):
    __sympy_class__ = sympy.chebyshevt_root


class laguerre(DifferentiableFunction, sympy.laguerre):
    __sympy_class__ = sympy.laguerre


class assoc_laguerre(DifferentiableFunction, sympy.assoc_laguerre):
    __sympy_class__ = sympy.assoc_laguerre


class gegenbauer(DifferentiableFunction, sympy.gegenbauer):
    __sympy_class__ = sympy.gegenbauer


class jacobi(DifferentiableFunction, sympy.jacobi):
    __sympy_class__ = sympy.jacobi


def jacobi_normalized(x):
    return diffify(sympy.jacobi_normalized(x))


class Ynm(DifferentiableFunction, sympy.Ynm):
    __sympy_class__ = sympy.Ynm


def Ynm_c(x):
    return diffify(sympy.Ynm_c(x))


class Znm(DifferentiableFunction, sympy.Znm):
    __sympy_class__ = sympy.Znm


class elliptic_k(DifferentiableFunction, sympy.elliptic_k):
    __sympy_class__ = sympy.elliptic_k


class elliptic_f(DifferentiableFunction, sympy.elliptic_f):
    __sympy_class__ = sympy.elliptic_f


class elliptic_e(DifferentiableFunction, sympy.elliptic_e):
    __sympy_class__ = sympy.elliptic_e


class elliptic_pi(DifferentiableFunction, sympy.elliptic_pi):
    __sympy_class__ = sympy.elliptic_pi


class beta(DifferentiableFunction, sympy.beta):
    __sympy_class__ = sympy.beta


class mathieus(DifferentiableFunction, sympy.mathieus):
    __sympy_class__ = sympy.mathieus


class mathieuc(DifferentiableFunction, sympy.mathieuc):
    __sympy_class__ = sympy.mathieuc


class mathieusprime(DifferentiableFunction, sympy.mathieusprime):
    __sympy_class__ = sympy.mathieusprime


class mathieucprime(DifferentiableFunction, sympy.mathieucprime):
    __sympy_class__ = sympy.mathieucprime


# New elementary functions in sympy 1.8
if float(sympy.__version__) >= 1.8:

    class motzkin(DifferentiableFunction, sympy.motzkin):
        __sympy_class__ = sympy.motzkin

    class riemann_xi(DifferentiableFunction, sympy.riemann_xi):
        __sympy_class__ = sympy.riemann_xi

    class betainc(DifferentiableFunction, sympy.betainc):
        __sympy_class__ = sympy.betainc

    class betainc_regularized(DifferentiableFunction, sympy.betainc_regularized):
        __sympy_class__ = sympy.betainc_regularized
