import numpy as np
from numpy import ndarray


def linear(x, beta_1, beta_2) -> ndarray:
    return beta_1*x-beta_2


# Modello esponenziale: Y = exp(a + b*X)
def exponensial(x: ndarray, beta_1, beta_2) -> ndarray:
    roba = None
    try:
        roba = np.exp((beta_1+x*beta_2))
    except RuntimeWarning:
        print('we')
        return None
    return roba


# Modello Y al quadrato: Y = sqrt(a + b*X)
def ySquare(x, beta_1, beta_2) -> ndarray:
    return np.sqrt((beta_1+x*beta_2))


# Modello radice quadrata di X: Y = a + b*sqrt(X)
def xSqrt(x, beta_1, beta_2) -> ndarray:
    return beta_1 + beta_2*np.sqrt(x)


# Modello radice quadrata doppia: Y = (a + b*sqrt(X))^2
def doubleSqrt(x, beta_1, beta_2) -> ndarray:
    return (beta_1 * beta_2*np.sqrt(x))**2


# Modello logaritmo di Y, radice quadrata di X: Y = exp(a + b*sqrt(X))
def xSqrt_yLogarithm(x, beta_1, beta_2) -> ndarray:
    return np.exp((beta_1 + beta_2*np.sqrt(x)))


# Modello Y al quadrato, radice quadrata di X: Y = sqrt(a + b*sqrt(X))
def xSqrt_ySquare(x, beta_1, beta_2) -> ndarray:
    return np.sqrt(beta_1 + beta_2*np.sqrt(x))


# Modello logaritmo di X: Y = a + b*ln(X)
def xLogorithm(x, beta_1, beta_2) -> ndarray:
    return beta_1 + beta_2*np.log(x)


# Modello radice quadrata di Y, logaritmo di X: Y = (a + b*ln(X))^2
def xLogarithm_ySqrt(x, beta_1, beta_2) -> ndarray:
    return (beta_1 + beta_2*np.log(x))**2


# Modello moltiplicativo: Y = a*X^b -----    Col_2 = exp(a + b*ln(Col_1))
def multiplicative(x, beta_1, beta_2) -> ndarray:
    return np.exp(beta_1 + beta_2*np.log(x))


# Modello Y al quadrato, logaritmo di X: Y = sqrt(a + b*ln(X))
def xLogarithm_ySquare(x, beta_1, beta_2) -> ndarray:
    return np.sqrt((beta_1 + beta_2*np.log(x)))


# Modello reciproco di X: Y = a + b/X
def xReciprocal(x, beta_1, beta_2) -> ndarray:
    return beta_1 + beta_2/x


# Modello radice quadrata di Y, reciproco di X: Y = (a + b/X)^2
def xReciprocal_ySqrt(x, beta_1, beta_2) -> ndarray:
    return (beta_1 + beta_2/x)**2


# Modello curva S: Y = exp(a + b/X)
def sCurve(x, beta_1, beta_2) -> ndarray:
    return np.exp(beta_1 + beta_2/x)


# Modello reciproco doppio: Y = 1/(a + b/X)
def doubleReciprocal(x, beta_1, beta_2) -> ndarray:
    return 1/(beta_1 + beta_2/x)


# Modello Y al quadrato, reciproco di X: Y = sqrt(a + b/X)
def xReciprocal_ySquare(x, beta_1, beta_2) -> ndarray:
    return np.sqrt(beta_1 + beta_2/x)


# Modello X al quadrato: Y = a + b*X^2
def xSquare(x, beta_1, beta_2) -> ndarray:
    return beta_1 + beta_2*x*x


# Modello radice quadrata di Y, X al quadrato: Y = (a + b*X^2)^2
def xSquare_ySqrt(x, beta_1, beta_2) -> ndarray:
    return (beta_1 + beta_2*x*x)**2


# Modello Logaritmo di Y, X al quadrato: Y = exp(a + b*X^2)
def xSquare_yLogarithm(x, beta_1, beta_2) -> ndarray:
    return np.exp(beta_1 + beta_2*x*x)


def xLogorithm_yReciprocal(x, beta_1, beta_2) -> ndarray:
    return 1/(beta_1 + beta_2*np.log(x))


# Modello Reciproco di Y, X al quadrato: Y = 1/(a + b*X^2)
def xSquare_yReciprocal(x, beta_1, beta_2) -> ndarray:
    return 1/(beta_1 + beta_2*x*x)


def yReciprocal(x, beta_1, beta_2) -> ndarray:
    return 1/(beta_1 + beta_2*x)


# Modello Doppio quadrato: Y = sqrt(a + b*X^2)
def doubleSquare(x, beta_1, beta_2) -> ndarray:
    return np.sqrt(beta_1 + beta_2*x*x)


def logistic(x, beta_1, beta_2) -> ndarray:
    return 1 / (1 + np.exp(-beta_1 * (x - beta_2)))


regressionFunctions = [linear, exponensial, ySquare, xSqrt, doubleSqrt, xSqrt_yLogarithm, xSqrt_ySquare, xLogorithm,
                       xLogarithm_ySqrt, multiplicative, xLogarithm_ySquare, xReciprocal, xReciprocal_ySqrt,
                       sCurve, doubleReciprocal, xReciprocal_ySquare, xSquare, xSquare_ySqrt, xSquare_yLogarithm,
                       xLogorithm_yReciprocal, xSquare_yReciprocal, yReciprocal, doubleSquare, logistic]
