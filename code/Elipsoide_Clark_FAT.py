from __future__ import division
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

from fatiando.gravmag import sphere
from fatiando import mesher, gridder, utils
from fatiando.vis import mpl

import scipy.special
import scipy.interpolate


class GeometricElement(object):
    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        self.props = {}
        if props is not None:
            for p in props:
                self.props[p] = props[p]

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value
        
class Ellipsoid (GeometricElement):
    '''
    '''
    def __init__(self, xc, yc, zc, a, b, c, alfa, delta, gamma, props):
        GeometricElement.__init__(self, props)
        self.xc = float(xc)
        self.yc = float(yc)
        self.zc = float(zc)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.alfa = float(alfa)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.center = np.array([xc, yc, zc])
        self.l1 = l1_v (alfa, delta)
        self.l2 = l2_v (alfa, delta, gamma)
        self.l3 = l3_v (alfa, delta, gamma)

        self.m1 = m1_v (alfa, delta)
        self.m2 = m2_v (alfa, delta, gamma)
        self.m3 = m3_v (alfa, delta, gamma)

        self.n1 = n1_v (delta)
        self.n2 = n2_v (delta, gamma)
        self.n3 = n3_v (delta, gamma)
    
        self.ln = ln_v (props['remanence'][2], props['remanence'][1])
        self.mn = mn_v (props['remanence'][2], props['remanence'][1])
        self.nn = nn_v (props['remanence'][1])
        
        self.mcon = np.array([[self.l1, self.m1, self.n1],[self.l2, self.m2, self.n2],[self.l3, self.m3, self.n3]])
        self.mconT = (self.mcon).T
        self.k_dec = np.array([[props['k1'][2]],[props['k2'][2]],[props['k3'][2]]])
        self.k_int = np.array([[props['k1'][0]],[props['k2'][0]],[props['k3'][0]]])
        self.k_inc = np.array([[props['k1'][1]],[props['k2'][1]],[props['k3'][1]]])
        if props['k1'][0] == props['k2'][0] and props['k1'][0] == props['k3'][0]:
            self.km = k_matrix2 (self.k_int,self.l1,self.l2,self.l3,self.m1,self.m2,self.m3,self.n1,self.n2,self.n3)
        else:
            self.Lr = Lr_v (self.k_dec, self.k_inc)
            self.Mr = Mr_v (self.k_dec, self.k_inc)
            self.Nr = Nr_v (self.k_inc)
            self.km = k_matrix (self.k_int,self.Lr,self.Mr,self.Nr,self.l1,self.l2,self.l3,self.m1,self.m2,self.m3,self.n1,self.n2,self.n3)
        
        #self.Ft = F_e (inten,lt,mt,nt,l1,l2,l3,m1,m2,m3,n1,n2,n3)
        #self.JN = JN_e (ellipsoids.props['remanence'][0],ln,mn,nn,l1,l2,l3,m1,m2,m3,n1,n2,n3)
        #self.N1,self.N2,self.N3 = N_desmag (ellipsoids.a,ellipsoids.b,ellipsoids.c,F2,E2)
        #self.JR = JR_e (km,JN,Ft)
        #self.JRD = JRD_e (km,N1,N2,N3,JR)
        #self.JRD_carte = mconT.dot(JRD)
        #self.JRD_ang = utils.vec2ang(JRD_carte)

    def __str__(self):
        """Return a string representation of the ellipsoids."""
        names = [('xc', self.xc), ('yc', self.yc), ('zc', self.zc),
                 ('a', self.a), ('b', self.b), ('c', self.c),
                 ('alfa', self.alfa),('delta', self.delta),('gamma', self.gamma)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)
        
        
def elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids):
    '''
    Calcula as tres componentes do campo magnetico de um elipsoide.
    
    a: escalar - semi eixo maior
    b: escalar - semi eixo intermediario
    c: escalar - semi eixo menor
    h: escalar -  profundidade
    alfa: escalar - azimute do elipsoide em relacao ao "a"
    delta: escalar - inclinacao do elipsoide em relacao ao "a"
    gamma: escalar - angulo entre o semi eixo "b" e a projecao do centro do elipsoide no plano xy
    xp: matriz - malha do eixo x
    yp: matriz - malha do eixo y
    zp: matriz - malha do eixo z
    xc: escalar - posicao x do centro do elipsoide
    yc: escalar - posicao y do centro do elipsoide
    J: vetor - magnetizacao do corpo
    '''

    # Calculo de parametros de direcao
    #l1 = l1_v (ellipsoids.alfa, ellipsoids.delta)
    #l2 = l2_v (ellipsoids.alfa, ellipsoids.delta, ellipsoids.gamma)
    #l3 = l3_v (ellipsoids.alfa, ellipsoids.delta, ellipsoids.gamma)

    #m1 = m1_v (ellipsoids.alfa, ellipsoids.delta)
    #m2 = m2_v (ellipsoids.alfa, ellipsoids.delta, ellipsoids.gamma)
    #m3 = m3_v (ellipsoids.alfa, ellipsoids.delta, ellipsoids.gamma)

    #n1 = n1_v (ellipsoids.delta)
    #n2 = n2_v (ellipsoids.delta, ellipsoids.gamma)
    #n3 = n3_v (ellipsoids.delta, ellipsoids.gamma)
    
    #ln = ln_v (ellipsoids.props['remanence'][2], ellipsoids.props['remanence'][1])
    #mn = mn_v (ellipsoids.props['remanence'][2], ellipsoids.props['remanence'][1])
    #nn = nn_v (ellipsoids.props['remanence'][1])
    
    #mcon = np.array([[l1, m1, n1],[l2, m2, n2],[l3, m3, n3]])
    #mconT = mcon.T
    #print mconT
    
    lt = ln_v (dec, inc)
    mt = mn_v (dec, inc)
    nt = nn_v (inc)
    #print l1,m1,n1
    #print l2,m2,n2
    #print l3,m3,n3
    # Coordenadas Cartesianas elipsoide
    x1 = x1_e (xp,yp,zp,ellipsoids.xc,ellipsoids.yc,ellipsoids.zc,ellipsoids.l1,ellipsoids.m1,ellipsoids.n1)
    x2 = x2_e (xp,yp,zp,ellipsoids.xc,ellipsoids.yc,ellipsoids.zc,ellipsoids.l2,ellipsoids.m2,ellipsoids.n2)
    x3 = x3_e (xp,yp,zp,ellipsoids.xc,ellipsoids.yc,ellipsoids.zc,ellipsoids.l3,ellipsoids.m3,ellipsoids.n3)

    # Calculos auxiliares
    p0 = p0_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,x1,x2,x3)
    p1 = p1_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,x1,x2,x3)
    p2 = p2_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,x1,x2,x3)
    p = p_e (p1,p2)
    q = q_e (p0,p1,p2)
    teta = teta_e (p,q)

    # Raizes da equacao cubica
    lamb = lamb_e (p,teta,p2)
    
    # Calculo de parametros para as integrais
    F,E,F2,E2,k,teta_linha = parametros_integrais(ellipsoids.a,ellipsoids.b,ellipsoids.c,lamb)
    
    # Magnetizacoes nas coordenadas do elipsoide
    #k_dec = np.array([[ellipsoids.props['k1'][2]],[ellipsoids.props['k2'][2]],[ellipsoids.props['k3'][2]]])
    #k_int = np.array([[ellipsoids.props['k1'][0]],[ellipsoids.props['k2'][0]],[ellipsoids.props['k3'][0]]])
    #k_inc = np.array([[ellipsoids.props['k1'][1]],[ellipsoids.props['k2'][1]],[ellipsoids.props['k3'][1]]])
    #if ellipsoids.props['k1'][0] == ellipsoids.props['k2'][0] and ellipsoids.props['k1'][0] == ellipsoids.props['k3'][0]:
    #    km = k_matrix2 (k_int,l1,l2,l3,m1,m2,m3,n1,n2,n3)
    #else:
    #    Lr = Lr_v (k_dec, k_inc)
    #    Mr = Mr_v (k_dec, k_inc)
    #    Nr = Nr_v (k_inc)
    #    km = k_matrix (k_int,Lr,Mr,Nr,l1,l2,l3,m1,m2,m3,n1,n2,n3)
    Ft = F_e (inten,lt,mt,nt,ellipsoids.l1,ellipsoids.l2,ellipsoids.l3,ellipsoids.m1,ellipsoids.m2,ellipsoids.m3,ellipsoids.n1,ellipsoids.n2,ellipsoids.n3)
    JN = JN_e (ellipsoids.props['remanence'][0],ellipsoids.ln,ellipsoids.mn,ellipsoids.nn,ellipsoids.l1,ellipsoids.l2,ellipsoids.l3,ellipsoids.m1,ellipsoids.m2,ellipsoids.m3,ellipsoids.n1,ellipsoids.n2,ellipsoids.n3)
    N1,N2,N3 = N_desmag (ellipsoids.a,ellipsoids.b,ellipsoids.c,F2,E2)
    JR = JR_e (ellipsoids.km,JN,Ft)
    JRD = JRD_e (ellipsoids.km,N1,N2,N3,JR)
    JRD_carte = (ellipsoids.mconT).dot(JRD)
    JRD_ang = utils.vec2ang(JRD_carte)
    #print Ft
    #print JN
    #print JRD
    #print N1,N2,N3
    #print JRD_ang
    
    # Derivadas de lambda em relacao as posicoes
    dlambx1 = dlambx1_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,x1,x2,x3,lamb)
    dlambx2 = dlambx2_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,x1,x2,x3,lamb)
    dlambx3 = dlambx3_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,x1,x2,x3,lamb)
    
    # Calculo das integrais
    A, B, C = integrais_elipticas(ellipsoids.a,ellipsoids.b,ellipsoids.c,k,teta_linha,F,E)
    
    # Geometria para o calculo de B (eixo do elipsoide)
    cte = cte_m (ellipsoids.a,ellipsoids.b,ellipsoids.c,lamb)
    V1, V2, V3 = v_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,x1,x2,x3,lamb)
    
    # Calculo matriz geometria para B1
    m11 = (cte*dlambx1*V1) - A
    m12 = cte*dlambx1*V2
    m13 = cte*dlambx1*V3

    # Calculo matriz geometria para B2
    m21 = cte*dlambx2*V1
    m22 = (cte*dlambx2*V2) - B
    m23 = cte*dlambx2*V3

    # Calculo matriz geometria para B3
    m31 = cte*dlambx3*V1
    m32 = cte*dlambx3*V2
    m33 = (cte*dlambx3*V3) - C
    
    # Problema Direto (Calcular o campo externo nas coordenadas do elipsoide)
    B1 = B1_e (m11,m12,m13,JRD,ellipsoids.a,ellipsoids.b,ellipsoids.c)
    B2 = B2_e (m21,m22,m23,JRD,ellipsoids.a,ellipsoids.b,ellipsoids.c)
    B3 = B3_e (m31,m32,m33,JRD,ellipsoids.a,ellipsoids.b,ellipsoids.c)
    
    #constante = constante_nova (a,b,c,lamb,JRD,x1,x2,x3)
    #B1 = B1_novo (constante,dlambx1,a,b,c,JRD,A)
    #B2 = B2_novo (constante,dlambx2,a,b,c,JRD,B)
    #B3 = B3_novo (constante,dlambx3,a,b,c,JRD,C)
    
    # Problema Direto (Calcular o campo externo nas coordenadas geograficas)
    Bx = Bx_c (B1,B2,B3,ellipsoids.l1,ellipsoids.l2,ellipsoids.l3)
    By = By_c (B1,B2,B3,ellipsoids.m1,ellipsoids.m2,ellipsoids.m3)
    Bz = Bz_c (B1,B2,B3,ellipsoids.n1,ellipsoids.n2,ellipsoids.n3)
    
    return Bx,By,Bz,JRD_ang
    
    # Problema Direto (Calcular o campo externo e anomalia nas coordenadas geograficas no SI)
def bx_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1
    
    for i in range(len(ellipsoids)):
        bx,by,bz,jrd_ang = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        res += bx
    res = res*ctemag
    return res
    
def by_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1
    
    for i in range(len(ellipsoids)):
        bx,by,bz,jrd_ang = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        res += by
    res = res*ctemag
    return res

def bz_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1
    
    for i in range(len(ellipsoids)):
        bx,by,bz,jrd_ang = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        res += bz
    res = res*ctemag
    return res
    
def tf_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1
    
    for i in range(len(ellipsoids)):
        bx,by,bz,jrd_ang = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        tf = bx*np.cos(inc)*np.cos(dec) + by*np.cos(inc)*np.sin(dec) + bz*np.sin(inc)
        res += tf
    res = res*ctemag
    return res,jrd_ang
    
def l1_v (alfa, delta):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    
    output:
    Direcao em radianos.
    '''
    
    l1 = (-np.cos(alfa)*np.cos(delta))
    return l1

def l2_v (alfa, delta, gamma):

    '''
    Orientacao do elipsoide com respeito ao eixo y.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    l2 = (np.cos(alfa)*np.cos(gamma)*np.sin(delta)+np.sin(alfa)*np.sin(gamma))
    return l2

def l3_v (alfa, delta, gamma):

    '''
    Orientacao do elipsoide com respeito ao eixo z.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    l3 = (np.sin(alfa)*np.cos(gamma)-np.cos(alfa)*np.sin(gamma)*np.sin(delta))
    return l3

def m1_v (alfa, delta):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    
    output:
    Direcao em radianos.
    '''
    
    m1 = (-np.sin(alfa)*np.cos(delta))
    return m1

def m2_v (alfa, delta, gamma):

    '''
    Orientacao do elipsoide com respeito ao eixo y.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    m2 = (np.sin(alfa)*np.cos(gamma)*np.sin(delta)-np.cos(alfa)*np.sin(gamma))
    return m2

def m3_v (alfa, delta, gamma):

    '''
    Orientacao do elipsoide com respeito ao eixo z.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    m3 = (-np.cos(alfa)*np.cos(gamma)-np.sin(alfa)*np.sin(gamma)*np.sin(delta))
    return m3

def n1_v (delta):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)

    output:
    Direcao em radianos.
    '''
    
    n1 = (-np.sin(delta))
    return n1

def n2_v (delta, gamma):

    '''
    Orientacao do elipsoide com respeito ao eixo y.
    
    input:
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    n2 = (-np.cos(gamma)*np.cos(delta))
    return n2

def n3_v (delta, gamma):

    '''
    Orientacao do elipsoide com respeito ao eixo z.
    
    input:
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    n3 = (np.sin(gamma)*np.cos(delta))
    return n3
    
def ln_v (declinacao, inclinacao):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    
    output:
    Direcao em radianos.
    '''
    
    ln = (np.cos(declinacao)*np.cos(inclinacao))
    return ln
    
def mn_v (declinacao, inclinacao):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    
    output:
    Direcao em radianos.
    '''
    
    mn = (np.sin(declinacao)*np.cos(inclinacao))
    return mn
    
def nn_v (inclinacao):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    
    output:
    Direcao em radianos.
    '''
    
    nn = np.sin(inclinacao)
    return nn

def Lr_v (k_dec, k_inc):
    '''
    Cossenos diretores dos eixos dos vetores de susceptibilidade magnetica.
    
    input:
    k_dec - declinacoes dos vetores de susceptibilidade.
    k_inc - inclinacoes dos vetores de susceptibilidade.
    '''
    Lr = np.zeros(3)
    for i in range (3):
        Lr[i] = np.cos(k_dec[i])*np.cos(k_inc[i])
    return Lr
    
def Mr_v (k_dec, k_inc):
    '''
    Cossenos diretores dos eixos dos vetores de susceptibilidade magnetica.
    
    input:
    k_dec - declinacoes dos vetores de susceptibilidade.
    k_inc - inclinacoes dos vetores de susceptibilidade.
    '''
    Mr = np.zeros(3)
    for i in range (3):
        Mr[i] = np.sin(k_dec[i])*np.cos(k_inc[i])
    return Mr
    
def Nr_v (k_inc):
    '''
    Cossenos diretores dos eixos dos vetores de susceptibilidade magnetica.
    
    input:
    k_inc - inclinacoes dos vetores de susceptibilidade.
    '''
    Nr = np.zeros(3)
    for i in range (3):
        Nr[i] = np.sin(k_inc[i])
    return Nr
    
def F_e (intensidadeT,lt,mt,nt,l1,l2,l3,m1,m2,m3,n1,n2,n3):
    '''
    Transformacao do vetor campo magnetico da Terra para as coordenadas nos eixos do elipsoide.
    '''
    Ft = intensidadeT*np.ravel(np.array([[(lt*l1+mt*m1+nt*n1)], [(lt*l2+mt*m2+nt*n2)], [(lt*l3+mt*m3+nt*n3)]]))
    return Ft
    
def JN_e (intensidade,ln,mn,nn,l1,l2,l3,m1,m2,m3,n1,n2,n3):
    '''
    transformacao do Vetor de magnetizacao remanente para as coordenadas nos eixos do elipsoide.
    '''
    JN = intensidade*np.ravel(np.array([[(ln*l1+mn*m1+nn*n1)], [(ln*l2+mn*m2+nn*n2)], [(ln*l3+mn*m3+nn*n3)]]))
    return JN

def N_desmag (a,b,c,F2,E2):
    '''
    Fator de desmagnetizacao ao longo do eixo de revolucao (N1) e em relacao ao plano equatorial (N2).
    '''
    N1 = ((4.*np.pi*a*b*c)/((a**2-b**2)*(a**2-c**2)**0.5)) * (F2-E2)
    N2 = (((4.*np.pi*a*b*c)*(a**2-c**2)**0.5)/((a**2-b**2)*(b**2-c**2))) * (E2 - ((b**2-c**2)/(a**2-c**2)) * F2 - ((c*(a**2-b**2))/(a*b*(a**2-c**2)**0.5)))
    N3 = ((4.*np.pi*a*b*c)/((b**2-c**2)*(a**2-c**2)**0.5)) * (((b*(a**2-c**2)**0.5)/(a*c)) - E2)
    return N1, N2, N3
    
def k_matrix (k_int,Lr,Mr,Nr,l1,l2,l3,m1,m2,m3,n1,n2,n3):
    '''
    Matriz de tensores de susceptibilidade.
    '''
    l = np.array([[l1],[l2],[l3]])
    m = np.array([[m1],[m2],[m3]])
    n = np.array([[n1],[n2],[n3]])
    k = np.zeros([3,3])
    for i in range (3):
        for j in range (3):
            for r in range (3):
                k[i,j] = k[i,j] + (k_int[r]*(Lr[r]*l[i] + Mr[r]*m[i] + Nr[r]*n[i])*(Lr[r]*l[j] + Mr[r]*m[j] + Nr[r]*n[j]))
    return k
    
def k_matrix2 (k_int,l1,l2,l3,m1,m2,m3,n1,n2,n3):
    '''
    Matriz de tensores de susceptibilidade.
    '''
    l = np.array([[l1],[l2],[l3]])
    m = np.array([[m1],[m2],[m3]])
    n = np.array([[n1],[n2],[n3]])
    k = np.zeros([3,3])
    for i in range (3):
        for j in range (3):
            for r in range (3):
                k[i,j] = k[i,j] + (k_int[r]*(l[r]*l[i] + m[r]*m[i] + n[r]*n[i])*(l[r]*l[j] + m[r]*m[j] + n[r]*n[j]))
    return k
    
def JR_e (km,JN,Ft):
    '''
    Vetor de magnetizacao resultante sem correcao da desmagnetizacao.
    '''
    JR = km.dot(Ft) + JN
    return JR
    
def JRD_e (km,N1,N2,N3,JR):
    '''
    Vetor de magnetizacao resultante com a correcao da desmagnetizacao.
    '''
    I = np.identity(3)
    kn0 = km[:,0]*N1
    kn1 = km[:,1]*N2
    kn2 = km[:,2]*N3
    kn = (np.vstack((kn0,kn1,kn2))).T
    A = I + kn
    JRD = (linalg.inv(A)).dot(JR)
    return JRD
    
def x1_e (xp,yp,zp,xc,yc,h,l1,m1,n1):
    '''
    Calculo da coordenada x no elipsoide
    input:
    xp,yp - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l1,m1,n1 - Orientacao do elipsoide (eixo x)
    output:
    x1 - Coordenada x do elipsoide.
    '''
    x1 = (xp-xc)*l1+(yp-yc)*m1+(-zp-h)*n1
    return x1

def x2_e (xp,yp,zp,xc,yc,h,l2,m2,n2):
    '''
    Calculo da coordenada y no elipsoide
    input:
    xp,yp - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l2,m2,n2 - Orientacao do elipsoide (eixo y).
    output:
    x2 - Coordenada y do elipsoide.
    '''
    x2 = (xp-xc)*l2+(yp-yc)*m2+(-zp-h)*n2
    return x2

def x3_e (xp,yp,zp,xc,yc,h,l3,m3,n3):
    '''
    Calculo da coordenada z no elipsoide
    input:
    xp,yp - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l3,m3,n3 - Orientacao do elipsoide (eixo z).
    output:
    x3 - Coordenada z do elipsoide.
    '''
    x3 = (xp-xc)*l3+(yp-yc)*m3+(-zp-h)*n3
    return x3
    
def p0_e (a,b,c,x1,x2,x3):
    '''
    Constante da equacao cubica: s^3 + p2*s^2 + p0 = 0
    input:
    a,b,c - Eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    output:
    p0 - Constante
    '''
    p0 = (a*b*c)**2-(b*c*x1)**2-(c*a*x2)**2-(a*b*x3)**2
    return p0
    
def p1_e (a,b,c,x1,x2,x3):
    '''
    Constante da equacao cubica: s^3 + p2*s^2 + p0 = 0
    input:
    a,b,c - Eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    output:
    p0 - Constante
    '''
    p1 = (a*b)**2+(b*c)**2+(c*a)**2-(b**2+c**2)*x1**2-(c**2+a**2)*x2**2-(a**2+b**2)*x3**2
    return p1
    
def p2_e (a,b,c,x1,x2,x3):
    '''
    Constante da equacao cubica: s^3 + p2*s^2 + p0 = 0
    input:
    a,b,c - Eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    output:
    p0 - Constante
    '''
    p2 = a**2+b**2+c**2-x1**2-x2**2-x3**2
    return p2
    
def p_e (p1,p2):
    '''
    Constante
    input:
    p1,p2 - constantes da equacao cubica
    output:
    p - Constante.
    '''
    p = p1-(p2**2)/3.
    return p
    
def q_e (p0,p1,p2):
    '''
    Constante
    input:
    p0,p1,p2 - constantes da equacao cubica
    output:
    q - Constante.
    '''
    q = p0-((p1*p2)/3.)+2*(p2/3.)**3
    return q
    
def teta_e (p,q):
    '''
    Constante angular (radianos)
    input:
    p - constante da equacao cubica
    q - constante
    output:
    teta - Constante.
    '''
    teta = np.arccos(-q/(2*np.sqrt((-p/3.)**3)))
    #teta = np.arccos((-q/2.)*np.sqrt((-p/3.)**3))
    return teta
    
def lamb_e (p,teta,p2):
    '''
    Maior raiz real da equacao cubica: s^3 + p2*s^2 + p0 = 0
    input:
    p,p2 - constantes da equacao cubica
    teta - constante angular (radianos)
    output:
    lamb - Maior raiz real.
    '''
    lamb = 2.*((-p/3.)**0.5)*np.cos(teta/3.)-(p2/3.)
    #lamb = 2*((-p/3.)*np.cos(teta/3.)-(p2/3.))**0.5
    #lamb = 2*((-p/3.)*np.cos(teta/3.))**0.5 - (p2/3.)
    return lamb

def mi_e (p,teta,p2):
    '''
    Raiz intermediaria real da equacao cubica: s^3 + p2*s^2 + p0 = 0
    input:
    p,p2 - constantes da equacao cubica
    teta - constante angular (radianos)
    output:
    mi - Raiz intermediaria real.
    '''
    mi = -2.*((-p/3.)**0.5)*np.cos(teta/3.+np.pi/3.)-(p2/3.)
    return mi

def ni_e (p,teta,p2):
    '''
    Menor raiz real da equacao cubica: s^3 + p2*s^2 + p0 = 0
    input:
    p,p2 - constantes da equacao cubica
    teta - constante angular (radianos)
    output:
    ni - Menor raiz real.
    '''
    ni = -2.*np.sqrt(-p/3.)*np.cos(teta/3. - np.pi/3.)-(p2/3.)
    return ni

def dlambx1_e (a,b,c,x1,x2,x3,lamb):
    '''
    Derivada de lamb em relacao ao eixo x1 do elipsoide.
    input:
    a,b,c, - semi-eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    dlambx1 - escalar
    '''
    dlambx1 = (2*x1/(a**2+lamb))/((x1/(a**2+lamb))**2+(x2/(b**2+lamb))**2+((x3/(c**2+lamb))**2))
    return dlambx1

def dlambx2_e (a,b,c,x1,x2,x3,lamb):
    '''
    Derivada de lamb em relacao ao eixo x2 do elipsoide.
    input:
    a,b,c, - semi-eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    dlambx2 - escalar
    '''
    dlambx2 = (2*x2/(b**2+lamb))/((x1/(a**2+lamb))**2+(x2/(b**2+lamb))**2+((x3/(c**2+lamb))**2))
    return dlambx2

def dlambx3_e (a,b,c,x1,x2,x3,lamb):
    '''
    Derivada de lamb em relacao ao eixo x3 do elipsoide.
    input:
    a,b,c, - semi-eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    dlambx3 - escalar
    '''
    dlambx3 = (2*x3/(c**2+lamb))/((x1/(a**2+lamb))**2+(x2/(b**2+lamb))**2+((x3/(c**2+lamb))**2))
    return dlambx3   

def cte_m (a,b,c,lamb):
    '''
    Fator geometrico do campo magnetico (fi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide
    lamb - Maior raiz real da equacao cubica.
    output:
    cte - constante escalar.
    '''
    cte = 1/np.sqrt((a**2+lamb)*(b**2+lamb)*(c**2+lamb))
    return cte
    
def v_e (a,b,c,x1,x2,x3,lamb):
    '''
    Constante do campo magnetico (fi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    v - matriz
    '''
    V1 = x1/(a**2+lamb)
    V2 = x2/(b**2+lamb)
    V3 = x3/(c**2+lamb)
    return V1, V2, V3

def B1_e (m11,m12,m13,J,a,b,c):
    '''
    Calculo do campo magnetico (Bi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    dlambx1 - matriz: derivada de lambda em relacao ao eixo x1.
    cte - matriz
    v - matriz
    A - matriz: integrais do potencial
    J - vetor: magnetizacao
    output:
    B1 - matriz
    '''
    B1 = 2*np.pi*a*b*c*(m11*J[0]+m12*J[1]+m13*J[2])
    return B1

def B2_e (m21,m22,m23,J,a,b,c):
    '''
    Calculo do campo magnetico (Bi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    dlambx2 - matriz: derivada de lambda em relacao ao eixo x2.
    cte - matriz
    v - matriz
    B - matriz: integrais do potencial
    J - vetor: magnetizacao
    output:
    B2 - matriz
    '''
    B2 = 2*np.pi*a*b*c*(m21*J[0]+m22*J[1]+m23*J[2])
    return B2
    
def B3_e (m31,m32,m33,J,a,b,c):
    '''
    Calculo do campo magnetico (Bi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    dlambx3 - matriz: derivada de lambda em relacao ao eixo x3.
    cte - matriz
    v - matriz
    C - matriz: integrais do potencial
    J - vetor: magnetizacao
    output:
    B3 - matriz
    '''
    B3 = 2*np.pi*a*b*c*(m31*J[0]+m32*J[1]+m33*J[2])
    return B3
    
def Bx_c (B1,B2,B3,l1,l2,l3):
    '''
    Calculo do campo magnetico (Bi) com relacao aos eixos geograficos
    input:
    B1,B2,B3 - vetores
    l1,l2,l3 - escalares.
    output:
    Bx - matriz
    '''
    Bx = B1*l1+B2*l2+B3*l3
    return Bx

def By_c (B1,B2,B3,m1,m2,m3):
    '''
    Calculo do campo magnetico (Bi) com relacao aos eixos geograficos
    input:
    B1,B2,B3 - vetores
    m1,m2,m3 - escalares.
    output:
    By - matriz
    '''
    By = B1*m1+B2*m2+B3*m3
    return By
    
def Bz_c (B1,B2,B3,n1,n2,n3):
    '''
    Calculo do campo magnetico (Bi) com relacao aos eixos geograficos
    input:
    B1,B2,B3 - vetores
    n1,n2,n3 - escalares.
    output:
    Bz - matriz
    '''
    Bz = B1*n1+B2*n2+B3*n3
    return Bz
    
def Alambda_simp_ext3(a,b,c,lamb):
    '''
    Integral do potencial utilizando o metodo de simpson.
    input:
    a,b,c - semi-eixos do elipsoide
    lamb - 
    output:
    A - matriz
    '''
    A = []

    umax = 1000000.0
    N = 300000
    aux1 = 3./8.
    aux2 = 7./6.
    aux3 = 23./24.
    
    for l in np.ravel(lamb):
        h = (umax - l)/(N-1)
        u = np.linspace(l, umax, N)
        R = np.sqrt((a**2 + u) + (b**2 + u) + (c**2 + u))
        f = 1./((a**2 + u)*R)
        aij = h*(aux1*f[0] + aux2*f[1] + aux3*f[2] + np.sum(f[3:N-3]) + aux3*f[-3] + aux2*f[-2] + aux1*f[-1])
        A.append(aij)
        
    A = np.array(A).reshape((lamb.shape[0], lamb.shape[1]))
    
    return A
    
def Blambda_simp_ext3(a,b,c,lamb):
    '''
    Integral do potencial utilizando o metodo de simpson.
    input:
    a,b,c - semi-eixos do elipsoide
    lamb - 
    output:
    B - matriz
    '''
    B = []

    umax = 1000000.0
    N = 300000
    aux1 = 3./8.
    aux2 = 7./6.
    aux3 = 23./24.
    
    for l in np.ravel(lamb):
        h = (umax - l)/(N-1)
        u = np.linspace(l, umax, N)
        R = np.sqrt((a**2 + u) + (b**2 + u) + (c**2 + u))
        f = 1./((b**2 + u)*R)
        bij = h*(aux1*f[0] + aux2*f[1] + aux3*f[2] + np.sum(f[3:N-3]) + aux3*f[-3] + aux2*f[-2] + aux1*f[-1])
        B.append(bij)
        
    B = np.array(B).reshape((lamb.shape[0], lamb.shape[1]))
    
    return B    

def Clambda_simp_ext3(a,b,c,lamb):
    '''
    Integral do potencial utilizando o metodo de simpson.
    input:
    a,b,c - semi-eixos do elipsoide
    lamb - 
    output:
    A - constante escalar
    '''
    C = []

    umax = 1000000.0
    N = 300000
    aux1 = 3./8.
    aux2 = 7./6.
    aux3 = 23./24.
    
    for l in np.ravel(lamb):
        h = (umax - l)/(N-1)
        u = np.linspace(l, umax, N)
        R = np.sqrt((a**2 + u) + (b**2 + u) + (c**2 + u))
        f = 1./((c**2 + u)*R)
        cij = h*(aux1*f[0] + aux2*f[1] + aux3*f[2] + np.sum(f[3:N-3]) + aux3*f[-3] + aux2*f[-2] + aux1*f[-1])
        C.append(cij)
        
    C = np.array(C).reshape((lamb.shape[0], lamb.shape[1]))
    
    return C
    
def Dlambda_simp_ext3(a, b, c, lamb):
    '''
    Integral do potencial utilizando o metodo de simpson.
    input:
    a,b,c - semi-eixos do elipsoide
    lamb - 
    output:
    D - constante escalar
    '''
    D = []

    umax = 1000000.0
    N = 300000
    aux1 = 3./8.
    aux2 = 7./6.
    aux3 = 23./24.
    
    for l in np.ravel(lamb):
        h = (umax - l)/(N-1)
        u = np.linspace(l, umax, N)
        R = np.sqrt((a**2 + u) + (b**2 + u) + (c**2 + u))
        f = 1./R
        dij = h*(aux1*f[0] + aux2*f[1] + aux3*f[2] + np.sum(f[3:N-3]) + aux3*f[-3] + aux2*f[-2] + aux1*f[-1])
        D.append(dij)
        
    D = np.array(D).reshape((lamb.shape[0], lamb.shape[1]))
    
    return D

def parametros_integrais(a,b,c,lamb):
    '''
    a: escalar - semi eixo maior
    b: escalar - semi eixo intermediario
    c: escalar - semi eixo menor
    lamb - Maior raiz real da equacao cubica.
    '''
    k = np.zeros_like(lamb)
    k1 = ((a**2-b**2)/(a**2-c**2))**0.5
    k.fill(k1)
    k2 = ((a**2-b**2)/(a**2-c**2))**0.5
    teta_linha = np.arcsin(((a**2-c**2)/(a**2+lamb))**0.5)
    teta_linha2 = np.arccos(c/a)
    F = scipy.special.ellipkinc(teta_linha, k)
    E = scipy.special.ellipeinc(teta_linha, k)
    F2 = scipy.special.ellipkinc(teta_linha2, k2)
    E2 = scipy.special.ellipeinc(teta_linha2, k2)
    return F,E,F2,E2,k,teta_linha
    
def integrais_elipticas(a,b,c,k,teta_linha,F,E):
    '''
    a: escalar - semi eixo maior
    b: escalar - semi eixo intermediario
    c: escalar - semi eixo menor
    k: matriz - parametro de geometria
    teta_linha: matriz - parametro de geometria
    F: matriz - integrais normais elipticas de primeiro tipo
    E: matriz - integrais normais elipticas de segundo tipo
    '''
    A2 = (2/((a**2-b**2)*(a**2-c**2)**0.5))*(F-E)
    B2 = ((2*(a**2-c**2)**0.5)/((a**2-b**2)*(b**2-c**2)))*(E-((b**2-c**2)/(a**2-c**2))*F-((k**2*np.sin(teta_linha)*np.cos(teta_linha))/(1-k**2*np.sin(teta_linha)*np.sin(teta_linha))**0.5))
    C2 = (2/((b**2-c**2)*(a**2-c**2)**0.5))*(((np.sin(teta_linha)*((1-k**2*np.sin(teta_linha)*np.sin(teta_linha))**0.5))/np.cos(teta_linha))-E)
    return A2,B2,C2
