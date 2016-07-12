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
    def __init__(self, xp, yp, zp, xc, yc, zc, a, b, c, alfa, delta, gamma, props):
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
        
        self.xp = (xp)
        self.yp = (yp)
        self.zp = (zp)
        
        self.l1 = self.l1_v()
        self.l2 = self.l2_v()
        self.l3 = self.l3_v()
        self.m1 = self.m1_v()
        self.m2 = self.m2_v()
        self.m3 = self.m3_v()
        self.n1 = self.n1_v()
        self.n2 = self.n2_v()
        self.n3 = self.n3_v()
            
        self.ln = np.cos(self.props['remanence'][2])*np.cos(self.props['remanence'][1])
        self.mn = np.sin(self.props['remanence'][2])*np.cos(self.props['remanence'][1])
        self.nn = np.sin(self.props['remanence'][1])
            
        self.mcon = np.array([[self.l1, self.m1, self.n1],[self.l2, self.m2, self.n2],[self.l3, self.m3, self.n3]])
        self.mconT = (self.mcon).T
        self.k_dec = np.array([[props['k1'][2]],[props['k2'][2]],[props['k3'][2]]])
        self.k_int = np.array([[props['k1'][0]],[props['k2'][0]],[props['k3'][0]]])
        self.k_inc = np.array([[props['k1'][1]],[props['k2'][1]],[props['k3'][1]]])
            
        if self.k_int[0] == self.k_int[1] and self.k_int[0] == self.k_int[2]:
            self.km = self.k_matrix2 ()
        else:
            self.Lr = self.Lr_v ()
            self.Mr = self.Mr_v ()
            self.Nr = self.Nr_v ()
            self.km = self.k_matrix ()
                
        self.x1 = self.x1_e()
        self.x2 = self.x2_e()
        self.x3 = self.x3_e()
        self.p0 = self.p0_e()
        self.p1 = self.p1_e()
        self.p2 = self.p2_e()
        self.p = self.p_e()
        self.q = self.q_e()
        self.teta = self.teta_e()
        self.lamb = self.lamb_e()
            
        self.F,self.E,self.F2,self.E2,self.k,self.teta_linha = self.parametros_integrais()
        self.JN = self.JN_e ()
        self.N1,self.N2,self.N3 = self.N_desmag ()
        
    def __str__(self):
        """Return a string representation of the ellipsoids."""
        names = [('xc', self.xc), ('yc', self.yc), ('zc', self.zc),
                 ('a', self.a), ('b', self.b), ('c', self.c),
                 ('alfa', self.alfa),('delta', self.delta),('gamma', self.gamma)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)
        
    def l1_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo x.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        
        output:
        Direcao em radianos.
        '''
        
        l1 = (-np.cos(self.alfa)*np.cos(self.delta))
        return l1
        
    def l2_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo y.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
        
        output:
        Direcao em radianos.
        '''
        
        l2 = (np.cos(self.alfa)*np.cos(self.gamma)*np.sin(self.delta)+np.sin(self.alfa)*np.sin(self.gamma))
        return l2

    def l3_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo z.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
        
        output:
        Direcao em radianos.
        '''
        
        l3 = (np.sin(self.alfa)*np.cos(self.gamma)-np.cos(self.alfa)*np.sin(self.gamma)*np.sin(self.delta))
        return l3

    def m1_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo x.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        
        output:
        Direcao em radianos.
        '''
        
        m1 = (-np.sin(self.alfa)*np.cos(self.delta))
        return m1

    def m2_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo y.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
        
        output:
        Direcao em radianos.
        '''
        
        m2 = (np.sin(self.alfa)*np.cos(self.gamma)*np.sin(self.delta)-np.cos(self.alfa)*np.sin(self.gamma))
        return m2

    def m3_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo z.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
        
        output:
        Direcao em radianos.
        '''
        
        m3 = (-np.cos(self.alfa)*np.cos(self.gamma)-np.sin(self.alfa)*np.sin(self.gamma)*np.sin(self.delta))
        return m3

    def n1_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo x.
        
        input:
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)

        output:
        Direcao em radianos.
        '''
        
        n1 = (-np.sin(self.delta))
        return n1

    def n2_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo y.
        
        input:
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
        
        output:
        Direcao em radianos.
        '''
        
        n2 = (-np.cos(self.gamma)*np.cos(self.delta))
        return n2

    def n3_v (self):

        '''
        Orientacao do elipsoide com respeito ao eixo z.
        
        input:
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
        
        output:
        Direcao em radianos.
        '''
        
        n3 = (np.sin(self.gamma)*np.cos(self.delta))
        return n3
    
    def x1_e (self):
        '''
        Calculo da coordenada x no elipsoide
        input:
        xp,yp - Matriz: Coordenadas geograficas (malha).
        h - Profundidade do elipsoide.
        l1,m1,n1 - Orientacao do elipsoide (eixo x)
        output:
        x1 - Coordenada x do elipsoide.
        '''
        x1 = (self.xp-self.xc)*self.l1+(self.yp-self.yc)*self.m1+(self.zp-self.zc)*self.n1
        return x1
    
    def x2_e (self):
        '''
        Calculo da coordenada y no elipsoide
        input:
        xp,yp - Matriz: Coordenadas geograficas (malha).
        h - Profundidade do elipsoide.
        l2,m2,n2 - Orientacao do elipsoide (eixo y).
        output:
        x2 - Coordenada y do elipsoide.
        '''
        x2 = (self.xp-self.xc)*self.l2+(self.yp-self.yc)*self.m2+(self.zp-self.zc)*self.n2
        return x2

    def x3_e (self):
        '''
        Calculo da coordenada z no elipsoide
        input:
        xp,yp - Matriz: Coordenadas geograficas (malha).
        h - Profundidade do elipsoide.
        l3,m3,n3 - Orientacao do elipsoide (eixo z).
        output:
        x3 - Coordenada z do elipsoide.
        '''
        x3 = (self.xp-self.xc)*self.l3+(self.yp-self.yc)*self.m3+(self.zp-self.zc)*self.n3
        return x3
        
    def Lr_v (self):
        '''
        Cossenos diretores dos eixos dos vetores de susceptibilidade magnetica.
        
        input:
        k_dec - declinacoes dos vetores de susceptibilidade.
        k_inc - inclinacoes dos vetores de susceptibilidade.
        '''
        Lr = np.zeros(3)
        for i in range (3):
            Lr[i] = np.cos(self.k_dec[i])*np.cos(self.k_inc[i])
        return Lr
        
    def Mr_v (self):
        '''
        Cossenos diretores dos eixos dos vetores de susceptibilidade magnetica.
        
        input:
        k_dec - declinacoes dos vetores de susceptibilidade.
        k_inc - inclinacoes dos vetores de susceptibilidade.
        '''
        Mr = np.zeros(3)
        for i in range (3):
            Mr[i] = np.sin(self.k_dec[i])*np.cos(self.k_inc[i])
        return Mr
        
    def Nr_v (self):
        '''
        Cossenos diretores dos eixos dos vetores de susceptibilidade magnetica.
        
        input:
        k_inc - inclinacoes dos vetores de susceptibilidade.
        '''
        Nr = np.zeros(3)
        for i in range (3):
            Nr[i] = np.sin(self.k_inc[i])
        return Nr

    def JN_e (self):
        '''
        transformacao do Vetor de magnetizacao remanente para as coordenadas nos eixos do elipsoide.
        '''
        JN = self.props['remanence'][0]*np.ravel(np.array([[(self.ln*self.l1+self.mn*self.m1+self.nn*self.n1)], [(self.ln*self.l2+self.mn*self.m2+self.nn*self.n2)], [(self.ln*self.l3+self.mn*self.m3+self.nn*self.n3)]]))
        return JN

    def N_desmag (self):
        '''
        Fator de desmagnetizacao ao longo do eixo de revolucao (N1) e em relacao ao plano equatorial (N2).
        '''
        N1 = ((4.*np.pi*self.a*self.b*self.c)/((self.a**2-self.b**2)*(self.a**2-self.c**2)**0.5)) * (self.F2-self.E2)
        N2 = (((4.*np.pi*self.a*self.b*self.c)*(self.a**2-self.c**2)**0.5)/((self.a**2-self.b**2)*(self.b**2-self.c**2))) * (self.E2 - ((self.b**2-self.c**2)/(self.a**2-self.c**2)) * self.F2 - ((self.c*(self.a**2-self.b**2))/(self.a*self.b*(self.a**2-self.c**2)**0.5)))
        N3 = ((4.*np.pi*self.a*self.b*self.c)/((self.b**2-self.c**2)*(self.a**2-self.c**2)**0.5)) * (((self.b*(self.a**2-self.c**2)**0.5)/(self.a*self.c)) - self.E2)
        return N1, N2, N3
    
    def k_matrix (self):
        '''
        Matriz de tensores de susceptibilidade.
        '''
        l = np.array([[self.l1],[self.l2],[self.l3]])
        m = np.array([[self.m1],[self.m2],[self.m3]])
        n = np.array([[self.n1],[self.n2],[self.n3]])
        k = np.zeros([3,3])
        for i in range (3):
            for j in range (3):
                for r in range (3):
                    k[i,j] = k[i,j] + (self.k_int[r]*(self.Lr[r]*l[i] + self.Mr[r]*m[i] + self.Nr[r]*n[i])*(self.Lr[r]*l[j] + self.Mr[r]*m[j] + self.Nr[r]*n[j]))
        return k
        
    def k_matrix2 (self):
        '''
        Matriz de tensores de susceptibilidade.
        '''
        l = np.array([[self.l1],[self.l2],[self.l3]])
        m = np.array([[self.m1],[self.m2],[self.m3]])
        n = np.array([[self.n1],[self.n2],[self.n3]])
        k = np.zeros([3,3])
        for i in range (3):
            for j in range (3):
                for r in range (3):
                    k[i,j] = k[i,j] + (self.k_int[r]*(l[r]*l[i] + m[r]*m[i] + n[r]*n[i])*(l[r]*l[j] + m[r]*m[j] + n[r]*n[j]))
        return k
    
    # Calculos auxiliares
    def p0_e (self):
        '''
        Constante da equacao cubica: s^3 + p2*s^2 + p0 = 0
        input:
        a,b,c - Eixos do elipsoide.
        x1,x2,x3 - Eixo de coordenadas do elipsoide.
        output:
        p0 - Constante
        '''
        p0 = (self.a*self.b*self.c)**2-(self.b*self.c*self.x1)**2-(self.c*self.a*self.x2)**2-(self.a*self.b*self.x3)**2
        return p0
        
    def p1_e (self):
        '''
        Constante da equacao cubica: s^3 + p2*s^2 + p0 = 0
        input:
        a,b,c - Eixos do elipsoide.
        x1,x2,x3 - Eixo de coordenadas do elipsoide.
        output:
        p0 - Constante
        '''
        p1 = (self.a*self.b)**2+(self.b*self.c)**2+(self.c*self.a)**2-(self.b**2+self.c**2)*self.x1**2-(self.c**2+self.a**2)*self.x2**2-(self.a**2+self.b**2)*self.x3**2
        return p1
        
    def p2_e (self):
        '''
        Constante da equacao cubica: s^3 + p2*s^2 + p0 = 0
        input:
        a,b,c - Eixos do elipsoide.
        x1,x2,x3 - Eixo de coordenadas do elipsoide.
        output:
        p0 - Constante
        '''
        p2 = self.a**2+self.b**2+self.c**2-self.x1**2-self.x2**2-self.x3**2
        return p2
        
    def p_e (self):
        '''
        Constante
        input:
        p1,p2 - constantes da equacao cubica
        output:
        p - Constante.
        '''
        p = self.p1-(self.p2**2)/3.
        return p
        
    def q_e (self):
        '''
        Constante
        input:
        p0,p1,p2 - constantes da equacao cubica
        output:
        q - Constante.
        '''
        q = self.p0-((self.p1*self.p2)/3.)+2*(self.p2/3.)**3
        return q
        
    def teta_e (self):
        '''
        Constante angular (radianos)
        input:
        p - constante da equacao cubica
        q - constante
        output:
        teta - Constante.
        '''
        teta = np.arccos(-self.q/(2*np.sqrt((-self.p/3.)**3)))
        return teta
        
    def lamb_e (self):
        '''
        Maior raiz real da equacao cubica: s^3 + p2*s^2 + p0 = 0
        input:
        p,p2 - constantes da equacao cubica
        teta - constante angular (radianos)
        output:
        lamb - Maior raiz real.
        '''
        lamb = 2.*((-self.p/3.)**0.5)*np.cos(self.teta/3.)-(self.p2/3.)
        return lamb
        
    def parametros_integrais(self):
        '''
        a: escalar - semi eixo maior
        b: escalar - semi eixo intermediario
        c: escalar - semi eixo menor
        lamb - Maior raiz real da equacao cubica.
        '''
        k = np.zeros_like(self.lamb)
        k1 = ((self.a**2-self.b**2)/(self.a**2-self.c**2))**0.5
        k.fill(k1)
        k2 = ((self.a**2-self.b**2)/(self.a**2-self.c**2))**0.5
        teta_linha = np.arcsin(((self.a**2-self.c**2)/(self.a**2+self.lamb))**0.5)
        teta_linha2 = np.arccos(self.c/self.a)
        F = scipy.special.ellipkinc(teta_linha, k)
        E = scipy.special.ellipeinc(teta_linha, k)
        F2 = scipy.special.ellipkinc(teta_linha2, k2)
        E2 = scipy.special.ellipeinc(teta_linha2, k2)
        return F,E,F2,E2,k,teta_linha     

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
    
    # Calculo do vetor de magnetizacao resultante
    lt = ln_v (dec, inc)
    mt = mn_v (dec, inc)
    nt = nn_v (inc)
    Ft = F_e (inten,lt,mt,nt,ellipsoids.l1,ellipsoids.l2,ellipsoids.l3,ellipsoids.m1,ellipsoids.m2,ellipsoids.m3,ellipsoids.n1,ellipsoids.n2,ellipsoids.n3)
    JR = JR_e (ellipsoids.km,ellipsoids.JN,Ft)
    JRD = JRD_e (ellipsoids.km,ellipsoids.N1,ellipsoids.N2,ellipsoids.N3,JR)
    JRD_carte = (ellipsoids.mconT).dot(JRD)
    JRD_ang = utils.vec2ang(JRD_carte)
    #print JRD_ang
        
    # Derivadas de lambda em relacao as posicoes
    dlambx1 = dlambx1_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,ellipsoids.x1,ellipsoids.x2,ellipsoids.x3,ellipsoids.lamb)
    dlambx2 = dlambx2_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,ellipsoids.x1,ellipsoids.x2,ellipsoids.x3,ellipsoids.lamb)
    dlambx3 = dlambx3_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,ellipsoids.x1,ellipsoids.x2,ellipsoids.x3,ellipsoids.lamb)
    #print dlambx1,dlambx2,dlambx3
    
    # Calculo das integrais
    A, B, C = integrais_elipticas(ellipsoids.a,ellipsoids.b,ellipsoids.c,ellipsoids.k,ellipsoids.teta_linha,ellipsoids.F,ellipsoids.E)
    
    # Geometria para o calculo de B (eixo do elipsoide)
    cte = cte_m (ellipsoids.a,ellipsoids.b,ellipsoids.c,ellipsoids.lamb)
    V1, V2, V3 = v_e (ellipsoids.a,ellipsoids.b,ellipsoids.c,ellipsoids.x1,ellipsoids.x2,ellipsoids.x3,ellipsoids.lamb)
    
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
    
    # Problema Direto (Calcular o campo externo nas coordenadas geograficas)
    Bx = Bx_c (B1,B2,B3,ellipsoids.l1,ellipsoids.l2,ellipsoids.l3)
    By = By_c (B1,B2,B3,ellipsoids.m1,ellipsoids.m2,ellipsoids.m3)
    Bz = Bz_c (B1,B2,B3,ellipsoids.n1,ellipsoids.n2,ellipsoids.n3)
    
    return Bx,By,Bz
    
    # Problema Direto (Calcular o campo externo e anomalia nas coordenadas geograficas no SI)
def bx_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1
    
    for i in range(len(ellipsoids)):
        bx,by,bz = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
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
        bx,by,bz = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
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
        bx,by,bz = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
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
        bx,by,bz = elipsoide (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        tf = bx*np.cos(inc)*np.cos(dec) + by*np.cos(inc)*np.sin(dec) + bz*np.sin(inc)
        res += tf
    res = res*ctemag
    return res
    
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
    
def F_e (intensidadeT,lt,mt,nt,l1,l2,l3,m1,m2,m3,n1,n2,n3):
    '''
    Transformacao do vetor campo magnetico da Terra para as coordenadas nos eixos do elipsoide.
    '''
    Ft = intensidadeT*np.ravel(np.array([[(lt*l1+mt*m1+nt*n1)], [(lt*l2+mt*m2+nt*n2)], [(lt*l3+mt*m3+nt*n3)]]))
    return Ft
    
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