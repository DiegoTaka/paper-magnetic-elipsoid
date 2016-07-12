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
    def __init__(self, xp, yp, zp, xc, yc, zc, a, b, c, azimuth, delta, gamma, props):
        GeometricElement.__init__(self, props)
        
        self.center = np.array([xc,yc,zc])        
        self.axis = np.array([a,b,c])        
        self.xp = (xp)
        self.yp = (yp)
        self.zp = (zp)
        self.conf = []
        
        if self.axis[0] > self.axis[1] and self.axis[1] > self.axis[2]:
            self.angles = np.array([azimuth+np.pi,delta,gamma])
            self.mcon,self.mconT = self.m_convT()
            self.conf.append('Triaxial')
        if self.axis[1] == self.axis[2] and self.axis[0] > self.axis[1]:
            self.angles = np.array([azimuth+np.pi,delta,gamma])
            self.mcon,self.mconT = self.m_convP()
            self.conf.append('Prolate')
        if self.axis[1] == self.axis[2] and self.axis[0] < self.axis[1]:
            self.angles = np.array([azimuth,delta,gamma])
            self.mcon,self.mconT = self.m_convO()
            self.conf.append('Oblate')
            
        self.ln = np.cos(self.props['remanence'][2])*np.cos(self.props['remanence'][1])
        self.mn = np.sin(self.props['remanence'][2])*np.cos(self.props['remanence'][1])
        self.nn = np.sin(self.props['remanence'][1])

        self.k_dec = np.array([[props['k1'][2]],[props['k2'][2]],[props['k3'][2]]])
        self.k_int = np.array([[props['k1'][0]],[props['k2'][0]],[props['k3'][0]]])
        self.k_inc = np.array([[props['k1'][1]],[props['k2'][1]],[props['k3'][1]]])
            
        if self.k_int[0] == (self.k_int[1] and self.k_int[2]):
            self.km = self.k_matrix()
            self.conf.append('Isotropic magnetization')
        else:
            self.km = self.k_matrix2()
            self.conf.append('Anysotropic magnetization')
                
        self.x1,self.x2,self.x3 = self.x_e()
            
        self.JN = self.JN_e ()
        
        if self.axis[0] > self.axis[1] and self.axis[1] > self.axis[2]:
            self.lamb,self.teta,self.q,self.p,self.p2,self.p1,self.p0 = self.lamb_T()
            self.dlambx1,self.dlambx2,self.dlambx3 = self.dlambx_T()
            self.F,self.E,self.F2,self.E2,self.k,self.teta_linha = self.parametros_integrais()
            self.N1,self.N2,self.N3 = self.N_desmagT()
            self.A, self.B, self.C = self.integrais_elipticas()
            self.m11,self.m12,self.m13,self.m21,self.m22,self.m23,self.m31,self.m32,self.m33, self.cte, self.V1, self.V2, self.V3 = self.mx()
     
        if self.axis[1] == self.axis[2] and self.axis[0] > self.axis[1]:
            self.N1,self.N2,self.N3 = self.N_desmagP()
            self.r = self.r_e ()
            self.delta = self.delta_e ()
            self.lamb = self.lamb_PO ()
            self.dlambx1,self.dlambx2,self.dlambx3 = self.dlambx_PO()
            
        if self.axis[1] == self.axis[2] and self.axis[0] < self.axis[1]:
            self.N1,self.N2,self.N3 = self.N_desmagO()
            self.r = self.r_e ()
            self.delta = self.delta_e ()
            self.lamb = self.lamb_PO ()
            self.dlambx1,self.dlambx2,self.dlambx3 = self.dlambx_PO()
        
    def __str__(self):
        """Return a string representation of the ellipsoids."""
        names = [('xc', self.self.center[0]), ('yc', self.self.center[1]), ('zc', self.self.center[2]),
                 ('a', self.axis[0]), ('b', self.axis[1]), ('c', self.axis[2]),
                 ('alpha', self.angles[0]),('delta', self.angles[1]),('gamma', self.angles[2])]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)
        
    def m_convT (self):

        '''
        Orientacao do elipsoide com respeito ao eixo x.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        
        output:
        Direcao em radianos.
        '''
        mcon = np.zeros((3,3))
        mcon[0][0] = (-np.cos(self.angles[0])*np.cos(self.angles[1]))
        mcon[1][0] = (np.cos(self.angles[0])*np.cos(self.angles[2])*np.sin(self.angles[1])+np.sin(self.angles[0])*np.sin(self.angles[2]))
        mcon[2][0] = (np.sin(self.angles[0])*np.cos(self.angles[2])-np.cos(self.angles[0])*np.sin(self.angles[2])*np.sin(self.angles[1]))
        mcon[0][1] = (-np.sin(self.angles[0])*np.cos(self.angles[1]))
        mcon[1][1] = (np.sin(self.angles[0])*np.cos(self.angles[2])*np.sin(self.angles[1])-np.cos(self.angles[0])*np.sin(self.angles[2]))
        mcon[2][1] = (-np.cos(self.angles[0])*np.cos(self.angles[2])-np.sin(self.angles[0])*np.sin(self.angles[2])*np.sin(self.angles[1]))
        mcon[0][2] = (-np.sin(self.angles[1]))
        mcon[1][2] = (-np.cos(self.angles[2])*np.cos(self.angles[1]))
        mcon[2][2] = (np.sin(self.angles[2])*np.cos(self.angles[1]))
        mconT = (mcon).T
        return mcon, mconT
        
    def m_convP (self):

        '''
        Orientacao do elipsoide com respeito ao eixo x.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        
        output:
        Direcao em radianos.
        '''
        mcon = np.zeros((3,3))
        mcon[0][0] = (-np.cos(self.angles[0])*np.cos(self.angles[1]))
        mcon[1][0] = (np.cos(self.angles[0])*np.sin(self.angles[1]))
        mcon[2][0] = (np.sin(self.angles[0]))
        mcon[0][1] = (-np.sin(self.angles[0])*np.cos(self.angles[1]))
        mcon[1][1] = (np.sin(self.angles[0])*np.cos(self.angles[1]))
        mcon[2][1] = (-np.cos(self.angles[0]))
        mcon[0][2] = (-np.sin(self.angles[1]))
        mcon[1][2] = (-np.cos(self.angles[2]))
        mcon[2][2] = (0)
        mconT = (mcon).T
        return mcon, mconT

    def m_convO (self):

        '''
        Orientacao do elipsoide com respeito ao eixo x.
        
        input:
        alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
        delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
        
        output:
        Direcao em radianos.
        '''
        mcon = np.zeros((3,3))
        mcon[0][0] = (np.cos(self.angles[0])*np.sin(self.angles[1]))
        mcon[1][0] = (-np.cos(self.angles[0])*np.cos(self.angles[1]))
        mcon[2][0] = (-np.sin(self.angles[0]))
        mcon[0][1] = (np.sin(self.angles[0])*np.sin(self.angles[1]))
        mcon[1][1] = (-np.sin(self.angles[0])*np.cos(self.angles[1]))
        mcon[2][1] = (np.cos(self.angles[0]))
        mcon[0][2] = (-np.cos(self.angles[1]))
        mcon[1][2] = (-np.sin(self.angles[1]))
        mcon[2][2] = (0)
        mconT = (mcon).T
        return mcon, mconT
        
    def x_e (self):
        '''
        Calculo da coordenada x no elipsoide
        input:
        xp,yp - Matriz: Coordenadas geograficas (malha).
        h - Profundidade do elipsoide.
        l1,m1,n1 - Orientacao do elipsoide (eixo x)
        output:
        x1 - Coordenada x do elipsoide.
        '''
        x1 = (self.xp-self.center[0])*self.mcon[0,0]+(self.yp-self.center[1])*self.mcon[0,1]+(self.zp-self.center[2])*self.mcon[0,2]
        x2 = (self.xp-self.center[0])*self.mcon[1,0]+(self.yp-self.center[1])*self.mcon[1,1]+(self.zp-self.center[2])*self.mcon[1,2]
        x3 = (self.xp-self.center[0])*self.mcon[2,0]+(self.yp-self.center[1])*self.mcon[2,1]+(self.zp-self.center[2])*self.mcon[2,2]
        return x1, x2, x3
        
    def JN_e (self):
        '''
        transformacao do Vetor de magnetizacao remanente para as coordenadas nos eixos do elipsoide.
        '''
        JN = self.props['remanence'][0]*np.ravel(np.array([[(self.ln*self.mcon[0,0]+self.mn*self.mcon[0,1]+self.nn*self.mcon[0,2])], [(self.ln*self.mcon[1,0]+self.mn*self.mcon[1,1]+self.nn*self.mcon[1,2])], [(self.ln*self.mcon[2,0]+self.mn*self.mcon[2,1]+self.nn*self.mcon[2,2])]]))
        return JN

    def N_desmagT (self):
        '''
        Fator de desmagnetizacao ao longo do eixo de revolucao (N1) e em relacao ao plano equatorial (N2).
        '''
        N1 = ((4.*np.pi*self.axis[0]*self.axis[1]*self.axis[2])/((self.axis[0]**2-self.axis[1]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5)) * (self.F2-self.E2)
        N2 = (((4.*np.pi*self.axis[0]*self.axis[1]*self.axis[2])*(self.axis[0]**2-self.axis[2]**2)**0.5)/((self.axis[0]**2-self.axis[1]**2)*(self.axis[1]**2-self.axis[2]**2))) * (self.E2 - ((self.axis[1]**2-self.axis[2]**2)/(self.axis[0]**2-self.axis[2]**2)) * self.F2 - ((self.axis[2]*(self.axis[0]**2-self.axis[1]**2))/(self.axis[0]*self.axis[1]*(self.axis[0]**2-self.axis[2]**2)**0.5)))
        N3 = ((4.*np.pi*self.axis[0]*self.axis[1]*self.axis[2])/((self.axis[1]**2-self.axis[2]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5)) * (((self.axis[1]*(self.axis[0]**2-self.axis[2]**2)**0.5)/(self.axis[0]*self.axis[2])) - self.E2)
        return N1, N2, N3
        
    def N_desmagP (self):
        '''
        Fator de desmagnetizacao ao longo do eixo de revolucao (N1) e em relacao ao plano equatorial (N2).
        '''
        N1 = ((4.*np.pi*self.axis[0]*self.axis[1]**2)/((self.axis[0]**2-self.axis[1]**2)**1.5)) * (np.log(((self.axis[0]**2/self.axis[1]**2)-1.)**0.5 + (self.axis[0]/self.axis[1])) - (1. - (self.axis[1]**2/self.axis[0]**2))**0.5)
        N2 = 2.*np.pi - N1/2
        N3 = N2
        return N1, N2, N3
        
    def N_desmagO (self):
        '''
        Fator de desmagnetizacao ao longo do eixo de revolucao (N1) e em relacao ao plano equatorial (N2).
        '''
        N1 = ((4.*np.pi*self.axis[0]*self.axis[1]**2)/((self.axis[1]**2-self.axis[0]**2)**1.5)) * ((((self.axis[1]**2-self.axis[0]**2)**0.5)/(self.axis[0])) - np.arctan(((self.axis[1]**2-self.axis[0]**2)**0.5)/(self.axis[0])))
        N2 = 2.*np.pi - N1/2.
        N3 = N2
        return N1, N2, N3

    def k_matrix (self):
        '''
        Matriz de tensores de susceptibilidade.
        '''
        km = np.zeros([3,3])
        for i in range (3):
            for j in range (3):
                for r in range (3):
                    km[i,j] = km[i,j] + (self.k_int[r]*(self.mcon[r,0]*self.mcon[i,0] + self.mcon[r,1]*self.mcon[i,1] + self.mcon[r,2]*self.mcon[i,2])*(self.mcon[r,0]*self.mcon[j,0] + self.mcon[r,1]*self.mcon[j,1] + self.mcon[r,2]*self.mcon[j,2]))
        return km

    def k_matrix2 (self):
        '''
        Matriz de tensores de susceptibilidade.
        '''
        Lr = np.zeros(3)
        Mr = np.zeros(3)
        Nr = np.zeros(3)
        for i in range (3):
            Lr[i] = np.cos(self.k_dec[i])*np.cos(self.k_inc[i])
            Mr[i] = np.sin(self.k_dec[i])*np.cos(self.k_inc[i])
            Nr[i] = np.sin(self.k_inc[i])
        km = np.zeros([3,3])
        for i in range (3):
            for j in range (3):
                for r in range (3):
                    km[i,j] = km[i,j] + (self.k_int[r]*(Lr[r]*self.mcon[i,0] + Mr[r]*self.mcon[i,1] + Nr[r]*self.mcon[i,2])*(Lr[r]*self.mcon[j,0] + Mr[r]*self.mcon[j,1] + Nr[r]*self.mcon[j,2]))
        return km

    def lamb_T (self):
        '''
        Maior raiz real da equacao cubica: s^3 + p2*s^2 + p0 = 0
        input:
        p,p2 - constantes da equacao cubica
        teta - constante angular (radianos)
        output:
        lamb - Maior raiz real.
        '''
        p0 = (self.axis[0]*self.axis[1]*self.axis[2])**2-(self.axis[1]*self.axis[2]*self.x1)**2-(self.axis[2]*self.axis[0]*self.x2)**2-(self.axis[0]*self.axis[1]*self.x3)**2
        p1 = (self.axis[0]*self.axis[1])**2+(self.axis[1]*self.axis[2])**2+(self.axis[2]*self.axis[0])**2-(self.axis[1]**2+self.axis[2]**2)*self.x1**2-(self.axis[2]**2+self.axis[0]**2)*self.x2**2-(self.axis[0]**2+self.axis[1]**2)*self.x3**2
        p2 = self.axis[0]**2+self.axis[1]**2+self.axis[2]**2-self.x1**2-self.x2**2-self.x3**2
        p = p1-(p2**2)/3.
        q = p0-((p1*p2)/3.)+2*(p2/3.)**3
        teta = np.arccos(-q/(2*np.sqrt((-p/3.)**3)))
        lamb = 2.*((-p/3.)**0.5)*np.cos(teta/3.)-(p2/3.)
        return lamb, teta, q, p, p2, p1, p0
    
    def r_e (self):
        '''
        Distancia do centro do elipsoide ao ponto observado
        input:
        X,Y - Matriz: Coordenadas geograficas (malha).
        h - Profundidade do elipsoide.
        l3,m3,n3 - Orientacao do elipsoide (eixo z).
        output:
        x3 - Coordenada z do elipsoide.
        '''
        r = ((self.x1)**2+(self.x2)**2+(self.x3)**2)**0.5
        return r
        
    def delta_e (self):
        '''
        Calculo auxiliar de lambda.
        input:
        X,Y - Matriz: Coordenadas geograficas (malha).
        h - Profundidade do elipsoide.
        l3,m3,n3 - Orientacao do elipsoide (eixo z).
        output:
        x3 - Coordenada z do elipsoide.
        '''

        delta = (self.r**4 + (self.axis[0]**2-self.axis[1]**2)**2 - 2*(self.axis[0]**2-self.axis[1]**2) * (self.x1**2 - self.x2**2 - self.x3**2))**0.5
        return delta    
        
    def lamb_PO (self):
        '''
        Maior raiz real da equacao cubica: s^3 + p2*s^2 + p0 = 0
        input:
        p,p2 - constantes da equacao cubica
        teta - constante angular (radianos)
        output:
        lamb - Maior raiz real.
        '''
        lamb = (self.r**2 - self.axis[0]**2 - self.axis[1]**2 + self.delta)/2.
        return lamb
        
    def parametros_integrais(self):
        '''
        a: escalar - semi eixo maior
        b: escalar - semi eixo intermediario
        c: escalar - semi eixo menor
        lamb - Maior raiz real da equacao cubica.
        '''
        k = np.zeros_like(self.lamb)
        k1 = ((self.axis[0]**2-self.axis[1]**2)/(self.axis[0]**2-self.axis[2]**2))**0.5
        k.fill(k1)
        teta_linha = np.arcsin(((self.axis[0]**2-self.axis[2]**2)/(self.axis[0]**2+self.lamb))**0.5)
        teta_linha2 = np.arccos(self.axis[2]/self.axis[0])
        F = scipy.special.ellipkinc(teta_linha, k)
        E = scipy.special.ellipeinc(teta_linha, k)
        F2 = scipy.special.ellipkinc(teta_linha2, k1)
        E2 = scipy.special.ellipeinc(teta_linha2, k1)
        return F,E,F2,E2,k,teta_linha

    def dlambx_T (self):
        '''
        Derivada de lamb em relacao ao eixo x3 do elipsoide.
        input:
        a,b,c, - semi-eixos do elipsoide.
        x1,x2,x3 - Eixo de coordenadas do elipsoide.
        lamb - Maior raiz real da equacao cubica.
        output:
        dlambx3 - escalar
        '''
        dlambx1 = (2*self.x1/(self.axis[0]**2+self.lamb))/((self.x1/(self.axis[0]**2+self.lamb))**2+(self.x2/(self.axis[1]**2+self.lamb))**2+((self.x3/(self.axis[2]**2+self.lamb))**2))        
        dlambx2 = (2*self.x2/(self.axis[1]**2+self.lamb))/((self.x1/(self.axis[0]**2+self.lamb))**2+(self.x2/(self.axis[1]**2+self.lamb))**2+((self.x3/(self.axis[2]**2+self.lamb))**2))
        dlambx3 = (2*self.x3/(self.axis[2]**2+self.lamb))/((self.x1/(self.axis[0]**2+self.lamb))**2+(self.x2/(self.axis[1]**2+self.lamb))**2+((self.x3/(self.axis[2]**2+self.lamb))**2))
        return dlambx1, dlambx2, dlambx3

    def dlambx_PO (self):
        '''
        Derivada de lamb em relacao ao eixo x3 do elipsoide.
        input:
        a,b,c, - semi-eixos do elipsoide.
        x1,x2,x3 - Eixo de coordenadas do elipsoide.
        lamb - Maior raiz real da equacao cubica.
        output:
        dlambx3 - escalar
        '''
        dlambx1 = self.x1*(1+(self.r**2-self.axis[0]**2+self.axis[1]**2)/self.delta)
        dlambx2 = self.x2*(1+(self.r**2+self.axis[0]**2-self.axis[1]**2)/self.delta)
        dlambx3 = self.x3*(1+(self.r**2+self.axis[0]**2-self.axis[1]**2)/self.delta)
        return dlambx1, dlambx2, dlambx3

    def integrais_elipticas(self):
        '''
        a: escalar - semi eixo maior
        b: escalar - semi eixo intermediario
        c: escalar - semi eixo menor
        k: matriz - parametro de geometria
        teta_linha: matriz - parametro de geometria
        F: matriz - integrais normais elipticas de primeiro tipo
        E: matriz - integrais normais elipticas de segundo tipo
        '''
        A2 = (2/((self.axis[0]**2-self.axis[1]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5))*(self.F-self.E)
        B2 = ((2*(self.axis[0]**2-self.axis[2]**2)**0.5)/((self.axis[0]**2-self.axis[1]**2)*(self.axis[1]**2-self.axis[2]**2)))*(self.E-((self.axis[1]**2-self.axis[2]**2)/(self.axis[0]**2-self.axis[2]**2))*self.F-((self.k**2*np.sin(self.teta_linha)*np.cos(self.teta_linha))/(1-self.k**2*np.sin(self.teta_linha)*np.sin(self.teta_linha))**0.5))
        C2 = (2/((self.axis[1]**2-self.axis[2]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5))*(((np.sin(self.teta_linha)*((1-self.k**2*np.sin(self.teta_linha)*np.sin(self.teta_linha))**0.5))/np.cos(self.teta_linha))-self.E)
        return A2,B2,C2

    def mx(self):
        '''
        '''
        cte = 1/np.sqrt((self.axis[0]**2+self.lamb)*(self.axis[1]**2+self.lamb)*(self.axis[2]**2+self.lamb))
        V1 = self.x1/(self.axis[0]**2+self.lamb)
        V2 = self.x2/(self.axis[1]**2+self.lamb)
        V3 = self.x3/(self.axis[2]**2+self.lamb)
        m11 = (cte*self.dlambx1*V1) - self.A
        m12 = cte*self.dlambx1*V2
        m13 = cte*self.dlambx1*V3
        m21 = cte*self.dlambx2*V1
        m22 = (cte*self.dlambx2*V2) - self.B
        m23 = cte*self.dlambx2*V3
        m31 = cte*self.dlambx3*V1
        m32 = cte*self.dlambx3*V2
        m33 = (cte*self.dlambx3*V3) - self.C
        return m11, m12, m13, m21, m22, m23, m31, m32, m33, cte, V1, V2, V3

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
    Ft = F_e (inten,lt,mt,nt,ellipsoids.mcon[0,0],ellipsoids.mcon[1,0],ellipsoids.mcon[2,0],ellipsoids.mcon[0,1],ellipsoids.mcon[1,1],ellipsoids.mcon[2,1],ellipsoids.mcon[0,2],ellipsoids.mcon[1,2],ellipsoids.mcon[2,2])
    JR = JR_e (ellipsoids.km,ellipsoids.JN,Ft)
    JRD = JRD_e (ellipsoids.km,ellipsoids.N1,ellipsoids.N2,ellipsoids.N3,JR)
    JRD_carte = (ellipsoids.mconT).dot(JRD)
    JRD_ang = utils.vec2ang(JRD_carte)
    #print JRD_ang
   
    # Problema Direto (Calcular o campo externo nas coordenadas do elipsoide)
    if ellipsoids.axis[0] > ellipsoids.axis[1] and ellipsoids.axis[1] > ellipsoids.axis[2]:
        B1 = B1_e (ellipsoids.m11,ellipsoids.m12,ellipsoids.m13,JRD,ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.axis[2])
        B2 = B2_e (ellipsoids.m21,ellipsoids.m22,ellipsoids.m23,JRD,ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.axis[2])
        B3 = B3_e (ellipsoids.m31,ellipsoids.m32,ellipsoids.m33,JRD,ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.axis[2])
    
    if ellipsoids.axis[1] == ellipsoids.axis[2] and ellipsoids.axis[0] > ellipsoids.axis[1]:
        f1 = f1_PO (ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.x1,ellipsoids.x2,ellipsoids.x3,ellipsoids.lamb,JRD)
        log = log_P (ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb)
        f2 = f2_P (ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb,log)
        B1 = B1_P (ellipsoids.dlambx1,JRD,f1,f2,log,ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb)
        B2 = B2_PO (ellipsoids.dlambx2,JRD,f1,f2)
        B3 = B3_PO (ellipsoids.dlambx3,JRD,f1,f2)
        
    if ellipsoids.axis[1] == ellipsoids.axis[2] and ellipsoids.axis[0] < ellipsoids.axis[1]:
        f1 = f1_PO (ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.x1,ellipsoids.x2,ellipsoids.x3,ellipsoids.lamb,JRD)
        tang = tang_O (ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb)
        f2 = f2_O (ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb,tang)    
        B1 = B1_O (ellipsoids.dlambx1,JRD,f1,f2,tang,ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb)
        B2 = B2_PO (ellipsoids.dlambx2,JRD,f1,f2)
        B3 = B3_PO (ellipsoids.dlambx3,JRD,f1,f2)        
    
    # Problema Direto (Calcular o campo externo nas coordenadas geograficas)
    Bx = Bx_c (B1,B2,B3,ellipsoids.mcon[0,0],ellipsoids.mcon[1,0],ellipsoids.mcon[2,0])
    By = By_c (B1,B2,B3,ellipsoids.mcon[0,1],ellipsoids.mcon[1,1],ellipsoids.mcon[2,1])
    Bz = Bz_c (B1,B2,B3,ellipsoids.mcon[0,2],ellipsoids.mcon[1,2],ellipsoids.mcon[2,2])
    
    return Bx,By,Bz
    
# Problema Direto (Calcular o campo externo e anomalia nas coordenadas geograficas no SI)
def jrd_cartesiano (inten,inc,dec,ellipsoids):
    lt = ln_v (dec, inc)
    mt = mn_v (dec, inc)
    nt = nn_v (inc)
    Ft = []
    JR = []
    JRD = []
    JRD_carte = []
    JRD_ang = []
    for i in range(len(ellipsoids)):
        Ft.append(F_e (inten,lt,mt,nt,ellipsoids[i].mcon[0,0],ellipsoids[i].mcon[1,0],ellipsoids[i].mcon[2,0],ellipsoids[i].mcon[0,1],ellipsoids[i].mcon[1,1],ellipsoids[i].mcon[2,1],ellipsoids[i].mcon[0,2],ellipsoids[i].mcon[1,2],ellipsoids[i].mcon[2,2]))
        JR.append(JR_e (ellipsoids[i].km,ellipsoids[i].JN,Ft[i]))
        JRD.append(JRD_e (ellipsoids[i].km,ellipsoids[i].N1,ellipsoids[i].N2,ellipsoids[i].N3,JR[i]))
        JRD_carte.append((ellipsoids[i].mconT).dot(JRD[i]))
        JRD_ang.append(utils.vec2ang(JRD_carte[i]))
    return JRD_ang

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
    
def f1_PO (a,b,x1,x2,x3,lamb,JRD):
    '''
    Calculo auxiliar do campo magnetico (fi) com relacao aos eixos do elipsoide 
    input:
    a,b - semi-eixos do elipsoide
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    v - matriz
    '''
    f1 = 2*np.pi*a*(b**2)*(((JRD[0]*x1)/(((a**2+lamb)**1.5)*(b**2+lamb))) + ((JRD[1]*x2 + JRD[2]*x3)/(((a**2+lamb)**0.5)*((b**2+lamb)**2))))
    return f1
    
def log_P (a,b,lamb):
    '''
    Calculo auxiliar do campo magnetico (fi) com relacao aos eixos do elipsoide 
    input:
    a,b - semi-eixos do elipsoide
    lamb - Maior raiz real da equacao cubica.
    output:
    cte - constante escalar.
    '''
    log = np.log(((a**2-b**2)**0.5+(a**2+lamb)**0.5)/((b**2+lamb)**0.5))
    return log
    
def f2_P (a,b,lamb,log):
    '''
    Calculo auxiliar do campo magnetico (fi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    v - matriz
    '''
    f2 = ((2*np.pi*a*(b**2))/((a**2-b**2)**1.5))*(log-((((a**2-b**2)*(a**2+lamb))**0.5)/(b**2+lamb)))
    return f2
    
def tang_O (a,b,lamb):
    '''
    Calculo auxiliar do campo magnetico (fi) com relacao aos eixos do elipsoide 
    input:
    a,b - semi-eixos do elipsoide
    lamb - Maior raiz real da equacao cubica.
    output:
    cte - constante escalar.
    '''
    tang = np.arctan(((b**2-a**2)/(a**2+lamb))**0.5)
    return tang
    
def f2_O (a,b,tang,lamb):
    '''
    Calculo auxiliar do campo magnetico (fi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    v - matriz
    '''
    f2 = ((2*np.pi*a*(b**2))/((b**2-a**2)**1.5))*(((((b**2-a**2)*(a**2+lamb))**0.5)/(b**2+lamb)) - tang)
    return f2

def B1_P (dlambx1,JRD,f1,f2,log,a,b,lamb):
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
    B1 = (dlambx1*f1) + ((4*np.pi*a*b**2)/((a**2-b**2)**1.5)) * JRD[0] * ((((a**2-b**2)/(a**2+lamb))**0.5) - log)
    return B1
    
def B1_O (dlambx1,JRD,f1,f2,tang,a,b,lamb):
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
    B1 = (dlambx1*f1) + ((4*np.pi*a*b**2)/((b**2-a**2)**1.5)) * JRD[0] * (tang - ((b**2-a**2)/(a**2+lamb))**0.5)
    return B1
    
def B2_PO (dlambx2,JRD,f1,f2):
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
    B2 = (dlambx2*f1) + (JRD[1]*f2)
    return B2
    
def B3_PO (dlambx3,JRD,f1,f2):
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
    B3 = (dlambx3*f1) + (JRD[2]*f2)
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