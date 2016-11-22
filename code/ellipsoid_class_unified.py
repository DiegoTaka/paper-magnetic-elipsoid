import numpy as np
from scipy import linalg

from fatiando import mesher, gridder, utils

import scipy.special

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
    Calculate the ellipsoid model for a body with a magnetization property.
    '''
    
    def __init__(self, xp, yp, zp, xc, yc, zc, a, b, c, azimuth, delta, gamma, props):
        GeometricElement.__init__(self, props)
        
        self.center = np.array([xc,yc,zc])
        self.axis = np.array([a,b,c])
        self.xp = (xp)
        self.yp = (yp)
        self.zp = (zp)
        self.conf = []
        
        self.azimuth = np.deg2rad(azimuth)
        self.delta = np.deg2rad(delta)
        self.gamma = np.deg2rad(gamma)
        
        if self.axis[0] > self.axis[1] and self.axis[1] > self.axis[2]: #Triaxial
            self.angles = np.array([self.azimuth+np.pi,self.delta,self.gamma])
            self.mcon,self.mconT = self.m_convTP()
            self.conf.append('Triaxial')
        elif self.axis[1] == self.axis[2] and self.axis[0] > self.axis[1]: #Prolate
            self.angles = np.array([self.azimuth+np.pi,self.delta,(np.pi/2)])
            self.mcon,self.mconT = self.m_convTP()
            self.conf.append('Prolate')
        elif self.axis[1] == self.axis[2] and self.axis[0] < self.axis[1]: #Oblate
            self.angles = np.array([self.azimuth,self.delta,(np.pi/2)])
            self.mcon,self.mconT = self.m_convO()
            self.conf.append('Oblate')
        else:
            raise ValueError("Input axis must have an ellipsoid shape!")
        
        self.inclirem = np.deg2rad(props['remanence'][1])
        self.declirem = np.deg2rad(props['remanence'][2])  
        self.ln = np.cos(self.declirem)*np.cos(self.inclirem)
        self.mn = np.sin(self.declirem)*np.cos(self.inclirem)
        self.nn = np.sin(self.inclirem)

        self.inck1 = np.deg2rad(props['k1'][1])
        self.inck2 = np.deg2rad(props['k2'][1])
        self.inck3 = np.deg2rad(props['k3'][1])
        self.deck1 = np.deg2rad(props['k1'][2])
        self.deck2 = np.deg2rad(props['k2'][2])
        self.deck3 = np.deg2rad(props['k3'][2])
        self.k_int = np.array([[props['k1'][0]],[props['k2'][0]],[props['k3'][0]]])
        self.k_inc = np.array([self.inck1,self.inck2,self.inck3])
        self.k_dec = np.array([self.deck1,self.deck2,self.deck3])
            
        if self.k_int[0] == (self.k_int[1] and self.k_int[2]):
            self.km = self.k_matrix()
            self.conf.append('Isotropic magnetization')
        else:
            self.km = self.k_matrix2()
            self.conf.append('Anisotropic magnetization')
                
        self.x1,self.x2,self.x3 = self.x_e()
            
        self.JN = self.JN_e ()
        
        if self.axis[0] > self.axis[1] and self.axis[1] > self.axis[2]:
            self.lamb,self.teta,self.q,self.p,self.p2,self.p1,self.p0 = self.lamb_T()
            self.dlambx1,self.dlambx2,self.dlambx3 = self.dlambx_T()
            self.F,self.E,self.F2,self.E2,self.k,self.theta_l = self.legendre_integrals()
            self.N1,self.N2,self.N3 = self.N_desmagT()
            self.A, self.B, self.C = self.potential_integrals()
            self.m11,self.m12,self.m13,self.m21,self.m22,self.m23,self.m31,self.m32,self.m33, self.cte, self.V1, self.V2, self.V3 = self.mx()
     
        if self.axis[1] == self.axis[2] and self.axis[0] > self.axis[1]:
            self.N1,self.N2 = self.N_desmagP()
            self.N3 = self.N2
            self.r = self.r_e ()
            self.delta = self.delta_e ()
            self.lamb = self.lamb_PO ()
            self.dlambx1,self.dlambx2,self.dlambx3 = self.dlambx_PO()
            
        if self.axis[1] == self.axis[2] and self.axis[0] < self.axis[1]:
            self.N1,self.N2 = self.N_desmagO()
            self.N3 = self.N2
            self.r = self.r_e ()
            self.delta = self.delta_e ()
            self.lamb = self.lamb_PO ()
            self.dlambx1,self.dlambx2,self.dlambx3 = self.dlambx_PO()
        
    def __str__(self):
        """
        Return a string representation of the ellipsoids.
        """
        
        names = [('xc', self.self.center[0]), ('yc', self.self.center[1]), ('zc', self.self.center[2]),
                 ('a', self.axis[0]), ('b', self.axis[1]), ('c', self.axis[2]),
                 ('alpha', self.angles[0]),('delta', self.angles[1]),('gamma', self.angles[2])]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)
        
    def m_convTP (self):
        '''
        Builds the matrix of coordinate system change to the center of the ellipsoid. Used for the triaxial 
        and prolate ellipsoids.
        
        input:
        alpha - Azimuth+180 in relation to the major-axe and the geographic north (0<=alpha<=360, radians).
        delta - Inclination between the major-axe and the horizontal plane (0<=delta<=90, radians).
        gamma - Angle between the intermediate-axe and the vertical projection of the horizontal plane to the
        center of the ellipsoid(radians).
        
        output:
        A 3x3 matrix.
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

    def m_convO (self):
        '''
        Builds the matrix of coordinate system change to the center of the ellipsoid. Used for the oblate ellipsoid.
        
        input:
        alpha - Azimuth in relation to the major-axe and the geographic north (0<=alpha<=360, radians).
        delta - Inclination between the major-axe and the horizontal plane (0<=delta<=90, radians).
        
        output:
        A 3x3 matrix.
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
        mcon[2][2] = (0.)
        mconT = (mcon).T
        return mcon, mconT
        
    def x_e (self):
        '''
        Calculates the new coordinates with origin at the center of the ellipsoid.

        input:
        xp,yp - Origin of the ellipsoid in the geographic coordinate.
        zp - Depth of the the ellipsoid.
        mcon - Matrix of conversion.
        
        output:
        x1, x2, x3 - The three axes of the coordinates.
        '''
        x1 = (self.xp-self.center[0])*self.mcon[0,0]+(self.yp-self.center[1])*self.mcon[0,1]-(self.zp+self.center[2])*self.mcon[0,2]
        x2 = (self.xp-self.center[0])*self.mcon[1,0]+(self.yp-self.center[1])*self.mcon[1,1]-(self.zp+self.center[2])*self.mcon[1,2]
        x3 = (self.xp-self.center[0])*self.mcon[2,0]+(self.yp-self.center[1])*self.mcon[2,1]-(self.zp+self.center[2])*self.mcon[2,2]
        return x1, x2, x3
        
    def JN_e (self):
        '''
        Changes the remanent magnetization vector to the body coordinate.
        
        input:
        ln,nn,mn - direction cosines of the remanent magnetization vector.
        mcon - matrix of conversion.
        
        output:
        JN - Remanent magnetization vector in the body coordinate.         
        '''
        
        JN = self.props['remanence'][0]*np.ravel(np.array([[(self.ln*self.mcon[0,0]+self.mn*self.mcon[0,1]+self.nn*self.mcon[0,2])], [(self.ln*self.mcon[1,0]+self.mn*self.mcon[1,1]+self.nn*self.mcon[1,2])], [(self.ln*self.mcon[2,0]+self.mn*self.mcon[2,1]+self.nn*self.mcon[2,2])]]))
        return JN

    def N_desmagT (self):
        '''
        Calculates the three demagnetization factor along major, intermediate and minor axis. For the triaxial ellipsoid use.
        
        input:
        a,b,c - Major, intermediate and minor axis, respectively.
        F2, E2 - Lagrange's normal eliptic integrals of first and second order.
        
        output:
        N1, N2, N3 - Major, intermediate and minor demagnetization factors, respectively.        
        '''
        
        N1 = ((4.*np.pi*self.axis[0]*self.axis[1]*self.axis[2])/((self.axis[0]**2-self.axis[1]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5)) * (self.F2-self.E2)
        N2 = (((4.*np.pi*self.axis[0]*self.axis[1]*self.axis[2])*(self.axis[0]**2-self.axis[2]**2)**0.5)/((self.axis[0]**2-self.axis[1]**2)*(self.axis[1]**2-self.axis[2]**2))) * (self.E2 - ((self.axis[1]**2-self.axis[2]**2)/(self.axis[0]**2-self.axis[2]**2)) * self.F2 - ((self.axis[2]*(self.axis[0]**2-self.axis[1]**2))/(self.axis[0]*self.axis[1]*(self.axis[0]**2-self.axis[2]**2)**0.5)))
        N3 = ((4.*np.pi*self.axis[0]*self.axis[1]*self.axis[2])/((self.axis[1]**2-self.axis[2]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5)) * (((self.axis[1]*(self.axis[0]**2-self.axis[2]**2)**0.5)/(self.axis[0]*self.axis[2])) - self.E2)
        return N1, N2, N3
        
    def N_desmagP (self):
        '''
        Calculates the three demagnetization factor along major, intermediate and minor axis. For the prolate ellipsoid use.
        
        input:
        a,b - Major and minor axis, respectively.
        
        output:
        N1, N2 - Major and minor demagnetization factors, respectively.        
        '''
        
        N1 = ((4.*np.pi*self.axis[0]*self.axis[1]**2)/((self.axis[0]**2-self.axis[1]**2)**1.5)) * (np.log(((self.axis[0]**2/self.axis[1]**2)-1.)**0.5 + (self.axis[0]/self.axis[1])) - (1. - (self.axis[1]**2/self.axis[0]**2))**0.5)
        N2 = 2.*np.pi - N1/2.
        return N1, N2
        
    def N_desmagO (self):
        '''
        Calculates the three demagnetization factor along major, intermediate and minor axis. For the oblate ellipsoid use.
        
        input:
        a,b - Minor and major axis, respectively.
        
        output:
        N1, N2 - Minor and Major demagnetization factors, respectively.        
        '''
        
        N1 = ((4.*np.pi*self.axis[0]*self.axis[1]**2)/((self.axis[1]**2-self.axis[0]**2)**1.5)) * ((((self.axis[1]**2-self.axis[0]**2)**0.5)/(self.axis[0])) - np.arctan(((self.axis[1]**2-self.axis[0]**2)**0.5)/(self.axis[0])))
        N2 = 2.*np.pi - N1/2.
        return N1, N2

    def k_matrix (self):
        '''
        Build susceptibility tensors matrix for the isotropic case in the body coordinates.
        
        input:
        mcon - Matrix of conversion.
        k_int - Intensity of the three directions of susceptibility.
        
        output:
        km - Susceptibility tensors matrix.        
        '''
        
        km = np.zeros([3,3])
        for i in range (3):
            for j in range (3):
                for r in range (3):
                    km[i,j] = km[i,j] + (self.k_int[r]*(self.mcon[r,0]*self.mcon[i,0] + self.mcon[r,1]*self.mcon[i,1] + self.mcon[r,2]*self.mcon[i,2])*(self.mcon[r,0]*self.mcon[j,0] + self.mcon[r,1]*self.mcon[j,1] + self.mcon[r,2]*self.mcon[j,2]))
        return km

    def k_matrix2 (self):
        '''
        Build the susceptibility tensors matrix for the anisotropic case in the body coordinates.
        
        input:
        mcon - Matrix of conversion.
        k_int - Intensity of the three directions of susceptibility.
        
        output:
        km - Susceptibility tensors matrix.        
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
        Calculates the larger root of the cubic equation: s^3 + p2*s^2 + p1*s + p0 = 0.
        Used in the triaxial ellipsoid.

        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        x1, x2, x3 - Axis of the body coordinate system.
        
        output:
        lamb - Larger root.
        teta, q, p, p2, p1, p0 - constants of the cubic equation.        
        '''
        
        p0 = (self.axis[0]*self.axis[1]*self.axis[2])**2-(self.axis[1]*self.axis[2]*self.x1)**2-(self.axis[2]*self.axis[0]*self.x2)**2-(self.axis[0]*self.axis[1]*self.x3)**2
        p1 = (self.axis[0]*self.axis[1])**2+(self.axis[1]*self.axis[2])**2+(self.axis[2]*self.axis[0])**2-(self.axis[1]**2+self.axis[2]**2)*self.x1**2-(self.axis[2]**2+self.axis[0]**2)*self.x2**2-(self.axis[0]**2+self.axis[1]**2)*self.x3**2
        p2 = self.axis[0]**2+self.axis[1]**2+self.axis[2]**2-self.x1**2-self.x2**2-self.x3**2
        p = p1-(p2**2)/3.
        q = p0-((p1*p2)/3.)+2*(p2/3.)**3
        p3 = (-q/(2*np.sqrt((-p/3.)**3)))
        for i in range (len(p3)):
            if p3[i] > 1.:
                p3[i] = 1.
        teta = np.arccos(p3)
        lamb = 2.*((-p/3.)**0.5)*np.cos(teta/3.)-(p2/3.)
        return lamb, teta, q, p, p2, p1, p0
    
    def r_e (self):
        '''
        Calculates the distance between the observation point and the center of the ellipsoid.
        Used in the prolate and oblate ellipsoids.
        
        input:
        x1, x2, x3 - Axis of the body coordinate system.
        
        output:
        r - Distance between the observation point and the center of the ellipsoid.        
        '''
        
        r = (self.x1)**2+(self.x2)**2+(self.x3)**2
        return r
        
    def delta_e (self):
        '''
        Calculates an auxiliar constant for lambda.
        
        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        r - Distance between the observation point and the center of the ellipsoid.
        x1, x2, x3 - Axis of the body coordinate system.
        
        output:
        delta - Auxiliar constant for lambda.        
        '''
        
        delta = (self.r**2 + (self.axis[0]**2-self.axis[1]**2)**2 - 2*(self.axis[0]**2-self.axis[1]**2) * (self.x1**2 - self.x2**2 - self.x3**2))**0.5
        return delta    
        
    def lamb_PO (self):
        '''
        Calculates the Larger root of the cartesian ellipsoidal equation.
        Used in the prolate and oblate ellipsoids.
        
        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        delta - Auxiliar constant for lambda.
        r - Distance between the observation point and the center of the ellipsoid.
        
        output:
        lamb - Larger root of the cartesian ellipsoidal equation.
        '''
        
        lamb = (self.r - self.axis[0]**2 - self.axis[1]**2 + self.delta)/2.
        return lamb
        
    def legendre_integrals(self):
        '''
        Calculates parameters and the Legendre's normal elliptic integrals of first and second order.
        
        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        lamb - Larger root of the cubic equation: s^3 + p2*s^2 + p1*s + p0 = 0.
        
        output:
        F - Legendre's normal elliptic integrals of first order.
        E - Legendre's normal elliptic integrals of second order.
        F2 - Legendre's normal elliptic integrals of first order (calculus of demagnetization factors).
        E2 - Legendre's normal elliptic integrals of second order (calculus of demagnetization factors).
        k - Legendre's normal elliptic integrals parameter.
        theta_l- Legendre's normal elliptic integrals parameter.        
        '''
        
        k = np.zeros_like(self.lamb)
        k1 = ((self.axis[0]**2-self.axis[1]**2)/(self.axis[0]**2-self.axis[2]**2))**0.5
        k.fill(k1)
        theta_l = np.arcsin(((self.axis[0]**2-self.axis[2]**2)/(self.axis[0]**2+self.lamb))**0.5)
        theta_l2 = np.arccos(self.axis[2]/self.axis[0])
        F = scipy.special.ellipkinc(theta_l, k**2)
        E = scipy.special.ellipeinc(theta_l, k**2)
        F2 = scipy.special.ellipkinc(theta_l2, k1**2)
        E2 = scipy.special.ellipeinc(theta_l2, k1**2)
        return F,E,F2,E2,k,theta_l

    def dlambx_T (self):
        '''
        Calculates the derivatives of the ellipsoid equation for each body coordinates in realation to lambda.
        Used for the triaxial ellipsoid.
        
        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        x1, x2, x3 - Axis of the body coordinate system.
        lamb - Larger root of the cubic equation: s^3 + p2*s^2 + p1*s + p0 = 0.
        
        output:
        dlambx1,dlambx2,dlambx3 - Derivatives of the ellipsoid equation for each body coordinates in realation to x1,x2 and x3.        
        '''
        
        dlambx1 = (2*self.x1/(self.axis[0]**2+self.lamb))/((self.x1/(self.axis[0]**2+self.lamb))**2+(self.x2/(self.axis[1]**2+self.lamb))**2+((self.x3/(self.axis[2]**2+self.lamb))**2))        
        dlambx2 = (2*self.x2/(self.axis[1]**2+self.lamb))/((self.x1/(self.axis[0]**2+self.lamb))**2+(self.x2/(self.axis[1]**2+self.lamb))**2+((self.x3/(self.axis[2]**2+self.lamb))**2))
        dlambx3 = (2*self.x3/(self.axis[2]**2+self.lamb))/((self.x1/(self.axis[0]**2+self.lamb))**2+(self.x2/(self.axis[1]**2+self.lamb))**2+((self.x3/(self.axis[2]**2+self.lamb))**2))
        return dlambx1, dlambx2, dlambx3

    def dlambx_PO (self):
        '''
        Calculates the derivatives of the ellipsoid equation for each body coordinates in realation to lambda.
        Used for the prolate and oblate ellipsoids.
        
        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        x1, x2, x3 - Axis of the body coordinate system.
        delta - Auxiliar constant for lambda.
        
        output:
        dlambx1,dlambx2,dlambx3 - Derivatives of the ellipsoid equation for each body coordinates in realation to x1,x2 and x3.        
        '''
        
        dlambx1 = self.x1*(1+(self.r-self.axis[0]**2+self.axis[1]**2)/self.delta)
        dlambx2 = self.x2*(1+(self.r+self.axis[0]**2-self.axis[1]**2)/self.delta)
        dlambx3 = self.x3*(1+(self.r+self.axis[0]**2-self.axis[1]**2)/self.delta)
        return dlambx1, dlambx2, dlambx3

    def potential_integrals(self):
        '''
        Calculates the integrals which is part of the solution of the potential field of an homogeneous ellipsoid (Dirichlet,1839).
        
        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        k - Legendre's normal elliptic integrals parameter.
        theta_l- Legendre's normal elliptic integrals parameter.
        F - Legendre's normal elliptic integrals of first order.
        E - Legendre's normal elliptic integrals of second order.
        
        output:
        A2,B2,C2 - Integrals of the potential field of an homogeneous ellipsoid.        
        '''
        
        A2 = (2/((self.axis[0]**2-self.axis[1]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5))*(self.F-self.E)
        B2 = ((2*(self.axis[0]**2-self.axis[2]**2)**0.5)/((self.axis[0]**2-self.axis[1]**2)*(self.axis[1]**2-self.axis[2]**2)))*(self.E-((self.axis[1]**2-self.axis[2]**2)/(self.axis[0]**2-self.axis[2]**2))*self.F-((self.k**2*np.sin(self.theta_l)*np.cos(self.theta_l))/(1-self.k**2*np.sin(self.theta_l)*np.sin(self.theta_l))**0.5))
        C2 = (2/((self.axis[1]**2-self.axis[2]**2)*(self.axis[0]**2-self.axis[2]**2)**0.5))*(((np.sin(self.theta_l)*((1-self.k**2*np.sin(self.theta_l)*np.sin(self.theta_l))**0.5))/np.cos(self.theta_l))-self.E)
        return A2,B2,C2

    def mx(self):
        '''
        Additional calculations for the ellipsoid magnetic field.
        
        input:
        a, b, c - Major, intermediate and minor axis, respectively.
        x1, x2, x3 - Axis of the body coordinate system.
        A,B,C - Integrals of the potential field of an homogeneous ellipsoid
        lamb - Larger root of the cubic equation: s^3 + p2*s^2 + p1*s + p0 = 0.
        
        output: 
        m11, m12, m13, m21, m22, m23, m31, m32, m33, cte, V1, V2, V3 - calculus for the ellipsoid magnetic field.        
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

def ellipsoid (xp,yp,zp,inten,inc,dec,ellipsoids):
    '''
    Calculate the potential fields of a homogeneous ellipsoid.

    **Magnetic**

    Calculates the magnetic effect produced by a triaxial, a prolate or/and an oblate ellipsoid. The functions are
    based on Clark et al. (1986) and Emerson et al. (1985).    
    '''
    
    lt,mt,nt = lmnn_v (dec, inc)
    Ft = F_e (inten,lt,mt,nt,ellipsoids.mcon[0,0],ellipsoids.mcon[1,0],ellipsoids.mcon[2,0],ellipsoids.mcon[0,1],ellipsoids.mcon[1,1],ellipsoids.mcon[2,1],ellipsoids.mcon[0,2],ellipsoids.mcon[1,2],ellipsoids.mcon[2,2])
    JR = JR_e (ellipsoids.km,ellipsoids.JN,Ft)
    JRD = JRD_e (ellipsoids.km,ellipsoids.N1,ellipsoids.N2,ellipsoids.N3,JR)
    JRD_carte = (ellipsoids.mconT).dot(JRD)
    JRD_ang = utils.vec2ang(JRD_carte)
   
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
        arctang = arctang_O (ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb)
        f2 = f2_O (ellipsoids.axis[0],ellipsoids.axis[1],arctang,ellipsoids.lamb)   
        B1 = B1_O (ellipsoids.dlambx1,JRD,f1,f2,arctang,ellipsoids.axis[0],ellipsoids.axis[1],ellipsoids.lamb)
        B2 = B2_PO (ellipsoids.dlambx2,JRD,f1,f2)
        B3 = B3_PO (ellipsoids.dlambx3,JRD,f1,f2)  
    
    Bx = Bx_c (B1,B2,B3,ellipsoids.mcon[0,0],ellipsoids.mcon[1,0],ellipsoids.mcon[2,0])
    By = By_c (B1,B2,B3,ellipsoids.mcon[0,1],ellipsoids.mcon[1,1],ellipsoids.mcon[2,1])
    Bz = Bz_c (B1,B2,B3,ellipsoids.mcon[0,2],ellipsoids.mcon[1,2],ellipsoids.mcon[2,2])
    
    return Bx,By,Bz

def jrd_cartesiano (inten,inc,dec,ellipsoids):
    '''
    Calculates the intensity and direction of the resultant vector of magnetization.
    
    input:
    inten - Intensity of the Earth's magnetic field.
    inc - Inclination of the Earth's magnetic field.
    dec - Declination of the Earth's magnetic field.
    ellipsoid - magnetic ellipsoid model.
    
    output: 
    JRD_ang - Vector with intensity and direction of the resultant vector of magnetization 
    in the cartesian coordinates(degrees).    
    '''
    
    inc = np.deg2rad(inc)
    dec = np.deg2rad(dec)
    lt,mt,nt = lmnn_v (dec, inc)
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
    
def lmnn_v (declination, inclination):
    '''
    Calculates de direction cosines of a vector.
    
    input:
    inc - Inclination.
    dec - Declination.
    
    output:
    ln,mn,nn - direction cosines.    
    '''
    
    ln = np.cos(declination)*np.cos(inclination)
    mn = np.sin(declination)*np.cos(inclination)
    nn = np.sin(inclination)
    return ln,mn,nn
    
def F_e (inten,lt,mt,nt,l1,l2,l3,m1,m2,m3,n1,n2,n3):
    '''
    Change the magnetization vetor of the Earth's field to the body coordinates.
    
    input:
    inten - Intensity of the Earth's magnetic field.
    lt,mt,nt - direction cosines of the Earth's magnetic field.
    l1,l2,l3,m1,m2,m3,n1,n2,n3 - matrix of body coordinates change.
    
    output:
    Ft - The magnetization vetor of the Earth's field to the body coordinates.    
    '''
    
    Ft = inten*np.ravel(np.array([[(lt*l1+mt*m1+nt*n1)], [(lt*l2+mt*m2+nt*n2)], [(lt*l3+mt*m3+nt*n3)]]))
    return Ft
    
def JR_e (km,JN,Ft):
    '''
    Calculates the resultant magnetization vector without self-demagnetization correction.
    
    input:
    km - matrix of susceptibilities tensor.
    JN - Remanent magnetization
    Ft - Magnetization vetor of the Earth's field in the body coordinates.
    
    output:
    JR - Resultant magnetization vector without self-demagnetization correction.   
    '''
    
    JR = km.dot(Ft) + JN
    return JR
    
def JRD_e (km,N1,N2,N3,JR):
    '''
    Calculates resultant magnetization vector with self-demagnetization correction.
    
    input:
    km - matrix of susceptibilities tensor.
    N1,N2,N3 - Demagnetization factors in relation to a, b and c, respectively.
    JR - resultant magnetization vector without self-demagnetization correction.
    
    output:
    JRD - Resultant magnetization vector without self-demagnetization correction.    
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
    Calculates the B1 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    Used in the triaxial ellipsoid.
    
    input:
    m11,m12,m13 - Calculus for the ellipsoid magnetic field.
    J - Resultant magnetization vector without self-demagnetization correction.
    a,b,c - Major, intermediate and minor axis, respectively.
    
    output:
    B1 - The B1 component of the magnetic field generated by n-ellipsoids in the body coordinates.    
    '''
    
    B1 = 2*np.pi*a*b*c*(m11*J[0]+m12*J[1]+m13*J[2])
    return B1

def B2_e (m21,m22,m23,J,a,b,c):
    '''
    Calculates the B2 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    Used in the triaxial ellipsoid.
    
    input:
    m21,m22,m23 - Calculus for the ellipsoid magnetic field.
    J - Resultant magnetization vector without self-demagnetization correction.
    a,b,c - Major, intermediate and minor axis, respectively.
    
    output:
    B2 - The B2 component of the magnetic field generated by n-ellipsoids in the body coordinates.    
    '''
    
    B2 = 2*np.pi*a*b*c*(m21*J[0]+m22*J[1]+m23*J[2])
    return B2
    
def B3_e (m31,m32,m33,J,a,b,c):
    '''
    Calculates the B3 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    Used in the triaxial ellipsoid.
    
    input:
    m31,m32,m33 - Calculus for the ellipsoid magnetic field.
    J - Resultant magnetization vector with self-demagnetization correction.
    a,b,c - Major, intermediate and minor axis, respectively.
    
    output:
    B3 - The B3 component of the magnetic field generated by n-ellipsoids in the body coordinates.    
    '''
    
    B3 = 2*np.pi*a*b*c*(m31*J[0]+m32*J[1]+m33*J[2])
    return B3
    
def f1_PO (a,b,x1,x2,x3,lamb,JRD):
    '''
    Auxiliar calculus of magnetic field generated by a prolate or an oblate ellipsoid.

    input:
    a,b - Major and minor axis, respectively.
    x1,x2,x3 - Axis of the body coordinate system.
    lamb - Larger root of the cartesian ellipsoidal equation.
    JRD - Resultant magnetization vector with self-demagnetization correction.
    
    output:
    f1 - Auxiliar calculus of magnetic field generated by a prolate or an oblate ellipsoid.
    '''
    
    f1 = 2*np.pi*a*(b**2)*(((JRD[0]*x1)/(((a**2+lamb)**1.5)*(b**2+lamb))) + ((JRD[1]*x2 + JRD[2]*x3)/(((a**2+lamb)**0.5)*((b**2+lamb)**2))))
    return f1
    
def log_P (a,b,lamb):
    '''
    Auxiliar calculus of magnetic field generated by a prolate ellipsoid.

    input:
    a,b - Major and minor axis, respectively.
    lamb - Larger root of the cartesian ellipsoidal equation.
    
    output:
    log - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    '''
    
    log = np.log(((a**2-b**2)**0.5+(a**2+lamb)**0.5)/((b**2+lamb)**0.5))
    return log
    
def f2_P (a,b,lamb,log):
    '''
    Auxiliar calculus of magnetic field generated by a prolate ellipsoid.

    input:
    a,b - Major and minor axis, respectively.
    lamb - Larger root of the cartesian ellipsoidal equation.
    log - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    
    output:
    f2 - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    '''
    
    f2 = ((2*np.pi*a*(b**2))/((a**2-b**2)**1.5))*(log-((((a**2-b**2)*(a**2+lamb))**0.5)/(b**2+lamb)))
    return f2
    
def arctang_O (a,b,lamb):
    '''
    Auxiliar calculus of magnetic field generated by an oblate ellipsoid.

    input:
    a,b - Major and minor axis, respectively.
    lamb - Larger root of the cartesian ellipsoidal equation.
    
    output:
    arctang - Auxiliar calculus of magnetic field generated by an oblate ellipsoid.
    '''
    
    arctang = np.arctan(((b**2-a**2)/(a**2+lamb))**0.5)
    return arctang
    
def f2_O (a,b,arctang,lamb):
    '''
    Auxiliar calculus of magnetic field generated by an oblate ellipsoid.

    input:
    a,b - Major and minor axis, respectively.
    lamb - Larger root of the cartesian ellipsoidal equation.
    arctang - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    
    output:
    f2 - Auxiliar calculus of magnetic field generated by an oblate ellipsoid.
    '''
    
    f2 = ((2*np.pi*a*b*b)/((b*b-a*a)**1.5))*(((((b*b-a*a)*(a*a+lamb))**0.5)/(b*b+lamb)) - arctang)
    return f2

def B1_P (dlambx1,JRD,f1,f2,log,a,b,lamb):
    '''
    Calculates the B1 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    Used in the prolate ellipsoid.
    
    input:
    dlambx1 - Derivative of the ellipsoid equation for each body coordinates in realation to x1.
    JRD - Resultant magnetization vector with self-demagnetization correction.
    f1,f2,log - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    a,b - Major and minor axis, respectively.
    lamb - Larger root of the cartesian ellipsoidal equation.
    
    output:
    B1 - The B1 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    '''
    
    B1 = (dlambx1*f1) + ((4*np.pi*a*b**2)/((a**2-b**2)**1.5)) * JRD[0] * ((((a**2-b**2)/(a**2+lamb))**0.5) - log)
    return B1
    
def B1_O (dlambx1,JRD,f1,f2,arctang,a,b,lamb):
    '''
    Calculates the B1 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    Used in the oblate ellipsoid.
    
    input:
    dlambx1 - Derivative of the ellipsoid equation for each body coordinates in realation to x1.
    JRD - Resultant magnetization vector with self-demagnetization correction.
    f1,f2,arctang - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    a,b - Major and minor axis, respectively.
    lamb - Larger root of the cartesian ellipsoidal equation.
    
    output:
    B1 - The B1 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    '''
    
    B1 = (dlambx1*f1) + ((4*np.pi*a*b**2)/((b**2-a**2)**1.5)) * JRD[0] * (arctang - ((b**2-a**2)/(a**2+lamb))**0.5)
    return B1
    
def B2_PO (dlambx2,JRD,f1,f2):
    '''
    Calculates the B2 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    Used in the prolate and oblate ellipsoid.
    
    input:
    dlambx2 - Derivative of the ellipsoid equation for each body coordinates in realation to x2.
    JRD - Resultant magnetization vector with self-demagnetization correction.
    f1,f2 - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    
    output:
    B2 - The B2 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    '''
    
    B2 = (dlambx2*f1) + (JRD[1]*f2)
    return B2
    
def B3_PO (dlambx3,JRD,f1,f2):
    '''
    Calculates the B3 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    Used in the prolate and oblate ellipsoids.
    
    input:
    dlambx3 - Derivative of the ellipsoid equation for each body coordinates in realation to x3.
    JRD - Resultant magnetization vector with self-demagnetization correction.
    f1,f2 - Auxiliar calculus of magnetic field generated by a prolate ellipsoid.
    
    output:
    B3 - The B3 component of the magnetic field generated by n-ellipsoids in the body coordinates.
    '''
    
    B3 = (dlambx3*f1) + (JRD[2]*f2)
    return B3
    
def Bx_c (B1,B2,B3,l1,l2,l3):
    '''
    Change the X component of the magnetic field generated by n-ellipsoids to the cartesian coordinates.
    
    input:
    B1,B2,B3 - Components of the magnetic field generated by n-ellipsoids to the body coordinates.
    l1,l2,l3 - Direction cosines for coordinates change.
    
    output:
    Bz - The X component of the magnetic field generated by n-ellipsoids to the cartesian coordinates.
    '''
    
    Bx = B1*l1+B2*l2+B3*l3
    return Bx

def By_c (B1,B2,B3,m1,m2,m3):
    '''
    Change the Y component of the magnetic field generated by n-ellipsoids to the cartesian coordinates.
    
    input:
    B1,B2,B3 - Components of the magnetic field generated by n-ellipsoids to the body coordinates.
    m1,m2,m3 - Direction cosines for coordinates change.
    
    output:
    Bz - The Y component of the magnetic field generated by n-ellipsoids to the cartesian coordinates.
    '''
    
    By = B1*m1+B2*m2+B3*m3
    return By

def Bz_c (B1,B2,B3,n1,n2,n3):
    '''
    Change the Z component of the magnetic field generated by n-ellipsoids to the cartesian coordinates.
    
    input:
    B1,B2,B3 - Components of the magnetic field generated by n-ellipsoids to the body coordinates.
    n1,n2,n3 - Direction cosines for coordinates change.
    
    output:
    Bz - The Z component of the magnetic field generated by n-ellipsoids to the cartesian coordinates.
    '''
    
    Bz = B1*n1+B2*n2+B3*n3
    return Bz
    
def bx_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    '''
    Calculates the X component of the magnetic field generated by n-ellipsoids.
    
    input:
    xp,yp,zp - grid of observation points x, y, and z.
    inten,inc,dec - Intensity, inclination and declination of the Earth's magnetic field.
    ellipsoids - ellipsoid model.
    
    output:
    res - The X component of the magnetic field generated by n-ellipsoids.
    '''
    
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1.
    inc = np.deg2rad(inc)
    dec = np.deg2rad(dec)
    
    for i in range(len(ellipsoids)):
        bx,by,bz = ellipsoid (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        res += bx
    res = res*ctemag
    return res
    
def by_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    '''
    Calculates the Y component of the magnetic field generated by n-ellipsoids.
    
    input:
    xp,yp,zp - grid of observation points x, y, and z.
    inten,inc,dec - Intensity, inclination and declination of the Earth's magnetic field.
    ellipsoids - ellipsoid model.
    
    output:
    res - The Y component of the magnetic field generated by n-ellipsoids.
    '''
    
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1.
    inc = np.deg2rad(inc)
    dec = np.deg2rad(dec)
    
    for i in range(len(ellipsoids)):
        bx,by,bz = ellipsoid (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        res += by
    res = res*ctemag
    return res

def bz_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    '''
    Calculates the Z component of the magnetic field generated by n-ellipsoids.
    
    input:
    xp,yp,zp - grid of observation points x, y, and z.
    inten,inc,dec - Intensity, inclination and declination of the Earth's magnetic field.
    ellipsoids - ellipsoid model.
    
    output:
    res - The Z component of the magnetic field generated by n-ellipsoids.
    '''
    
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1.
    inc = np.deg2rad(inc)
    dec = np.deg2rad(dec)
    
    for i in range(len(ellipsoids)):
        bx,by,bz = ellipsoid (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        res += bz
    res = res*ctemag
    return res
    
def tf_c(xp,yp,zp,inten,inc,dec,ellipsoids):
    '''
    Calculates the approximated total-field anomaly generated by n-ellipsoids.
    
    input:
    xp,yp,zp - grid of observation points x, y, and z.
    inten,inc,dec - Intensity, inclination and declination of the Earth's magnetic field.
    ellipsoids - ellipsoid model.
    
    output:
    res - The approximated total-field anomaly generated by n-ellipsoids.
    '''
    
    if xp.shape != yp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same shape!")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    ctemag = 1.
    inc = np.deg2rad(inc)
    dec = np.deg2rad(dec)
    
    for i in range(len(ellipsoids)):
        bx,by,bz = ellipsoid (xp,yp,zp,inten,inc,dec,ellipsoids[i])
        tf = bx*np.cos(inc)*np.cos(dec) + by*np.cos(inc)*np.sin(dec) + bz*np.sin(inc)
        res += tf
    res = res*ctemag
    return res