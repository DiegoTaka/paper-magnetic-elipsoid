from __future__ import division
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

from fatiando.gravmag import sphere
from fatiando import mesher, gridder, utils
from fatiando.vis import mpl

import scipy.special
import scipy.interpolate

def elipsoide (a,b,h,alfa,delta,X,Y,Z,xc,yc,declinacao,inclinacao,intensidade,declinacaoT,inclinacaoT,intensidadeT,km):
    '''
    Calcula as tres componentes do campo magnetico de um elipsoide.
    
    a: escalar - semi eixo maior
    b: escalar - semi eixo intermediario
    c: escalar - semi eixo menor
    h: escalar -  profundidade
    alfa: escalar - azimute do elipsoide em relacao ao "a"
    delta: escalar - inclinacao do elipsoide em relacao ao "a"
    gamma: escalar - angulo entre o semi eixo "b" e a projecao do centro do elipsoide no plano xy
    X: matriz - malha do eixo x
    Y: matriz - malha do eixo y
    Z: matriz - malha do eixo z
    xc: escalar - posicao x do centro do elipsoide
    yc: escalar - posicao y do centro do elipsoide
    J: vetor - magnetizacao do corpo
    '''

    # Calculo de parametros de direcao
    l1 = l1_v (alfa, delta)
    l2 = l2_v (alfa, delta)
    l3 = l3_v (alfa, delta)

    m1 = m1_v (alfa, delta)
    m2 = m2_v (alfa, delta)
    m3 = m3_v (alfa, delta)

    n1 = n1_v (delta)
    n2 = n2_v (delta)
    n3 = n3_v (delta)
    
    ln = ln_v (declinacao, inclinacao)
    mn = mn_v (declinacao, inclinacao)
    nn = nn_v (inclinacao)
    
    lt = ln_v (declinacaoT, inclinacaoT)
    mt = mn_v (declinacaoT, inclinacaoT)
    nt = nn_v (inclinacaoT)

    # Magnetizacoes nas coordenadas do elipsoide
    F = F_e (intensidadeT,lt,mt,nt,l1,l2,l3,m1,m2,m3,n1,n2,n3)
    JN = JN_e (intensidade,ln,mn,nn,l1,l2,l3,m1,m2,m3,n1,n2,n3)
    N1,N2 = N_desmag (a,b)
    JR = JR_e (km,JN,F)
    JRD = JRD_e (km,N1,N2,F,JR)
    mcon = np.array([[l1, m1, n1],[l2, m2, n2],[l3, m3, n3]])
    mconT = (mcon).T
    JRD_carte = (mconT).dot(JRD)
    JRD_ang = utils.vec2ang(JRD_carte)
    print JRD_ang
    
    # Coordenadas Cartesianas elipsoide
    x1 = x1_e (X,Y,Z,xc,yc,h,l1,m1,n1)
    x2 = x2_e (X,Y,Z,xc,yc,h,l2,m2,n2)
    x3 = x3_e (X,Y,Z,xc,yc,h,l3,m3,n3)

    # Calculos auxiliares
    r = r_e (x1,x2,x3)
    delta = delta_e (r,a,b,x1,x2,x3)

    # Raizes da equacao cubica
    lamb = lamb_e (r,a,b,delta)

    # Derivadas de lambda em relacao as posicoes
    dlambx1 = dlambx1_e (a,b,x1,x2,x3,lamb,r,delta)
    dlambx2 = dlambx2_e (a,b,x1,x2,x3,lamb,r,delta)
    dlambx3 = dlambx3_e (a,b,x1,x2,x3,lamb,r,delta)
    
    # Calculos auxiliares do campo
    f1 = f1_e (a,b,x1,x2,x3,lamb,JRD)
    tang = tang_e (a,b,lamb)
    f2 = f2_e (a,b,lamb,tang)
    
    # Problema Direto (Calcular o campo externo nas coordenadas do elipsoide)
    B1 = B1_e (dlambx1,JRD,f1,f2,tang,a,b,lamb)
    B2 = B2_e (dlambx2,JRD,f1,f2)
    B3 = B3_e (dlambx3,JRD,f1,f2)
    
    # Problema Direto (Calcular o campo externo nas coordenadas geograficas)
    Bx = Bx_c (B1,B2,B3,l1,l2,l3)
    By = By_c (B1,B2,B3,m1,m2,m3)
    Bz = Bz_c (B1,B2,B3,n1,n2,n3)
    
    #Constante magnetica SI
    ctemag = 1.
    
    # Problema Direto (Calcular o campo externo nas coordenadas geograficas no SI)
    Bx = Bx*ctemag
    By = By*ctemag
    Bz = Bz*ctemag
    
    return Bx, By, Bz
    
def l1_v (alfa, delta):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    
    output:
    Direcao em radianos.
    '''
    
    l1 = (np.cos(alfa)*np.sin(delta))
    return l1

def l2_v (alfa, delta):

    '''
    Orientacao do elipsoide com respeito ao eixo y.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    l2 = (-np.cos(alfa)*np.cos(delta))
    return l2

def l3_v (alfa, delta):

    '''
    Orientacao do elipsoide com respeito ao eixo z.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    l3 = (-np.sin(alfa))
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
    
    m1 = (np.sin(alfa)*np.sin(delta))
    return m1

def m2_v (alfa, delta):

    '''
    Orientacao do elipsoide com respeito ao eixo y.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    m2 = (-np.sin(alfa)*np.cos(delta))
    return m2

def m3_v (alfa, delta):

    '''
    Orientacao do elipsoide com respeito ao eixo z.
    
    input:
    alfa - Azimute com relacao ao eixo-maior. (0<=alfa<=360)
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    m3 = (np.cos(alfa))
    return m3

def n1_v (delta):

    '''
    Orientacao do elipsoide com respeito ao eixo x.
    
    input:
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)

    output:
    Direcao em radianos.
    '''
    
    n1 = (-np.cos(delta))
    return n1

def n2_v (delta):

    '''
    Orientacao do elipsoide com respeito ao eixo y.
    
    input:
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    n2 = (-np.sin(delta))
    return n2

def n3_v (delta):

    '''
    Orientacao do elipsoide com respeito ao eixo z.
    
    input:
    delta - Inclinacao com relacao ao eixo-maior. (0<=delta<=90)
    gamma - Angulo entre o eixo-maior e a projecao vertical do centro do elipsoide com o plano. (-90<=gamma<=90)
    
    output:
    Direcao em radianos.
    '''
    
    n3 = 0
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
    F = intensidadeT*np.array([[(lt*l1+mt*m1+nt*n1)], [(lt*l2+mt*m2+nt*n2)], [(lt*l3+mt*m3+nt*n3)]])
    return F
    
def JN_e (intensidade,ln,mn,nn,l1,l2,l3,m1,m2,m3,n1,n2,n3):
    '''
    transformacao do Vetor de magnetizacao remanente para as coordenadas nos eixos do elipsoide.
    '''
    JN = intensidade*np.array([[(ln*l1+mn*m1+nn*n1)], [(ln*l2+mn*m2+nn*n2)], [(ln*l3+mn*m3+nn*n3)]])
    return JN

def N_desmag (a,b):
    '''
    Fator de desmagnetizacao ao longo do eixo de revolucao (N1) e em relacao ao plano equatorial (N2).
    '''
    N1 = ((4.*np.pi*a*b**2)/((b**2-a**2)**1.5)) * ((((b**2-a**2)**0.5)/(a)) - np.arctan(((b**2-a**2)**0.5)/(a)))
    N2 = 2.*np.pi - N1/2.
    return N1, N2
    
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
    
def JR_e (k,JN,F):
    '''
    Vetor de magnetizacao resultante sem correcao da desmagnetizacao.
    '''
    JR = k.dot(F) + JN
    return JR
    
def JRD_e (k,N1,N2,F,JR):
    '''
    Vetor de magnetizacao resultante com a correcao da desmagnetizacao.
    '''
    I = np.identity(3)
    kn0 = k[:,0]*N1
    kn1 = k[:,1]*N2
    kn2 = k[:,2]*N2
    kn = (np.vstack((kn0,kn1,kn2))).T
    A = I + kn
    JRD = (linalg.inv(A)).dot(JR)
    return JRD
    
def x1_e (X,Y,Z,xc,yc,h,l1,m1,n1):
    '''
    Calculo da coordenada x no elipsoide
    input:
    X,Y - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l1,m1,n1 - Orientacao do elipsoide (eixo x)
    output:
    x1 - Coordenada x do elipsoide.
    '''
    x1 = (X-xc)*l1+(Y-yc)*m1+(-Z-h)*n1
    return x1

def x2_e (X,Y,Z,xc,yc,h,l2,m2,n2):
    '''
    Calculo da coordenada y no elipsoide
    input:
    X,Y - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l2,m2,n2 - Orientacao do elipsoide (eixo y).
    output:
    x2 - Coordenada y do elipsoide.
    '''
    x2 = (X-xc)*l2+(Y-yc)*m2+(-Z-h)*n2
    return x2

def x3_e (X,Y,Z,xc,yc,h,l3,m3,n3):
    '''
    Calculo da coordenada z no elipsoide
    input:
    X,Y - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l3,m3,n3 - Orientacao do elipsoide (eixo z).
    output:
    x3 - Coordenada z do elipsoide.
    '''
    x3 = (X-xc)*l3+(Y-yc)*m3+(-Z-h)*n3
    return x3

def r_e (x1,x2,x3):
    '''
    Distancia do centro do elipsoide ao ponto observado
    input:
    X,Y - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l3,m3,n3 - Orientacao do elipsoide (eixo z).
    output:
    x3 - Coordenada z do elipsoide.
    '''
    r = ((x1)**2+(x2)**2+(x3)**2)**0.5
    return r
    
def delta_e (r,a,b,x1,x2,x3):
    '''
    Calculo auxiliar de lambda.
    input:
    X,Y - Matriz: Coordenadas geograficas (malha).
    h - Profundidade do elipsoide.
    l3,m3,n3 - Orientacao do elipsoide (eixo z).
    output:
    x3 - Coordenada z do elipsoide.
    '''

    delta = (r**4 + (a**2-b**2)**2 - 2*(a**2-b**2) * (x1**2 - x2**2 - x3**2))**0.5
    return delta    
    
def lamb_e (r,a,b,delta):
    '''
    Maior raiz real da equacao cubica: s^3 + p2*s^2 + p0 = 0
    input:
    p,p2 - constantes da equacao cubica
    teta - constante angular (radianos)
    output:
    lamb - Maior raiz real.
    '''
    lamb = (r**2 - a**2 - b**2 + delta)/2.
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

def dlambx1_e (a,b,x1,x2,x3,lamb,r,delta):
    '''
    Derivada de lamb em relacao ao eixo x1 do elipsoide.
    input:
    a,b,c, - semi-eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    dlambx1 - escalar
    '''
    dlambx1 = x1*(1+(r**2-a**2+b**2)/delta)
    return dlambx1

def dlambx2_e (a,b,x1,x2,x3,lamb,r,delta):
    '''
    Derivada de lamb em relacao ao eixo x2 do elipsoide.
    input:
    a,b,c, - semi-eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    dlambx2 - escalar
    '''
    dlambx2 = x2*(1+(r**2+a**2-b**2)/delta)
    return dlambx2

def dlambx3_e (a,b,x1,x2,x3,lamb,r,delta):
    '''
    Derivada de lamb em relacao ao eixo x3 do elipsoide.
    input:
    a,b,c, - semi-eixos do elipsoide.
    x1,x2,x3 - Eixo de coordenadas do elipsoide.
    lamb - Maior raiz real da equacao cubica.
    output:
    dlambx3 - escalar
    '''
    dlambx3 = x3*(1+(r**2+a**2-b**2)/delta)
    return dlambx3   

def f1_e (a,b,x1,x2,x3,lamb,JRD):
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
    
def tang_e (a,b,lamb):
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
    
def f2_e (a,b,tang,lamb):
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

def B1_e (dlambx1,JRD,f1,f2,tang,a,b,lamb):
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

def B2_e (dlambx2,JRD,f1,f2):
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
    
def B3_e (dlambx3,JRD,f1,f2):
    '''
    Calculo do campo magnetico (Bi) com relacao aos eixos do elipsoide 
    input:
    a,b,c - semi-eixos do elipsoide.
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