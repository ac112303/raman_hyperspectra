'''
Vertex Component Analysis (VCA)

Recherche des endmembers https://github.com/Laadr/VCA

def vca(Y,R,verbose = True,snr_input = 0):
    return Ae,indice,Yp

Calcul de la matrice de mélange https://github.com/Laadr/SUNSAL/blob/master/sunsal.py

def sunsal(M,y,AL_iters=1000,lambda_0=0.,positivity=False,addone=False,tol=1e-4,x0 = None,verbose=False):
       return x,res_p,res_d,i
'''




__all__  = ['sunsal', 'vca']


def sunsal(M,y,AL_iters=1000,lambda_0=0.,positivity=False,addone=False,tol=1e-4,x0 = None,verbose=False):


    """
     SUNSAL -> sparse unmixing via variable splitting and augmented
     Lagrangian methods
    --------------- Description --------------------------------------------
     SUNSAL solves the following l2-l1 optimization  problem
     [size(M) = (L,p); size(X) = (p,N)]; size(Y) = (L,N)]
            min  (1/2) ||M X-y||^2_F + lambda ||X||_1
             X
     where ||X||_1 = sum(sum(abs(X)).
       CONSTRAINTS ACCEPTED:
       1) POSITIVITY:  X >= 0;
       2) ADDONE:  sum(X) = ones(1,N);
       NOTES:
          1) The optimization w.r.t each column of X is decoupled. Thus,
             SUNSAL solves N simultaneous problems.
          2) SUNSAL solves the following  problems:
             a) BPDN - Basis pursuit denoising l2-l1
                       (lambda > 0, POSITIVITY = 'no', ADDONE, 'no')
             b) CBPDN - Constrained basis pursuit denoising l2-l1
                       (lambda > 0, POSITIVITY = 'yes', ADDONE, 'no')
             c) CLS   - Constrained least squares
                        (lambda = 0, POSITIVITY = 'yes', ADDONE, 'no')
             c) FCLS   - Fully constrained least squares
                        (lambda >=0 , POSITIVITY = 'yes', ADDONE, 'yes')
                         In this case, the regularizer ||X||_1  plays no role,
                         as it is constant.
    -------------------- Line of Attack  -----------------------------------
     SUNSAL solves the above optimization problem by introducing a variable
     splitting and then solving the resulting constrained optimization with
     the augmented Lagrangian method of multipliers (ADMM).
            min  (1/2) ||M X-y||^2_F + lambda ||Z||_1
             X,Z
            subject to: sum(X) = ones(1,N)); Z >= 0; X = Z
     Augmented Lagrangian (scaled version):
          L(X,Z,D) = (1/2) ||M X-y||^2_F + lambda ||Z||_1 + mu/2||X-Z-D||^2_F
     where D are the scale Lagrange multipliers
     ADMM:
         do
           X  <-- arg min L(X,Z,D)
                       X, s.t: sum(X) = ones(1,N));
           Z  <-- arg min L(X,Z,D)
                       Z, s.t: Z >= 0;
           D  <-- D - (X-Z);
         while ~stop_rulde
    For details see
    [1] J. Bioucas-Dias and M. Figueiredo, "Alternating direction algorithms
    for constrained sparse regression: Application to hyperspectral unmixing",
    in 2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal
    Processing-WHISPERS'2010, Raykjavik, Iceland, 2010.
    ------------------------------------------------------------------------
    ====================== Required inputs =============
     M - [L(channels) x p(endmembers)] mixing matrix
     y - matrix with  L(channels) x N(pixels).
         each pixel is a linear mixture of p endmembers
         signatures y = M*x + noise,
    ====================== Optional inputs =============================
     AL_ITERS - Minimum number of augmented Lagrangian iterations - Default: 1000
     lambda_0 - regularization parameter. lambda is either a scalar
              or a vector with N components (one per column of x)
              Default: 0.
     positivity  = {True, False}; Enforces the positivity constraint: X >= 0 - Default: False
     addone  = {True, False}; Enforces the positivity constraint: X >= 0 - Default: False
     tol    - tolerance for the primal and  dual residuals - Default: 1e-4;
     verbose   = {True, False}; Default: False
    =========================== Outputs ==================================
    x      estimated mixing matrix [pxN]
    res_p  primal residual
    res_d  dual residual
    i      number of iteration until convergence
    ------------------------------------------------------------------
    Author: Jose Bioucas-Dias, 2009
    -------------------------------------------------------------------------
    Copyright (July, 2009):        José Bioucas-Dias (bioucas@lx.it.pt)
    SUNSAL is distributed under the terms of the GNU General Public License 2.0.
    Permission to use, copy, modify, and distribute this software for any purpose without fee is hereby granted, provided that this entire notice is included in all copies of any software which is or includes a copy or modification of this software and in all copies of the supporting documentation for such software.
    This software is being provided "as is", without any express or implied warranty.  In particular, the authors do not make any representation or warranty of any kind concerning the merchantability of this software or its fitness for any particular purpose."
    ---------------------------------------------------------------------
    Software translated from matlab to python by Adrien Lagrange (ad.lagrange@gmail.com), 2018.
    """
    import sys
    import scipy as sp
    import scipy.linalg as splin
    from numpy import linalg as LA

    #--------------------------------------------------------------
    # test for number of required parametres
    #--------------------------------------------------------------
    [LM,p] = M.shape # mixing matrixsize
    [L,N] = y.shape # data set size
    if LM != L:
        sys.exit('mixing matrix M and data set y are inconsistent')

    ##
    #--------------------------------------------------------------
    # Local variables
    #--------------------------------------------------------------

    #--------------------------------------------------------------
    # Read the optional parameters
    #--------------------------------------------------------------
    AL_iters = int(AL_iters)
    if (AL_iters < 0 ):
        sys.exit('AL_iters must a positive integer')

    # If lambda is scalar convert it into vector
    lambda_0 = ( lambda_0 * sp.ones((N,p)) ).T
    if (lambda_0<0).any():
        sys.exit('lambda_0 must be positive')

    # compute mean norm
    norm_m = splin.norm(M)*(25+p)/float(p)
    # rescale M and Y and lambda
    M = M/norm_m
    y = y/norm_m
    lambda_0 = lambda_0/norm_m**2

    if x0 is not None:
        if (x0.shape[0]==p) or (x0.shape[0]==N):
            sys.exit('initial X is not inconsistent with M or Y')


    #---------------------------------------------
    # just least squares
    #---------------------------------------------
    if (lambda_0.sum() == 0) and (not positivity) and (not addone):
        z = sp.dot(splin.pinv(M),y)
        # primal and dual residues
        res_p = 0.
        res_d = 0.
        return z,res_p,res_d,None

    #---------------------------------------------
    # least squares constrained (sum(x) = 1)
    #---------------------------------------------
    SMALL = 1e-12;
    if (lambda_0.sum() == 0) and (addone) and (not positivity):
        F = sp.dot(M.T,M)
        # test if F is invertible
        if LA.cond(F) > SMALL:
            # compute the solution explicitly
            IF = splin.inv(F);
            z = sp.dot(sp.dot(IF,M.T),y) - (1./IF.sum())*sp.dot(sp.sum(IF,axis=1,keepdims=True) , ( sp.dot(sp.dot(sp.sum(IF,axis=0,keepdims=True),M.T),y) - 1.))
            # primal and dual residues
            res_p = 0
            res_d = 0

            return z,res_p,res_d,None
        else:
            sys.exit('Bad conditioning of M.T*M')


    #---------------------------------------------
    #  Constants and initializations
    #---------------------------------------------
    mu_AL = 0.01
    mu = 10*lambda_0.mean() + mu_AL

    [UF,SF] = splin.svd(sp.dot(M.T,M))[:2]
    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
    x_aux = sp.sum(Aux,axis=1,keepdims=True)
    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))


    yy = sp.dot(M.T,y)

    #---------------------------------------------
    #  Initializations
    #---------------------------------------------

    # no intial solution supplied
    if x0 is None:
       x = sp.dot( sp.dot(IF,M.T) , y)
    else:
        x = x0

    z = x
    # scaled Lagrange Multipliers
    d  = 0*z

    #---------------------------------------------
    #  AL iterations - main body
    #---------------------------------------------
    tol1 = sp.sqrt(N*p)*tol
    tol2 = sp.sqrt(N*p)*tol
    i=1
    res_p = sp.inf
    res_d = sp.inf
    maskz = sp.ones(z.shape)
    mu_changed = 0

    #--------------------------------------------------------------------------
    # constrained  leat squares (CLS) X >= 0
    #--------------------------------------------------------------------------
    if (lambda_0.sum() ==  0)  and (not addone):
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            z = sp.maximum(x-d,0)
            # minimize with respect to x
            x = sp.dot(IF,yy + mu*(z+d))
            # Lagrange multipliers update
            d -= (x-z)

            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    # Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    # x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    # IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

        #--------------------------------------------------------------------------
        # Fully constrained  leat squares (FCLS) X >= 0
        #--------------------------------------------------------------------------
    elif (lambda_0.sum() ==  0)  and addone:
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            z = sp.maximum(x-d,0)
            # minimize with respect to x
            x = sp.dot(IF1,yy + mu*(z+d)) + x_aux
            # Lagrange multipliers update
            d -= (x-z)

            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

        #--------------------------------------------------------------------------
        # generic SUNSAL: lambda > 0
        #--------------------------------------------------------------------------
    else:
        # implement soft_th
        while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
            # save z to be used later
            if (i%10) == 1:
                z0 = z
            # minimize with respect to z
            nu = x-d
            z = sp.sign(nu) * sp.maximum(sp.absolute(nu) - lambda_0/mu,0)
            # teste for positivity
            if positivity:
                z = sp.maximum(z,0)
            # teste for sum-to-one
            if addone:
                x = sp.dot(IF1,yy+mu*(z+d)) + x_aux
            else:
                x = sp.dot(IF,yy+mu*(z+d))
            # Lagrange multipliers update
            d -= (x-z)

            # update mu so to keep primal and dual residuals whithin a factor of 10
            if (i%10) == 1:
                # primal residue
                res_p = splin.norm(x-z)
                # dual residue
                res_d = mu*splin.norm(z-z0)
                if verbose:
                    print("i = {:d}, res_p = {:f}, res_d = {:f}\n").format(i,res_p,res_d)
                # update mu
                if res_p > 10*res_d:
                    mu = mu*2
                    d = d/2
                    mu_changed = True
                elif res_d > 10*res_p:
                    mu = mu/2
                    d = d*2
                    mu_changed = True

                if  mu_changed:
                    # update IF and IF1
                    IF = sp.dot( sp.dot(UF,sp.diag(1./(SF+mu))) , UF.T )
                    Aux = (1./IF.sum()) * sp.sum(IF,axis=1,keepdims=True)
                    x_aux = sp.sum(Aux,axis=1,keepdims=True)
                    IF1 = IF - sp.dot(Aux,sp.sum(IF,axis=0,keepdims=True))
                    mu_changed = False

            i+=1

    return x,res_p,res_d,i





#############################################
# Internal functions
#############################################

def estimate_snr(Y,r_m,x):

    import scipy as sp

    [L, N] = Y.shape           # L number of bands (channels), N number of pixels
    [p, N] = x.shape           # p number of endmembers (reduced dimension)

    P_y     = sp.sum(Y**2)/float(N)
    P_x     = sp.sum(x**2)/float(N) + sp.sum(r_m**2)
    snr_est = 10*sp.log10( (P_x - p/L*P_y)/(P_y - P_x) )

    return snr_est



def vca(Y,R,verbose = True,snr_input = 0):

    '''
    Vertex Component Analysis
    #
    Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)

    Arguments:
      Y - matrix with dimensions L(channels) x N(pixels)
          each pixel is a linear mixture of R endmembers
          signatures Y = M x s, where s = gamma x alfa
          gamma is a illumination perturbation factor and
          alfa are the abundance fractions of each endmember.
      R - positive integer number of endmembers in the scene

    Returns:
     Ae     - estimated mixing matrix (endmembers signatures)
     indice - pixels that were chosen to be the most pure
     Yp     - Data matrix Y projected.

     ------- Optional parameters---------
     snr_input - (float) signal to noise ratio (dB)
     v         - [True | False]
     ------------------------------------

     Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
     This code is a translation of a matlab code provided by
     Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
     available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
     Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))

     more details on:
     Jose M. P. Nascimento and Jose M. B. Dias
     "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
     submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004

    '''
    import sys
    import scipy as sp
    import scipy.linalg as splin

  #############################################
  # Initializations
  #############################################
    if len(Y.shape)!=2:
        sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

    [L, N]=Y.shape   # L number of bands (channels), N number of pixels

    R = int(R)
    if (R<0 or R>L):
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')

  #############################################
  # SNR Estimates
  #############################################

    if snr_input==0:
        y_m = sp.mean(Y,axis=1,keepdims=True)
        Y_o = Y - y_m           # data with zero-mean
        Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:R]  # computes the R-projection matrix
        x_p = sp.dot(Ud.T, Y_o)                 # project the zero-mean data onto p-subspace

        SNR = estimate_snr(Y,y_m,x_p);

        if verbose:
            print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10*sp.log10(R)

  #############################################
  # Choosing Projective Projection or
  #          projection to p-1 subspace
  #############################################

    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")

        d = R-1
        if snr_input==0: # it means that the projection is already computed
            Ud = Ud[:,:d]
        else:
            y_m = sp.mean(Y,axis=1,keepdims=True)
            Y_o = Y - y_m  # data with zero-mean

            Ud  = splin.svd(sp.dot(Y_o,Y_o.T)/float(N))[0][:,:d]  # computes the p-projection matrix
            x_p =  sp.dot(Ud.T,Y_o)                 # project thezeros mean data onto p-subspace

        Yp =  sp.dot(Ud,x_p[:d,:]) + y_m      # again in dimension L

        x = x_p[:d,:] #  x_p =  Ud.T * Y_o is on a R-dim subspace
        c = sp.amax(sp.sum(x**2,axis=0))**0.5
        y = sp.vstack(( x, c*sp.ones((1,N)) ))
    else:
        if verbose:
            print("... Select the projective proj.")

        d = R
        Ud  = splin.svd(sp.dot(Y,Y.T)/float(N))[0][:,:d] # computes the p-projection matrix

        x_p = sp.dot(Ud.T,Y)
        Yp =  sp.dot(Ud,x_p[:d,:])      # again in dimension L (note that x_p has no null mean)

        x =  sp.dot(Ud.T,Y)
        u = sp.mean(x,axis=1,keepdims=True)        #equivalent to  u = Ud.T * r_m
        y =  x / sp.dot(u.T,x)


  #############################################
  # VCA algorithm
  #############################################

    indice = sp.zeros((R),dtype=int)
    A = sp.zeros((R,R))
    A[-1,0] = 1

    for i in range(R):
        w = sp.random.rand(R,1);
        f = w - sp.dot(A,sp.dot(splin.pinv(A),w))
        f = f / splin.norm(f)

        v = sp.dot(f.T,y)

        indice[i] = sp.argmax(sp.absolute(v))
        A[:,i] = y[:,indice[i]]        # same as x(:,indice(i))

    Ae = Yp[:,indice]

    return Ae,indice,Yp

