def _minimize_bfgs_twophase(fun, x0, args=(), jac=None, callback=None,
                   gtol=1e-6, norm=Inf, eps=_epsilon, maxiter=None,
                   disp=False, return_all=False, error_list=False,
                   **unknown_options):

    _check_unknown_options(unknown_options)
    f = fun
    fprime = jac
    epsilon = eps
    retall = return_all

    x0 = asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    func_calls, f = wrap_function(f, args)

    old_fval = f(x0)

    if fprime is None:
        grad_calls, myfprime = wrap_function(approx_fprime, (f, epsilon))
    else:
        grad_calls, myfprime = wrap_function(fprime, args)
    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = numpy.eye(N, dtype=int)
    Hk = I
    #H_k = I

    # Sets the initial step guess to dx ~ 1

    #old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk_bar = -np.dot(Hk, gfk)
        try:
            #alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
            #         _line_search_wolfe12(f, myfprime, xk, pk, gfk,
            #                              old_fval, old_old_fval, amin=1e-100, amax=1e100)
            alpha_kbar = armijo_search(f, xk, pk_bar, gfk)
            if alpha_kbar == None:
                warnflag = 2
                break
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break


        #Vector update:
        xk_bar = xk + alpha_kbar * pk_bar
        if retall:
            allvecs.append(xk_bar)

        #Component updates of the Hessian:
        #1. sk = difference of the parameters:
        sk = xk_bar - xk
        #New Updated vector:
        #xk = xkp1_bar
        #if gfkp1 is None:
        #    gfkp1 = myfprime(xkp1)

        #Calculating New Gradients
        gfk_bar = myfprime(xk_bar)

        #2. yk = difference of the gradients of the parameters
        yk = gfk_bar - gfk

        sTy = np.dot(sk.T, yk)
        #yts = np.dot(yk.T, sk)
        gnorm = vecnorm(gfk, ord=norm) #Is this not gfk_bar? = No
        if (gnorm) > 1e-2:
            w = 2
        else:
            w = 100

        if sTy < 0:
            sTs = np.dot(sk.T, sk)
            #xhi = w - (sTy / (sTs * gnorm))
            xhi = w - (sTy / (sTs * gnorm))
        else:
            xhi = w
        #snorm = vecnorm(sk, ord=norm)

        #xhi = w*gnorm + numpy.maximum([-numpy.dot(yk, sk)/ numpy.square(snorm)],0)

        #Transpose of yk???
        yk1 = yk + xhi*gnorm*sk
        '''gfk = gfk_bar

        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break'''

        if not numpy.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (numpy.dot(yk1, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        if isinf(rhok):  # this is patch for numpy
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")

        #Calculate Hessian
        A1 = I - sk[:, numpy.newaxis] * yk1[numpy.newaxis, :] * rhok
        A2 = I - yk1[:, numpy.newaxis] * sk[numpy.newaxis, :] * rhok
        H_k = numpy.dot(A1, numpy.dot(Hk, A2)) + (rhok * sk[:, numpy.newaxis] *
                                                 sk[numpy.newaxis, :])

        '''A1 = I - rhok * numpy.dot(sk, yk1.T)
        A2 = I - rhok * numpy.dot(yk1, sk.T)
        H_k = numpy.dot(A1, numpy.dot(Hk, A2)) + (rhok * numpy.dot(sk, sk.T))'''

        #Calculate New Hessian with lambda
        Hk = 0.5 * Hk + (1.0 - 0.5) * H_k
        pk = -np.dot(Hk, gfk)

        try:
            #alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
            #         _line_search_wolfe12(f, myfprime, xk, pk, gfk,
            #                              old_fval, old_old_fval, amin=1e-100, amax=1e100)
            alpha_k = armijo_search(f, xk, pk, gfk)
            if alpha_k == None:
                warnflag = 2
                break
        except _LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break
        xkpl = xk + alpha_k * pk
        gfkl = myfprime(xkpl)

        gfk = gfkl
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        xk = xkpl
        old_fval = f(xk)
        error_list.append(old_fval)

    fval = old_fval
    if np.isnan(fval):
        # This can happen if the first call to f returned NaN;
        # the loop is then never entered.
        warnflag = 2

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % func_calls[0])
        print("         Gradient evaluations: %d" % grad_calls[0])

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=func_calls[0],
                            njev=grad_calls[0], status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result
