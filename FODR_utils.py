#!/usr/bin/env python
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pickle
# import matplotlib.pylab as pl
import warnings
from functools import partial
from scipy import optimize
import time
# from statistics import NormalDist
from scipy import signal
from scipy import stats
from scipy.integrate import quad
import pandas as pd
# import numpy.polynomial.polynomial as poly
from scipy.interpolate import interp1d
import sympy as sy
from sympy import nsimplify , simplify , exp, sqrt ,log,re ,im,lambdify,log
# from sympy import oo, Symbol, integrate
# from scipy.optimize import curve_fit
import os



def compute_meanfield_n(input_output,N,mu,p_val,m_val):
    c = 1-p_val
    S1=(1-c) / ( 1-(mu*m_val*c)-( c*(m_val**2)*mu*(1-mu)/(1-m_val*(1-mu)) ) )
    S2=(m_val*mu/(1-m_val*(1-mu)))* S1
    if  input_output=='inSub1_outSub1':
        return N*mu*S1
    elif input_output=='inSub1_outSub2' or input_output=='inSub1_outSub3':
        return N*(1-mu)*S2
    elif input_output=='inSub1_outAll':
        return N*(mu*S1+(1-mu)*S2)
    elif input_output=='inAll_outAll':
        return N*((1 - c) / (1 - m_val * c))
    else:
        print('invalid input_output')

def compute_meanfield_s(input_output,mu,p_val,m_val):
    c = 1-p_val
    S1=(1-c) / ( 1-(mu*m_val*c)-( c*(m_val**2)*mu*(1-mu)/(1-m_val*(1-mu)) ) )
    S2=(m_val*mu/(1-m_val*(1-mu)))* S1
    if  input_output=='inSub1_outSub1':
        return S1
    elif input_output=='inSub1_outSub2':
        return S2
    elif input_output=='inSub1_outAll' or input_output=='inSub1_outSub3':
        return mu*S1+(1-mu)*S2
    elif input_output=='inAll_outAll':
        return (1 - c) / (1 - m_val * c)
    else:
        print('invalid input_output')

#
def sigma_dist(input_output,N,mu,p_val,m_val,G_eval):
    n = sy.symbols('n', real=True, imaginary=False)
    mean=compute_meanfield_n(input_output,N,mu,p_val,m_val)
    return float(sqrt(G_eval.evalf(subs={n: mean})))


def compute_dist_params(input_output,N,mu,m_val,p_val):
    if input_output=='inAll_outAll':
        if m_val==0:
            a_val = -1
            b_val=0
            c_val = (1 - 2*p_val)
            d_val = p_val * N
        else:
            a_val=m_val*(1-p_val)-1
            b_val=2*m_val*(p_val-1)/N
            c_val=m_val*(1-p_val)-2*p_val+1
            d_val = p_val*N

    elif input_output=='inSub1_outSub1':
        if m_val==0:
            a_val = -1
            b_val=0
            c_val = (1 - 2*p_val)
            d_val = p_val * mu*N
        else:
            A=m_val*mu/(1-(1-mu)*m_val)
            B=m_val*(mu+(1-mu)*A)
            a_val=B*(1-p_val)-1
            b_val=-2*B*(1-p_val)/(N*mu)
            c_val=B*(1-p_val)-2*p_val+1
            d_val=N*mu*p_val
    elif input_output=='inSub1_outSub2':
        # print('inSub1_outSub2')
        A = (1 - mu) * m_val / (1 / (1 - p_val) - mu * m_val)
        B = (p_val / (1 - p_val)) / (1 / (1 - p_val) - mu * m_val)
        a_val = m_val * mu * A + m_val * (1 - mu) - 1
        b_val = -2 * m_val * (mu * A + (1 - mu)) / (N * (1 - mu))
        c_val = 1 + m_val * mu * A + m_val * (1 - mu) - 2 * m_val * mu * B
        d_val = m_val * mu * B * (1 - mu) * N
    else:
        print('invalid input_output')
    return a_val,b_val,c_val, d_val


def prob_dist_analytic(s_val,expression_eval):
    n = sy.symbols('n', real=True, imaginary=False)
    return expression_eval.evalf(subs={n: s_val})


def funcmin_interp(x,xx1,pdf1,xx2,pdf2):
    if x>xx1[0] and x<xx1[-1]:
        f1=interp1d(xx1, pdf1)
        p1=f1(x)
    else:
        p1=0
    if x>xx2[0] and x<xx2[-1]:
        f2=interp1d(xx2, pdf2)
        p2=f2(x)
    else:
        p2 = 0
    return min(p1,p2)


def add_gaussian_noise_to_beta(a,b,sigma_noise):
    k_signal = 5;k_noise = 5
    n = 100000
    mean = stats.beta.mean(a, b);
    sigma = stats.beta.std(a, b)
    bound_left = mean - max(k_signal*sigma,k_noise*sigma_noise)
    bound_right = mean + max(k_signal*sigma,k_noise*sigma_noise)
    x = np.linspace(bound_left, bound_right, n)
    temp=np.zeros(n)
    temp[np.arange(n)[np.logical_and(x>0,x<1)]]= stats.beta.pdf(x[np.logical_and(x>0,x<1)], a, b)
    pdf_original = temp/ (sum(temp)*(x[1]-x[0]))
    gaussian_noise = stats.norm.pdf(x, mean, sigma_noise)
    temp_conv = signal.convolve(pdf_original, gaussian_noise, mode='same')
    prob = temp_conv / sum(temp_conv)
    p_at_one = sum(prob[np.where(x >= 1)[0]])
    p_at_zero = sum(prob[np.where(x <= 0)[0]])
    pdf_final = prob / (x[1] - x[0])  # so that the integral is 1
    return p_at_zero, p_at_one, x , pdf_final


def analytic_pdf_helper(num_sample,step='None',*params):
    k_signal = 5;k_noise = 5;n_short=1000;eps=1e-4
    N, N_sub, input_output, m_val, mu, sigma_noise,logh,expr,G = params
    sigma_noise_adapted = sigma_noise * N
    n, a, b, c, d = sy.symbols(['n', 'a', 'b', 'c', 'd'], real=True, imaginary=False)
    p_val = 1 - np.exp(-1 * np.power(10, logh))
    a_val, b_val, c_val, d_val = compute_dist_params(input_output, N, mu, m_val, p_val)
    expr_eval = expr.evalf(subs={a: a_val, b: b_val, c: c_val, d: d_val})
    G_eval = G.evalf(subs={b: b_val, c: c_val, d: d_val})
    minn = 0;maxx = N_sub - 1
    mean_n = compute_meanfield_n(input_output, N, mu, p_val, m_val)
    n = sy.symbols('n', real=True, imaginary=False)
    f = lambdify(n, expr_eval, 'numpy')
    n_list = np.linspace(minn, maxx, n_short)
    ##dealing with low values of p
    temp_logp = f(n_list)
    temp_p = np.exp(temp_logp - np.max(temp_logp[~np.isnan(temp_logp)]))
    toy_p = temp_p / np.sum(temp_p[~np.isnan(temp_p)])
    low_values_flags = toy_p > eps
    sigma = n_list[low_values_flags][-1] - n_list[low_values_flags][0]
    bound_left = mean_n - max(k_signal * sigma, k_noise * sigma_noise_adapted)
    bound_right = mean_n + max(k_signal * sigma, k_noise * sigma_noise_adapted)
    if step=='None':
        x_disc = np.linspace(bound_left, bound_right, num_sample)
    else:
        x_disc=np.arange(bound_left, bound_right, step)
    ind = [np.logical_and(x_disc >= minn, x_disc <= maxx)][0]
    temp_logp = f(x_disc[ind])
    temp_p = np.exp(temp_logp - np.max(temp_logp[~np.isnan(temp_logp)]))
    prob = np.zeros_like(x_disc)
    prob[ind] = temp_p / np.sum(temp_p[~np.isnan(temp_p)])
    x = x_disc / N_sub
    pdf = prob / (x[1] - x[0])
    mean = mean_n / N_sub
    return x_disc, prob


def evaluate_dist(input_output,N,mu,m_val,p_val,expr,G):
    n, a, b, c, d = sy.symbols(['n', 'a', 'b', 'c', 'd'], real=True, imaginary=False)
    a_val, b_val, c_val, d_val = compute_dist_params(input_output, N, mu, m_val, p_val)
    expr_eval = expr.evalf(subs={a: a_val, b: b_val, c: c_val, d: d_val})
    G_eval = G.evalf(subs={b: b_val, c: c_val, d: d_val})
    return expr_eval,G_eval

def generate_discrete_pdf_analytic(right_boundary,expr,G,logh,*params):
    plot=0; step='None'
    k_signal = 5;k_noise = 5;num_sample = 100000
    N, input_output, m_val, mu, sigma_noise = params
    sigma_noise_adapted = sigma_noise * N
    p_val = 1 - np.exp(-1 * np.power(10, logh))
    if input_output=='inAll_outAll':
        expr_eval,G_eval=evaluate_dist(input_output,N,mu,m_val,p_val,expr,G)
        mean_n = compute_meanfield_n(input_output,N, mu, p_val, m_val)
        sigma = sigma_dist(input_output, mu, p_val, m_val, G_eval)
        bound_left = mean_n - max(k_signal * sigma, k_noise * sigma_noise_adapted)
        bound_right = mean_n + max(k_signal * sigma, k_noise * sigma_noise_adapted)
        x_disc= np.linspace(bound_left, bound_right, num_sample)
        temp = np.array([prob_dist_analytic(0,N,x_disc[i], expr_eval) for i in range(num_sample)])
        x=x_disc/(N)
        pdf= (temp / sum(temp)) / (x[1]-x[0])
        mean=mean_n/N
    if input_output=='inSub1_outSub1' or input_output=='inSub1_outSub2':
        params2 = [N, N*mu, input_output, m_val, mu, sigma_noise, logh, expr, G]
        x_disc, prob = analytic_pdf_helper(num_sample,step,*params2)
        x = x_disc / (N*mu)
        pdf = prob / (x[1] - x[0])

    if input_output == 'inSub1_outAll':
        expr_eval, G_eval = evaluate_dist('inSub1_outSub1',N,mu,m_val,p_val,expr,G)
        mean_n1 = compute_meanfield_n('inSub1_outSub1',N, mu, p_val, m_val)
        mean_n2 = compute_meanfield_n('inSub1_outSub2', N, mu, p_val, m_val)

        if m_val != 0:
            params = [N, N*mu, 'inSub1_outSub1', m_val, mu, sigma_noise, logh, expr, G]
            x1_discrete, prob1 = analytic_pdf_helper(num_sample,step,*params)
            params = [N, N*(1-mu), 'inSub1_outSub2', m_val, mu, sigma_noise, logh, expr, G]
            step=x1_discrete[1]-x1_discrete[0]
            x2_discrete, prob2 = analytic_pdf_helper(num_sample,step,*params)
            if right_boundary == 1:
                mean = (mean_n1 + mean_n2)/N
                x_discrete= x2_discrete + mean_n1
                x = x_discrete/ N
                pdf = prob2 / (x[1] - x[0])
            elif right_boundary == 0:
                temp_conv = signal.convolve(prob1, prob2, mode='same')
                prob = temp_conv / sum(temp_conv)
                mean = (mean_n1 + mean_n2) / N
                x_discrete = x1_discrete + ((x2_discrete[-1]+x2_discrete[0])/2)
                x = x_discrete / N
                pdf= prob / (x[1] - x[0])  # so that the integral is 1
            if plot == 1:
                plt.figure()
                plt.plot(x1_discrete, prob1,color='blue')
                plt.plot(x2_discrete, prob2,color='orange')
                plt.plot(x_discrete, prob,color='green')
            if plot == 1:
                plt.figure()
                plt.plot(x,pdf)
        elif m_val == 0:
            if right_boundary == 0:
                params = [N, N*mu, 'inSub1_outSub1', m_val, mu, sigma_noise, logh, expr, G]
                x1_discrete, prob1 = analytic_pdf_helper( num_sample,*params)
                x=x1_discrete/N
                pdf=prob1 /(x[1]-x[0])
                mean=mean_n1/N
    else:
        print('invalid input_output')
    return x,pdf,mean


def add_gaussian_noise_to_analytic_dist( input_output,N,right_boundary,sigma_noise,*dist_params):
    logh,m_val, mu, expr, G=dist_params
    params=[N, input_output, m_val, mu, sigma_noise]
    x,pdf_discrete,mean=generate_discrete_pdf_analytic(right_boundary,expr,G,logh,*params)
    gaussian_noise = stats.norm.pdf(x, mean, sigma_noise)
    temp_conv = signal.convolve(pdf_discrete, gaussian_noise, mode='same')
    prob = temp_conv / sum(temp_conv)
    p_at_one = sum(prob[np.where(x >= 1)[0]])
    p_at_zero = sum(prob[np.where(x <= 0)[0]])
    pdf_final = prob / (x[1] - x[0])  # so that the integral is 1
    return p_at_zero, p_at_one, x , pdf_final


def generate_gaussian_pdf(mu,sigma_noise):
    k_noise=5
    n=100000
    x= np.linspace(mu - k_noise * sigma_noise, mu + k_noise * sigma_noise, n)
    temp_pdf = stats.norm.pdf(x, mu, sigma_noise)
    prob = temp_pdf / sum(temp_pdf)
    p_at_one = sum(prob[np.where(x >= 1)[0]])
    p_at_zero = sum(prob[np.where(x <= 0)[0]])
    pdf = prob / (x[1] - x[0])
    return p_at_zero,p_at_one,x,pdf


def compute_first_logh(*params,plot,ovelrlap_with,s_analytic,logh_data,loga_data,logb_data,logh_init):
    N, N_sub, input_output, m_val, mu, sigma_noise, error=params
    func = partial(compute_overlap_with_boundaries,input_output, ovelrlap_with,m_val, sigma_noise, error,m_val, logh_init, s_analytic,
                       loga_data,logb_data,logh_data)
    if ovelrlap_with=='left':
        maxDelta = max(logh_data) - min(logh_data) - 1e-12
    if ovelrlap_with == 'right':
        maxDelta = -1 * (max(logh_data) - min(logh_data) - 1e-12)

    k=100
    s=np.linspace(0, maxDelta, k)
    r=np.zeros(k)
    for i in range(k):
        print(i)
        r[i]=func(s[i])
    f = interp1d(r, s)
    delta_logh = f(0)
    if plot==1:
        plt.figure()
        plt.scatter(s,r)
        plt.plot([delta_logh,delta_logh],[min(r),max(r)],'--')
        print('delta_logh=',delta_logh)
    return logh_init+delta_logh


def compute_first_logh_analytic(plot,inf,ovelrlap_with,expr,G,logh_data,logh_init,*params):
    N, N_sub, input_output, m_val, mu, sigma_noise, error=params
    func = partial(compute_overlap_with_boundaries_analytic,inf,ovelrlap_with,input_output,N, m_val, mu, sigma_noise, error,expr,G,logh_init,logh_data)
    if ovelrlap_with=='left':
        maxDelta = max(logh_data) - min(logh_data) - 1e-12
    if ovelrlap_with == 'right':
        maxDelta = -1 * (max(logh_data) - min(logh_data) - 1e-12)
    k=20
    r = np.zeros(k)
    s=np.linspace(0, maxDelta, k)
    for i in range(k):
        print(i)
        r[i] = func(s[i])
    f = interp1d(r, s)
    delta_logh = f(0)
    if plot==1:
        plt.figure()
        plt.scatter(s, r)
        plt.plot(s, r)
        plt.plot([delta_logh, delta_logh], [min(r), max(r)], '--')

    return logh_init+delta_logh



def compute_overlap_with_boundaries(ovelrlap_with,input_output,sigma_noise,error,m_val,logh_init,s_analytic,logh_data,loga_data,logb_data,delta_logh):
    # N, N_sub, input_output, m_val, mu, sigma_noise, error=params
    f_a = interp1d(logh_data, loga_data);f_b = interp1d(logh_data, logb_data)
    a = np.power(10, f_a(logh_init + delta_logh));b = np.power(10, f_b(logh_init + delta_logh))
    p_at_zero, p_at_one, x, pdf = add_gaussian_noise_to_beta(a, b, sigma_noise)
    if ovelrlap_with == 'right':
        if input_output=='inAll_outAll':
            p_at_zero_maxRight, p_at_one_maxRight, x_maxRight, pdf_maxRight = generate_gaussian_pdf(1,sigma_noise)
        elif input_output=='inSub1_outAll' or input_output=='inSub1_outSub3': #fix later for different input types:
            if m_val!=0:
                a = np.power(10,loga_data[-1]);b = np.power(10, logb_data[-1])
                p_at_zero_maxRight,p_at_one_maxRight, x_maxRight, pdf_maxRight = add_gaussian_noise_to_beta(a, b, sigma_noise)
            else:
                p_at_zero_maxRight, p_at_one_maxRight, x_maxRight, pdf_maxRight = generate_gaussian_pdf(s_analytic[-1],
                                                                                                        sigma_noise)

        return (min(p_at_zero_maxRight, p_at_zero) + min(p_at_one_maxRight, p_at_one) +
                quad(funcmin_interp, max(x_maxRight[0], x[0], 0), min(x_maxRight[-1], x[-1], 1),
                     args=( x, pdf,x_maxRight, pdf_maxRight))[0]) / 2 - error
    elif ovelrlap_with == 'left':
        p_at_zero_maxLeft,p_at_one_maxLeft,  x_maxLeft, pdf_maxLeft = generate_gaussian_pdf(0, sigma_noise)
        return (min(p_at_zero_maxLeft, p_at_zero) +min(p_at_one_maxLeft, p_at_one)+
                quad(funcmin_interp, max(x_maxLeft[0], x[0], 0), min(x[-1], x_maxLeft[-1], 1), args=(x, pdf, x_maxLeft, pdf_maxLeft))[
                    0]) / 2 - error


def compute_overlap_with_boundaries_analytic(inf,ovelrlap_with,input_output,N, m_val, mu, sigma_noise, error,expr,G,logh_init,logh_data,delta_logh):
    if inf==1:
        p_val = 1 - np.exp(-1 * np.power(10, logh_init + delta_logh))
        mean2 = compute_meanfield_s(input_output, mu, p_val, m_val)
        p_at_zero, p_at_one, x, pdf = generate_gaussian_pdf(mean2, sigma_noise)

        if ovelrlap_with == 'right':
            if input_output=='inAll_outAll':
                p_at_zero_maxRight,p_at_one_maxRight,  x_maxRight, pdf_maxRight=generate_gaussian_pdf(1,sigma_noise)
            else:
                p_val = 1 - np.exp(-1 * np.power(10, logh_data[-1]))
                mean_max = compute_meanfield_s(input_output, mu, p_val, m_val)
                p_at_zero_maxRight,p_at_one_maxRight,  x_maxRight, pdf_maxRight = generate_gaussian_pdf(mean_max,sigma_noise)
            return (min(p_at_zero_maxRight, p_at_zero) + min(p_at_one_maxRight, p_at_one) +
                    quad(funcmin_interp, max(x[0], x_maxRight[0], 0), min(x[-1], x_maxRight[-1], 1),
                         args=(x, pdf, x_maxRight, pdf_maxRight))[0]) / 2 - error
        elif ovelrlap_with == 'left':
            p_at_zero_maxLeft, p_at_one_maxLeft, x_maxLeft, pdf_maxLeft = generate_gaussian_pdf(0, sigma_noise)
            return (min(p_at_zero_maxLeft, p_at_zero) +min(p_at_one_maxLeft, p_at_one)+ quad(funcmin_interp, 0, min(x[-1], x_maxLeft[-1], 1),
                                                 args=(x, pdf, x_maxLeft, pdf_maxLeft))[0]) / 2 - error

    elif inf==0:
        dist_params = [logh_init+delta_logh, m_val, mu, expr, G]
        right_boundary=0
        p_at_zero, p_at_one, x, pdf = add_gaussian_noise_to_analytic_dist(input_output,N,right_boundary, sigma_noise,*dist_params)
        if ovelrlap_with == 'right':
            if input_output=='inSub1_outAll' or input_output=='inSub1_outSub2' :
                if m_val!=0:
                    dist_params = [logh_data[-1], m_val, mu, expr, G]
                    right_boundary=1
                    p_at_zero_maxRight, p_at_one_maxRight, x_maxRight, pdf_maxRight = add_gaussian_noise_to_analytic_dist(input_output,N, right_boundary, sigma_noise,*dist_params)
                else:
                    p_val = 1 - np.exp(-1 * np.power(10, logh_data[-1]))
                    mean_max = compute_meanfield_s(input_output, mu, p_val, m_val)
                    p_at_zero_maxRight, p_at_one_maxRight, x_maxRight, pdf_maxRight= generate_gaussian_pdf(mean_max, sigma_noise)

                return (min(p_at_zero_maxRight, p_at_zero) + min(p_at_one_maxRight, p_at_one) +
                        quad(funcmin_interp, max(x_maxRight[0], x[0], 0), min(x_maxRight[-1], x[-1], 1),
                             args=(x_maxRight, pdf_maxRight, x, pdf))[0]) / 2 - error
        elif ovelrlap_with == 'left':
            p_at_zero_maxLeft,p_at_one_maxLeft,  x_maxLeft, pdf_maxLeft = generate_gaussian_pdf(0, sigma_noise)
            return (min(p_at_zero_maxLeft, p_at_zero) +
                    quad(funcmin_interp, 0, min(x[-1], x_maxLeft[-1], 1), args=(x, pdf, x_maxLeft, pdf_maxLeft))[
                        0]) / 2 - error


def find_overlap_with_next_point(sigma_noise,error,logh1,loga_data,logb_data,logh_data,delta_logh):
    f_a = interp1d(logh_data, loga_data);f_b = interp1d(logh_data, logb_data)
    a1 = np.power(10, f_a(logh1));b1 = np.power(10, f_b(logh1))
    a2 = np.power(10, f_a(logh1 + delta_logh));b2 = np.power(10, f_b(logh1 + delta_logh))
    p1_at_zero, p1_at_one, x1, pdf1 = add_gaussian_noise_to_beta(a1, b1, sigma_noise)
    p2_at_zero, p2_at_one, x2, pdf2 = add_gaussian_noise_to_beta(a2, b2, sigma_noise)
    return (min(p1_at_zero, p2_at_zero)+min(p1_at_one,p2_at_one) + quad(funcmin_interp, max(min(x1[0],x2[0]),0), min(max(x1[-1], x2[-1]),1),
                                          args=(x1, pdf1, x2, pdf2))[0]) / 2 - error


def find_overlap_with_next_point_analytic(input_output,inf, p1_at_zero, p1_at_one, x1, pdf1 ,N,m_val,mu,sigma_noise,error,logh1, expr,G,delta_logh):
    if inf == 1:
        p_val = 1 - np.exp(-1 * np.power(10, logh1 + delta_logh))
        mean2 = compute_meanfield_s(input_output, mu, p_val, m_val)
        p2_at_zero, p2_at_one, x2, pdf2 = generate_gaussian_pdf(mean2, sigma_noise)
    else:
        right_boundary = 0
        dist_params = [logh1+delta_logh, m_val, mu, expr, G]
        p2_at_zero, p2_at_one, x2, pdf2 = add_gaussian_noise_to_analytic_dist(input_output,N,right_boundary, sigma_noise,
                                                                         *dist_params)
    return (min(p1_at_zero, p2_at_zero)+min(p1_at_one,p2_at_one) + quad(funcmin_interp, max(min(x1[0],x2[0]),0), min(max(x1[-1], x2[-1]),1),
                                          args=(x1, pdf1, x2, pdf2))[0]) / 2 - error


def find_all_discriminable_inputs(overlap_with,logh_max,logh_min,loga_data, logb_data ,logh_data,*params):
    N, N_sub, input_output, m_val, mu, sigma_noise, error = params
    g=0
    logh_selected = []
    if overlap_with=='left':
        logh = logh_min
        logh_selected.append(logh)
        # find the discriminable points starting from left
        while logh <= logh_max:
            try:
                func = partial(find_overlap_with_next_point, sigma_noise, error, logh,loga_data, logb_data,logh_data)
                maxDelta = max(logh_data) - logh - 1e-8
                minDelta = max(min(logh_data) - logh, 0)
                delta_logh = optimize.bisect(func, minDelta, maxDelta)
                g += 1
                print(g)
                logh = logh + delta_logh
                logh_selected.append(logh)
            except Exception as e:
                print(e)
                break
    # find the discriminable points starting from right
    if overlap_with=='right':
        logh = logh_max
        logh_selected.append(logh)
        while logh >= logh_min:
            try:
                func = partial(find_overlap_with_next_point, sigma_noise, error, logh,loga_data, logb_data,logh_data)
                maxDelta = -1 * (logh - min(logh_data) - 1e-8)
                minDelta=0
                delta_logh = optimize.bisect(func, minDelta, maxDelta)
                g += 1
                print(g)
                logh = logh + delta_logh
                logh_selected.append(logh)
            except Exception as e:
                print(e)
                break
    return logh_selected


def find_all_discriminable_inputs_analytic(inf,overlap_with,logh_max,logh_min, s_analytic,logh_data,expr,G,*params):
    N, N_sub, input_output, m_val, mu, sigma_noise, error = params
    g=0
    right_boundary=0
    logh_selected = []
    if overlap_with == 'left':
        logh = logh_min
        logh_selected.append(logh)
        # find the discriminable points starting from left
        while logh <= logh_max:
            if inf==1:
                # f_mu = interp1d(logh_data, s_analytic);mu1 = f_mu(logh)
                p_val = 1 - np.exp(-1 * np.power(10, logh))
                mean=compute_meanfield_s(input_output, mu, p_val, m_val)
                p1_at_zero, p1_at_one, x1, pdf1 = generate_gaussian_pdf(mean, sigma_noise)
            else:
                dist_params = [logh, m_val, mu, expr, G]
                p1_at_zero, p1_at_one, x1, pdf1 = add_gaussian_noise_to_analytic_dist( input_output,N,right_boundary, sigma_noise,*dist_params)

            try:
                func = partial(find_overlap_with_next_point_analytic, input_output, inf, p1_at_zero, p1_at_one, x1, pdf1 ,N, m_val, mu, sigma_noise,
                               error, logh, s_analytic, logh_data, expr, G)
                maxDelta = max(logh_data) - logh - 1e-8
                minDelta = max(min(logh_data) - logh, 0)
                delta_logh = optimize.bisect(func, minDelta, maxDelta)
                g += 1
                print(g)
                logh = logh + delta_logh
                logh_selected.append(logh)
            except Exception as e:
                print(e)
                break
    # find the discriminable points starting from right
    elif overlap_with == 'right':
        logh = logh_max
        logh_selected.append(logh)
        while logh >= logh_min:
            if inf==1:
                p_val = 1 - np.exp(-1 * np.power(10, logh))
                mean=compute_meanfield_s(input_output, mu, p_val, m_val)
                p1_at_zero, p1_at_one, x1, pdf1 = generate_gaussian_pdf(mean, sigma_noise)
            else:
                dist_params = [logh, m_val, mu, expr, G]
                p1_at_zero, p1_at_one, x1, pdf1 = add_gaussian_noise_to_analytic_dist( input_output,N,right_boundary,sigma_noise,*dist_params)
            try:
                func = partial(find_overlap_with_next_point_analytic, input_output, inf, p1_at_zero, p1_at_one, x1, pdf1 , N, m_val, mu, sigma_noise,
                               error, logh, s_analytic, logh_data, expr, G)
                maxDelta = -1 * (logh - min(logh_data) - 1e-8)
                minDelta = 0
                delta_logh = optimize.bisect(func, minDelta, maxDelta)
                g += 1
                print(g)
                logh = logh + delta_logh
                logh_selected.append(logh)
            except Exception as e:
                print(e)
                break

    return logh_selected


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    taken from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    """

    if x.ndim != 1:
        raise ValueError( "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError( "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def compute_FODR(typ_computation,only_FODR,sub_typ,n_realization,input_output,path_data,path_save,eps_list,sigma_noise,window,error,l):
    N=10000; mu=0.2
    if input_output=='inSub1_outAll' or input_output=='inAll_outAll' :
        N_sub=10000
    elif input_output=='inSub1_outSub1' or input_output=='inSub1_outSub3':
        N_sub=2000
    elif input_output=='inSub1_outSub2':
        N_sub=8000
    else:
        print('invalid input_output')

    epsilon=eps_list[l]
    m_val=1-epsilon
    print('epsilon=', epsilon)
    if epsilon==1:
        n, a, c, d= sy.symbols(['n', 'a', 'c', 'd'], real=True, imaginary=False)
        G = ( c * n + d)
        intgr_output = 2*(d*(c-a)*log(c*n+d)+a*c*n)/c**2
    else:
        n, a, b, c, d = sy.symbols(['n', 'a', 'b', 'c', 'd'], real=True, imaginary=False)
        G = (b * n ** 2 + c * n + d)
        intgr_output = (((a / b) * log(G )) - (((a * c - 2 * b * d) / (b * sqrt(-4 * b * d + c ** 2))) * log(
            (sqrt(-4 * b * d + c ** 2) - (2 * b * n + c)) / (sqrt(-4 * b * d + c ** 2) + (2 * b * n + c)))))
    expr = intgr_output - intgr_output.evalf(subs={n: 0})-log(G)
    for realization_count in range(n_realization):
        print('n_realization=',realization_count)
        col_names=['epsilon','sigma_noise','FODR_inf','no. points_inf','FODR','no. points','FODR_theo','no. points_theo','logh from left_inf','logh from right_inf','logh from left','logh from right','logh from left_theo','logh from right_theo',]
        dat=[[epsilon,sigma_noise,0,0,0,0,0,0,'','','','','','']]
        df_discriminated=pd.DataFrame(dat,columns = col_names)
        df_discriminated['logh from left']=df_discriminated['logh from left'].astype('object')
        df_discriminated['logh from right'] = df_discriminated['logh from right'].astype('object')
        df_discriminated['logh from left_inf']=df_discriminated['logh from left_inf'].astype('object')
        df_discriminated['logh from right_inf'] = df_discriminated['logh from right_inf'].astype('object')
        df_discriminated['logh from left_theo'] = df_discriminated['logh from left_theo'].astype('object')
        df_discriminated['logh from right_theo'] = df_discriminated['logh from right_theo'].astype('object')
        logh_list = np.arange(-7, 1, 0.05).tolist()
        h_list = np.power(10, logh_list)
        p_list=1-np.exp(-1*h_list)
        s_analytic = compute_meanfield_n(input_output,N, mu, p_list, m_val)/N_sub
        params = [N,N_sub,input_output,m_val,mu,sigma_noise,error]
        plot=1
        if typ_computation=='theo':
            inf=0
            print('theo')
            print('left')
            logh_maxLeft = compute_first_logh_analytic(plot, inf, 'left', expr, G,logh_list, logh_list[0],*params)
            print('right')
            logh_maxRight = compute_first_logh_analytic(plot, inf, 'right', expr, G,logh_list,logh_list[-1],*params)
            FODR_theo=10 * (logh_maxRight - logh_maxLeft)
            print('FODR_theo=',FODR_theo)
            df_discriminated.loc[0, 'FODR_theo'] = FODR_theo
            df_discriminated.to_pickle(
                path_save + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                    realization_count) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')
            if only_FODR==0:

                print('logh from left')
                logh_selected_from_left_theo =find_all_discriminable_inputs_analytic(inf,'left', logh_maxRight, logh_maxLeft, s_analytic,
                                               logh_list,expr, G,*params)
                df_discriminated.at[0, 'logh from left_theo'] = logh_selected_from_left_theo
                logh_selected_from_right_theo =find_all_discriminable_inputs_analytic(inf,'right', logh_maxRight, logh_maxLeft, s_analytic,
                                               logh_list,expr, G,*params)
                print('logh from right')
                N_d_theo=(len(logh_selected_from_right_theo )+len(logh_selected_from_left_theo ))/2
                df_discriminated.loc[0, 'no. points_theo'] = N_d_theo

                df_discriminated.at[0, 'logh from right_theo'] = logh_selected_from_right_theo
                df_discriminated.to_pickle(
                    path_save + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                        realization_count) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')

        ###find parameters for T=window
        if typ_computation=='simu':
            df_params = pd.read_pickle(
                path_data + 'epsilon=' + '{:.2e}'.format(epsilon) + '_' + sub_typ + '_window=' + str(
                    window) + '_realization=' + str(realization_count) + '.pkl')
            ind1 = np.where(df_params.loc[:, 'a'] >= 0)[0];ind2 = np.where(df_params.loc[:, 'b'] >= 0)[0]
            ind = np.intersect1d(ind1, ind2)
            logh_data = np.array(df_params.loc[ind, 'logh'])
            loga_data = np.log10(np.array(df_params.loc[ind, 'a']));logb_data = np.log10(np.array(df_params.loc[ind, 'b']))
            print('simu')
            logh_maxLeft = compute_first_logh(plot,input_output, 'left', sigma_noise, error,m_val,s_analytic, loga_data,logb_data,logh_data, logh_data[0])
            logh_maxRight = compute_first_logh(plot,input_output, 'right', sigma_noise, error,m_val,s_analytic, loga_data,logb_data,logh_data, logh_data[-1])
            FODR= 10 * (logh_maxRight - logh_maxLeft)
            df_discriminated.loc[0, 'FODR'] = FODR
            df_discriminated.to_pickle(
                path_save + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                    realization_count) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')
            if only_FODR==0:
                print('logh from left')
                logh_selected_from_left = find_all_discriminable_inputs('left', logh_maxRight, logh_maxLeft,
                                              loga_data, logb_data, logh_data,*params)
                print('logh from right')
                logh_selected_from_right= find_all_discriminable_inputs('right', logh_maxRight, logh_maxLeft,
                                              loga_data, logb_data, logh_data,*params)
                N_d= (len(logh_selected_from_right) + len(logh_selected_from_left)) / 2
                df_discriminated.loc[0, 'no. points'] = N_d
                df_discriminated.at[0, 'logh from left'] = logh_selected_from_left
                df_discriminated.at[0, 'logh from left'] = logh_selected_from_left
                df_discriminated.to_pickle(
                    path_save + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                        realization_count) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')

        if typ_computation=='inf':
            inf=1
            print('inf')
            # # plt.figure()
            print('left')
            logh_maxLeft = compute_first_logh_analytic(plot,input_output, inf, 'left',m_val,mu,expr,G, sigma_noise, error, s_analytic,
                                              logh_list, logh_list[0])
            # # plt.figure()
            print('right')
            logh_maxRight = compute_first_logh_analytic(plot,input_output, inf, 'right',m_val,mu,expr,G, sigma_noise, error, s_analytic,
                                              logh_list, logh_list[-1])
            FODR_inf=10 * (logh_maxRight - logh_maxLeft)
            df_discriminated.loc[0, 'FODR_inf'] = FODR_inf

            df_discriminated.to_pickle(
                path_save + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                    realization_count) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')
            if only_FODR==0:
                logh_selected_from_left_inf =find_all_discriminable_inputs_analytic('left', sigma_noise, error, logh_maxRight, logh_maxLeft,
                                              loga_data, logb_data, logh_data)
                logh_selected_from_right_inf =find_all_discriminable_inputs_analytic('left', sigma_noise, error, logh_maxRight, logh_maxLeft,
                                              loga_data, logb_data, logh_data)
                print('logh from right')
                N_d_inf=(len(logh_selected_from_right_inf )+len(logh_selected_from_left_inf))/2
                df_discriminated.loc[0, 'no. points_inf'] = N_d_inf
                df_discriminated.at[0, 'logh from left_inf'] = logh_selected_from_left_inf
                df_discriminated.at[0, 'logh from right_inf'] = logh_selected_from_right_inf
                df_discriminated.to_pickle(
                    path_save + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                        realization_count) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')

        df_discriminated.to_pickle(
            path_save + '/FODR_epsilon=' + '{:.2e}'.format(epsilon) + '_window=' + str(window) + '_realization=' + str(
                realization_count) + '_error=' + str(error) + '_sigma_noise=' + str(sigma_noise) + '.pkl')
    return 1




from functools import partial
import numpy as np
import os
import multiprocessing as mp


plot = 1
only_FODR = 1
window_list = [1, 10, 100, 1000, 10000]
window_list = [1]  # ,10,100,1000,10000]
n_realization = 1
# sub_typ='subFix'
sub_typ = 'subFull'
# sub_typ='subRandom'
# input_output='inSub1_outSub2'
input_output = 'inSub1_outAll'
# input_output='inSub1_outSub3'
path_data = 'betaParams_Oct2021/'

N = 10000
error = 0.2
sigma_base = 1e-2
eps_list = np.logspace(-4, 0, 9)[:1]
# eps_list=np.logspace(-5,0,25)
l_list = 1 - eps_list

path = '6.3/'
#os.mkdir(path)
#os.mkdir(path +'/'+sub_typ)

if 0:

    typ_computation = 'simu'
    path_to_save = path + '/' + sub_typ + '/' + typ_computation + '/'
    #os.mkdir(path +'/'+sub_typ+'/'+typ_computation)
    for window in window_list:
        print('window=', window)
        sigma_noise = sigma_base
        funct = partial(FODR_utils.compute_FODR, typ_computation, only_FODR, sub_typ, n_realization, input_output, path_data,
                        path_to_save, eps_list, sigma_noise, window, error)
        pool = mp.Pool(processes=len(eps_list))
        results = pool.map(funct, np.arange(len(eps_list)))
        pool.close()
        pool.join()

#        with ProcessPoolExecutor() as executor:
#            outputs = executor.map(funct, np.arange(len(eps_list)))
if 1:
    #eps_list = np.logspace(-5, 0, 30)
    # l_list = 1 - eps_list
    window = 1
    typ_computation = 'theo'
    path_to_save = path + '/' + sub_typ + '/' + typ_computation + '/'
    #os.mkdir(path + '/' + sub_typ + '/' + typ_computation)
    sigma_noise = sigma_base
    funct = partial(compute_FODR, typ_computation, only_FODR, sub_typ, n_realization, input_output, path_data,
                    path_to_save, eps_list, sigma_noise, window, error)
    funct(0)
    # pool = mp.Pool(processes=len(eps_list))
    # results = pool.map(funct, np.arange(len(eps_list)))
    # pool.close()
    # pool.join()
   # with ProcessPoolExecutor() as executor:
   #     outputs = executor.map(funct, np.arange(len(eps_list)))

if 0:
    eps_list = np.logspace(-5, 0, 30)
    l_list = 1 - eps_list
    typ_computation = 'inf'
    path_to_save = path + '/' + sub_typ + '/' + typ_computation + '/'
    os.mkdir(path + '/' + sub_typ + '/' + typ_computation)
    sigma_noise = sigma_base
    funct = partial(FODR_utils.compute_FODR, typ_computation, only_FODR, sub_typ, n_realization, input_output, path_data,
                    path_to_save, eps_list, sigma_noise, window, error)
    pool = mp.Pool(processes=len(eps_list))
    results = pool.map(funct, np.arange(len(eps_list)))
    pool.close()
    pool.join()
