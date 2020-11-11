'''
Paper on N-th harmonic number approximation
http://mi.mathnet.ru/eng/irj251
'''

import numpy as np

'''
Calculate analogue of Euler–Mascheroni constant for chosen 's'
'''
def get_gamma_s(s):
    # approximation polynome taken from linked paper 
    pm_s = 1.01956 + 0.223632*s + 3.45985 * 1e-2 * s - 9.32331*1e-4*(s**2) - 1.40047*1e-5*(s**3) +7.63*1e-6*(s**4)
    return 2*np.arctan(pm_s)/np.pi

#N=1000000
#K=128
N=100
K=8
alpha = 0.5
gamma_s = get_gamma_s(0.5)

aprox_total = (np.power(N,1 - alpha) - 1)/(1-alpha) + gamma_s # approximation of N-th harmonic number (partial sum of generalized harmonic series)
segment_len = (aprox_total - 1) / K #subtract 1 because ranks shifted by 2 to avoid 1/0 and 1/1 cases
print(f'Approximated total = {aprox_total}, aproximated segment length = {segment_len}')

prob_sum = 0
j=0
print(f'[{j}]', end=' ')
for i in range(N):
    prob_sum += np.power(i+2, -alpha)
    print(i, end=' ')
    if prob_sum >= segment_len:
        j+=1
        print(f'\n[{j}]', end=' ')
        print(f'Segment boundary {i}')
        prob_sum = 0.