"""
R과 WinBug를 이용한 베이지안 통계학, p.104
이항모형의 비공액사전분포 예
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st



def func(a):
	mu = 1 ; tau2 = 4
	y = 5 ; n = 10
	posterior = (np.exp(a*y))/(((1+np.exp(a))**n)* \
				(np.exp(-1/(2*tau2)*(a-mu)**2)))
	return posterior

span = np.arange(-3,3, 0.1)
fn1 = []

for eta in span:

	fn1.append(func(eta))

sum1 = sum(fn1)
fn2 = fn1/sum1

mean = np.mean(fn2)

print(mean)


## P.105

from scipy import stats as st

theta = np.arange(0.3, 0.6, 0.001)
alpha = 1 ; beta = 1

###################
###### P.111 ######
###################

 
a = 5 ; b = 100
y = np.array([46, 58, 40, 47, 47, 53, 43, 48, \
 			50, 55, 49, 50, 52, 56, 49, 54, 51, 50, \
 			52, 50])
n = 20
theta = 50
a1 = (n+a)/2
b1 = ((n-1)*np.var(y)+b)/2

x = np.linspace(st.invgamma.ppf(0.01,a1,scale=b1),
				st.invgamma.ppf(0.99,a1,scale=b1), 100)

mean, var, skew, kurt = st.invgamma.stats(x,
										moments='mvsk')


fig, ax = plt.subplots(1,1)
# posterior
ax.plot(x, st.invgamma.pdf(x,a1,scale=b1), 'r-',
		label='invgamma.pdf')
# prior
ax.plot(x, st.invgamma.pdf(x,a/2,scale=b/2), 'green')
# likelihood
ax.plot(x, st.invgamma.pdf(x,(n-2)/2,scale=b1-b/2),
												'blue')
plt.xlim(0,60)

plt.show()


###############################################
# P.127 무정보적 공액사전분포를 이용한
#       결합사후분포로부터 시뮬레이션을
#  	    수행하여 표본 생성하기   	
###############################################


yy = np.random.normal(0,np.sqrt(4),20)
NN = 10000 ; mu0 = 0 ; tau0 = 1
alpha = 1 ; beta = 1 
lda = 1/(tau0*tau0)
mu = [] ; r = []
nn = len(yy)
u = (nn-1)*np.var(yy)
ybar = np.mean(yy)

a1 = (nn+alpha)/2
scale = .5*(nn*lda*(ybar-mu0)*(ybar-mu0))/ \
								(nn+lda)+u+beta
r = st.gamma.rvs(a1, scale=1/scale, size=NN)
# r = np.random.gamma(a1, 1/scale, NN)

for i in r:
	# gamma_r.append(st.gamma.rvs(a1,size= 10000,
	# 								scale=1/scale))
	# num = st.gamma.rvs(a1,scale=1/scale,size=1)
	mu.append(
		np.random.normal((nn*lda*mu0)/(nn+lda),
								1/(i*(nn+lda)))
												)
	# r.append(num)

bins_mu = np.linspace(-3,3,120)
bins_r = np.linspace(0,0.8,80)

plt.figure(1)
plt.subplot(211)
plt.hist(mu, bins_mu, color='green', label='mu')
plt.subplot(212)
plt.hist(r, bins_r, color='blue', label='r')
plt.show()

""" 
이건 참고하자! 그래프 그리는 방법이다!
# histogram with line on Gamma distribution

shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 1000)

# Display the histogram of the samples,
#   along with the probability density function:

import matplotlib.pyplot as plt
import scipy.special as sps
count, bins, ignored = plt.hist(s, 50, normed=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /
                    (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.show()
"""



##############################################
# P.131 무정보적 "반"공액사전분포를 이용한 
# 		결합사후분포로부터 시뮬레이션을 
#  	    수행하여 표본 생성하기
##############################################

yy = np.random.normal(0, np.sqrt(4), 20)
NN = 10000 ; mu0 = 0 ; eta0 = 2
alpha = 1 ; beta = 1 
mu = [0] ; r = [0]
nn = len(yy)
u = (nn-1)*np.var(yy)
ybar = np.mean(yy)

a1 = (nn+alpha)/2

for m in range(1,NN):
	
	num = st.gamma.rvs(a1,
		scale=1/(.5*(nn*(ybar-mu[m-1])*(ybar-mu[m-1]) \
			      +u+beta)),
		size=1)
	r.append(num)
	mu.append(
		np.random.normal((nn*r[m]*ybar*eta0*mu0)/ \
			(nn*r[m]+eta0),	1/(nn*r[m]+eta0))
												)
	

bins_mu = np.linspace(-0.7,0.7,140)
bins_r = np.linspace(0,0.8,80)

plt.figure(1)
plt.subplot(211)
plt.hist(mu, bins_mu, color='green', label='mu')
plt.subplot(212)
plt.hist(r, bins_r, color='blue', label='r')
plt.show()




##############################################
# P.133 예제 4-5) 미국 대통령 여론조사
##############################################


NN = 1000
yy = np.array([727, 583, 137])
alpha = np.array([1,1,1])
theta = st.dirichlet.rvs(alpha+yy, size=NN)
# count = 0
objt = []

for i in theta:
	if i[0]>i[1]:
		# count+=1    # Check out count to be 1000
		objt.append(i[0]-i[1])

print('theta1이  theta2 보다 작은 경우의 수는 %s 번 입니다.' \
		 % len(objt))

bins = np.linspace(0.00,0.20,20)


plt.figure(1)
plt.hist(theta[0]-theta[1], bins, color='gray', 
		label='theta1-theta2')
plt.show()

