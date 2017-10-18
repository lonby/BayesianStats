# -*- coding: utf-8 -*- 

import numpy as np
import scipy.stats as st
import scipy.special as sp
import matplotlib.pyplot as plt


# 7.2. MCMC

##################################################
# P.199 예제7-2) 베타-이항분포에서 깁스 샘플링 방법을 통해
# 추출한 샘플과 실제 베타-이항분포로부터 랜돔으로 추출한 샘플을
# 비교해 보자.
##################################################

def betabinom(nsim, n, alpha, beta):
	pmix = st.beta.rvs(alpha, beta, size=nsim)
	# scipy 의 binomial 함수식과 R의 계산식의 input
	# 방식이 다르고, numpy 는 같으므로 numpy로 계산
	return np.random.binomial(n, pmix, size=nsim)

nsim=500	# gibbs sampling 으로 표본 500개 생성
k=10	# gibbs 수열 10번 반복 후 한 개의 (X,Y)표본 선택
n=16 ; alpha=2; beta=4	# beta & nbinom 조건부 분포 가정

x=[]
for i in range(nsim):

	# 초기값 설정
	y_temp = [ st.beta.rvs(1,1,size=1) ]
	x_temp = [ np.random.binomial(n,y_temp[0][0],1) ]

	for j in range(k):
		y_temp.append(
					    st.beta.rvs(x_temp[j]+alpha,
									n-x_temp[j]+beta,
									size=1)
					  									)

		x_temp.append( np.random.binomial(n,
								y_temp[j+1][0],1) )

	x.append(x_temp[k][0])

direct = betabinom(nsim,n,alpha,beta)
gibbs = x


histogram = plt.figure()

bins = np.linspace(0,16, 17)
plt.hist(gibbs,bins, alpha=0.5,label='gibbs')
plt.hist(direct,bins,alpha=0.5,label='direct')
plt.legend()

plt.show()



##################################################
# P.202 예제7-2 계속) 베타-이항분포에서 깁스 샘플링 방법을 통해
# f(x|y)의 평균으로부터 주변확률분포(f(x))를 추정해보고,
# 실제 분포인 베타-이항분포를 비교해보자.
##################################################



def factorial(n):
	
	if n==0 :
		return 1
	
	else:
		f = n
		for i in range(1,n):
			f *= i
		return f

def nCr(n,r):
	return factorial(n) // factorial(n-r) // factorial(r)

def betabinom(n,x,alpha,beta):
	f = nCr(n,x)*(sp.gamma(alpha+beta)*sp.gamma(x+alpha)* \
	sp.gamma(n-x+beta))/(sp.gamma(alpha)*sp.gamma(beta)* \
	sp.gamma(alpha+beta+n))
	return f


nsim=500	# gibbs sampling 으로 표본 500개 생성
k=10	# gibbs 수열 10번 반복 후 한 개의 (X,Y)표본 선택
n=16 ; alpha=2; beta=4	# beta & nbinom 조건부 분포 가정

x=[]
y=[]
for i in range(nsim):

	# 초기값 설정
	y_temp = [ st.beta.rvs(1,1,size=1) ]
	x_temp = [ np.random.binomial(n,y_temp[0][0],1) ]

	for j in range(k):
		y_temp.append(
					    st.beta.rvs(x_temp[j]+alpha,
									n-x_temp[j]+beta,
									size=1)
					  									)

		x_temp.append( np.random.binomial(n,
								y_temp[j+1][0],1) )

	x.append(x_temp[k][0])
	y.append(y_temp[k][0])


gibbs=[] ; exact=[]
for i in range(17):
	
	gibbs.append(np.mean(st.binom.pmf(i,n,y)))
	exact.append(betabinom(n,i,alpha,beta))


num_bins = 17
bins = np.linspace(1,17, 17)
fig,ax = plt.subplots()
ax.plot(bins,gibbs, alpha=0.5,label='gibbs', color='r')
ax.plot(bins,exact,alpha=0.5,label='exact', color='b')
plt.legend()
plt.show()


# gibbs 추정값의 직접 평균
xmean1 = np.mean(x)	# 5.748~~
# 조건부평균을 이용한 평균
xmean2 = n*np.mean(y)	# 5.6456~~
dmean = np.mean(direct)	# 5.5~~

