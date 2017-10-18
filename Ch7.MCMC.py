import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt


def betabinom(nsim, n, alpha, beta):
	pmix = st.beta.rvs(alpha, beta, size=nsim)
	return st.nbinom.rvs(n, pmix, size=nsim)

nsim=500	# gibbs sampling 으로 표본 500개 생성
k=10	# gibbs 수열 10번 반복 후 한 개의 (X,Y)표본 선택
n=16 ; alpha=2; beta=4	# beta & nbinom 조건부 분포 가정

x=[]
for i in range(nsim):

	# 초기값 설정
	y_temp = [ st.beta.rvs(1,1,size=1) ]
	
	# scipy 의 binomial 함수식과 R의 계산식의 input
	# 방식이 다르고, numpy 는 같으므로 numpy로 계산
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

direct = betabinom(m,n,alpha,beta)
Gibbs = 