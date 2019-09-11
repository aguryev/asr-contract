
import gzip
import numpy.random as rnd
import matplotlib.pyplot as plt
from numpy import log, exp

class ASR:
	#
	# A model of Accelerated Share Repurchase contract
	# by Olivier Guéant, Jiang Pu, Guillaume Royer
	# 'Accelerated Share Repurchase: pricing and execution strategy'
	# arXiv: https://arxiv.org/pdf/1312.5617.pdf
	# 
	def __init__(self, S0, sigma, T, N, V, Q, eta, fi, gamma):
		self.S0 = S0
		self.sigma = sigma
		self.T = T
		self.N = N
		self.V = V
		self.Q = Q
		self.eta = eta
		self.fi = fi
		self.gamma = gamma

	def initialize(self, NQ, INF):
		# perform all necessary methods to start working with a class object
		self.set_q_grid(NQ)
		self.set_infinity(INF)
		self.store_TETAs()
		self.set_tetas_filename()
		self.set_tetas_gzip_filename()
		
	def set_q_grid(self, nq):
		# set the computational grid for q - the number of shares
		self.nq = nq
		self.dq = self.Q // nq
		self.qs = [i*self.dq for i in range(nq + 1)]

	def set_infinity(self, inf):
		# set value for the infinity
		self.infinity = inf

	def store_TETAs(self):
		# define space for storring TETAs
		self.TETA_values= [[[0 for k in range(2 * i *(i - 1) + 1)] for j in range(self.nq + 1)] for i in range (1, self.T+1)]

	def set_tetas_filename(self):
		self.tfilename = 'teta_qgrid_{}_gamma_{:.1e}.txt'.format(self.nq, self.gamma)

	def set_tetas_gzip_filename(self):
		self.gzip_filename = 'teta_qgrid_{}_gamma_{:.1e}.gzip'.format(self.nq, self.gamma)

	def save_TETAs(self):	
		with open(self.tfilename, 'w') as tetas:
			tetas.write('{}\n'.format(self.nq))
			for i in range(self.T):
				for j in range(self.nq + 1):
					for k in range(2 * i *(i + 1) + 1):
						tetas.write('{} {} {} {}\n'.format(i + 1, self.qs[j], k, self.TETA_values[i][j][k]))

	def read_TETAs(self, tfilename):
		# read values of TETA form txt file
		with open(tfilename, 'r') as tetas:
			init = True
			for line in tetas.readlines():
				if init:
					self.set_q_grid(int(line.strip()))
					self.store_TETAs()
					#print ('nq = ', line)
					init = False
				else:
					#print ('line = ', line)
					a = line.split(' ')
					n = int(a[0])
					q_index = self.qs.index(int(a[1]))
					z = int(a[2])
					self.TETA_values[n-1][q_index][z] = float(a[3])

	def save_gzip_TETAs(self):	
		with gzip.open(self.gzip_filename, 'wb') as tetas:
			tetas.write('{}\n'.format(self.nq).encode())
			for i in range(self.T):
				for j in range(self.nq + 1):
					for k in range(2 * i *(i + 1) + 1):
						tetas.write('{} {} {} {}\n'.format(i + 1, self.qs[j], k, self.TETA_values[i][j][k]).encode())

	def read_gzip_TETAs(self, tfilename):
		# read values of TETA form txt file
		with gzip.open(tfilename, 'rb') as tetas:
			init = True
			for line in tetas.readlines():
				if init:
					self.set_q_grid(int(line.decode().strip()))
					self.store_TETAs()
					#print ('nq = ', line)
					init = False
				else:
					a = line.decode().split(' ')
					n = int(a[0])
					q_index = self.qs.index(int(a[1]))
					z = int(a[2])
					self.TETA_values[n-1][q_index][z] = float(a[3])

	def get_TETAs(self):
		for i in range(self.T):
			n = self.T - i
			print('n = ', n, end = ' ')
			for j in range (self.nq + 1):
				#print (self.qs[j], end = ' ')
				for z in range (2 * n *(n - 1) + 1):
					self.TETA_values[n-1][j][z] = self.TETA(n, self.qs[j], z)
			print('**** DONE!')

	def set_example_S(self, x):
		# set price trajectory as in example #x
		if x == 1:
			trajectory = [1, 0, 0,1, 0, -1, 1, 2, 0, 0, 1, 0, 0, 0, -1, 2, 0, 0, 1, -2, -1, 2, 0, 2, 2, 0, 1, 0, -1, 0, 0, -1, -1, 0, 0, -2, 0, 0, 0, 1, -1, 0, 2, 1, 2, 0, 1, -2, -1, -1, 1,-1, 1, 0, 1, -1, -1, 1, -1, 1, 0, 1, 1]
		elif x == 2:
			trajectory = [-2, -2, 1, 0, -1, 0, 1, -2, 1, -1, 0, 1, -1, -1, 2, 0, 2, -1, 0, 0, -2, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, -1, 1, -1, 2, 1, -1, 0, 0, -1, 1, -1, 0, -2, 0, 0, 1, 0, 1, 0, 0, -2, -1, 0, 0, 2, 0, -2, 0, 2]
		elif x == 3:
			trajectory = [2, 0, -2, 0, 0, 1, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, -2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, -1, 1, 2, -1, 1, -2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 2, 0, -1, -1, 2, 1, 2, 0]
		self.s = [0]*(self.T + 1)
		self.s[0] = self.S0
		factor = 1.25 # it is used in the example scenarios
		for i in range(self.T):
			eps = trajectory[i]*factor
			self.s[i+1] = self.s[i] + self.sigma * eps

	def set_S(self):
		# set a random price trajectory
		eps = rnd.choice([-2, -1, 0, 1, 2], self.T, p = [1/12, 1/6, 1/2, 1/6, 1/12])
		self.s = [0]*(self.T + 1)
		self.s[0] = self.S0
		for i in range(self.T):
			self.s[i+1] = self.s[i] + self.sigma * eps[i]

	def get_A(self):
		# calculate A - the arithmetic average of dialy VWAPs
		self.a = [self.S0]*(self.T + 1)
		sm = self.s[0]
		for i in range(1, self.T + 1):
			sm += self.s[i]
			self.a[i] = sm / (i + 1)

	def get_Z(self):
		# calculate the variable Z
		self.z = [0]*(self.T + 1)
		for i in range(self.T + 1):
			self.z[i] = (self.s[i] - self.a[i]) / self.sigma

	def get_PI(self):
		# calculate the indifference price of the ASR contract
		for i in range(self.nq + 1):
			vi = self.Q - self.qs[i]
			term = self.L(vi / self.V)*self.V + self.TETA_values[0][i][0]
			if i == 0:
				pi = term
			else:
				pi = min(pi, term)
		self.PI = pi / self.Q

	def get_q(self):
		# calculates remains for the optimal strategy
		self.q = [1]*(self.T + 1)
		self.v = [0]*(self.T + 1)
		q = self.Q
		for i in range(1, self.T+1):
			v = -1
			tmin = 0
			Z = int(i * (self.z[i] + i - 1))
			Z = max(Z, 0)
			Z = min(Z, 2*i*(i-1))
			for j in range (self.nq + 1):
				vi = q - self.qs[j]
				if vi >= 0 and vi <= self.V:
					if (v < 0) or (tmin > self.TETA_values[i-1][j][Z]):
						tmin = self.TETA_values[i-1][j][Z]
						v = vi
			q -= v
			self.q[i] = q / self.Q
			self.v[i] = v

	def save_results(self):
		# saves the results for a specific trajectory
		data_filename = 'data_qgrid_{}_gamma_{:.1e}.txt'.format(self.nq, self.gamma)
		with open(data_filename, 'w') as data:
			data.write('Time\tPrice\tBought\tRemains\tAverage\tZ\n')
			for i in range(self.T + 1):
				data.write('{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\n'.format(i, self.s[i], self.v[i], self.q[i], self.a[i], self.z[i]))

	def L(self, ro):
		# function of the execution costs
		return (self.eta * (abs(ro))**(1 + self.fi))

	def l(self, q):
		# penalty function
		if q == 0:
			return 0
		else:
			return self.infinity

	def TETA(self, n, q, z):
		# recursive equation for TETA
		if n == self.T:
			return self.l(q)
		elif n >= self.N[0] and n <= self.N[1]:
			return min(self.TETA_tilda(n, q, z), self.l(q))
		else:
			return self.TETA_tilda(n, q, z)

	def TETA_tilda(self, n, q, z):
		p = [1/12, 1/6, 1/2, 1/6, 1/12]
		for i in range(self.nq + 1):
			sm = 0
			for j in range(5):
				term_1 = self.sigma * ((j - 2) * (q - self.Q/(n + 1)) - self.Q / (n + 1) * (z/n - (n - 1)))
				term_2 = self.L((q - self.qs[i]) / self.V) * self.V
				#print(z + n*j)
				term_3 = self.TETA_values[n][i][z + n * j]
				term_4 = self.gamma * (term_1 + term_2 + term_3)
				sm += p[j] * exp(term_4)
				#if (n == 55) and (q == 20000000) and (z == 2970):
					#print ('For j = {}: TETA_{}({}, {}) = {}'.format(j, n+1, self.qs[i], z + n * j, term_3))
					#print ('For j = {}: {}'.format(j, p[j] * exp(term_4)))
			term = log(sm) / self.gamma
			if i == 0:
				inf = term
			else:
				inf = min(inf, term)
		#if (n == 55) and (q == 20000000) and (z == 2970):
			#print ('TETA_{}({}, {}) = {}'.format(n, q, z, inf))
		return inf

	def plot_trajectory(self):
		# plot the price trajectory: price, average price, Z

		# create two different scales
		fig, price = plt.subplots()
		zet = price.twinx()

		# plot prices
		price.plot([0, self.T], [self.S0, self.S0], 'k:', linewidth = 1)				# initial price S0
		a, = price.plot(range(self.T+1), self.a, 'g:', label = 'Average Price', linewidth = 2)		# average price
		s, = price.plot(range(self.T+1), self.s, 'r', label = 'Price', linewidth = 2)			# price
		price.set_xlabel('Time')
		price.set_ylabel('Price')
		price.set_ylim(35, 55)

		# plot Z
		z, = zet.plot(range(self.T+1), self.z, 'k--', label = 'Z', linewidth = 1)			# Z
		zet.set_ylabel('Z')
		zet.set_ylim(-15, 15)

		fig.tight_layout
		plt.legend(handles = [s, a, z], loc = 3)
		plt.show()

	def plot_otimal_strategy(self):
		# plot the optimal strategy: optimal strategy and Z

		# create two different scales
		fig, strg= plt.subplots()
		zet = strg.twinx()

		# plot the optimal strategy
		v, = strg.plot(range(self.T+1), self.q, 'r', label = 'Optimal Strategy', linewidth = 2)	# optimal strategy
		strg.set_xlabel('Time')
		strg.set_ylabel('Remain to Buy')
		strg.set_ylim(0, 1)

		# plot Z
		zet.plot([0, self.T], [0, 0], 'k', linewidth = 1)	# Z = 0
		z, = zet.plot(range(self.T+1), self.z, 'k--', label = 'Z', linewidth = 1)		# Z
		zet.set_ylabel('Z')
		zet.set_ylim(-15, 15)
		
		fig.tight_layout
		plt.legend(handles = [v, z], loc = 3)
		plt.show()
