import numpy as np

class HMM:

	"""
	HMM model class
	Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites"""
	def __init__(self, A, states, emissions, pi, B):
		self.A = A
		self.B = B
		self.states = states
		self.emissions = emissions
		self.pi = pi
		self.N = len(states)
		self.M = len(emissions)
		self.make_states_dict()

	def make_states_dict(self):

		"""Make dictionary mapping between states and indexes"""
		self.states_dict = dict(zip(self.states, list(range(self.N))))
		self.emissions_dict = dict(zip(self.emissions, list(range(self.M))))

	def viterbi_algorithm(self, seq):
		"""
		Function implementing the Viterbi algorithm
		Args:
		seq: Observation sequence (list of observations. must be in the emmissions dict)
		Returns:
		nu: Porbability of the hidden state at time t given an obeservation sequence
		hidden_states_sequence: Most likely state sequence """
		seq_length = len(seq)
		no_of_states=self.N
		ver = np.zeros((seq_length, no_of_states))
		temp = np.zeros((seq_length, no_of_states), dtype=int)
		for s in range(self.N):
			ver[0, s] = self.pi[s] * self.B[s, self.emissions_dict[seq[0]]]
			temp[0, s] = 0
		for i in range(1,seq_length):
			for j in range(0,no_of_states):
				max_ver=-1;max_temp=-1;
				for k in range(no_of_states):
					val=ver[i - 1,k] * self.A[k, j]*self.B[j, self.emissions_dict[seq[i]]]
					if val>max_ver:
						max_ver=val
						max_temp=k
				ver[i,j]=max_ver
				temp[i,j]=max_temp
		ver_max=-1
		temp_max=-1
		for i in range(no_of_states):
			check_val=ver[seq_length-1,i]
			if check_val>ver_max:
				ver_max=check_val
				temp_max=i
		new_states=[temp_max]
		for i in range(seq_length-1,0,-1):
			new_states.append(temp[i,new_states[-1]])
		new_states.reverse()
		new_dict=dict()
		for k,v in self.states_dict.items():
			new_dict[v]=k
		self.states_dict = new_dict
		return_list=[]
		for i in new_states:
			return_list.append(self.states_dict[i])
		return return_list
	pass