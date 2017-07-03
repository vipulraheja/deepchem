import numpy as np
import unittest
import deepchem as dc

class SeluTest(test.TestCase):
	def _npSelu(self, np_features):
		scale = 1.0507009873554804934193349852946
		scale_alpha = 1.7580993408473768599402175208123
		return np.where(np_features < 0, scale_alpha * (np.exp(np_features) - 1), scale * np_features)


	def testNpSelu(self):
		self.assertAllClose(
			np.array([[-1.0433095, 0.73549069, -0.6917582, 0.3152103 , -0.16730527],
				[0.1050701 , -0.45566732, 0.5253505, -0.88505305, 0.9456309]]),
			self._npSelu(np.array([[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7, 0.9]
		])))
