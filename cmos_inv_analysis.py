import numpy as np
import scipy.optimize as scipy

V_dd = 2
GND = 0

# Dati fisici / geometrici presi da:
# https://dolly.ingmo.unimore.it/2020/mod/resource/view.php?id=1715

# IMPORTANTE:
# Suppongo che la corrente di saturazione è raggiunta quando V_min = min{ V_ds_n, V_dsat_n, V_gs_n - V_t_n} = V_gs_n - V_t_n.
# Questa caratteristica è tipica di transistor lunghi, poiché V_dsat_n = E_crit * L (quindi con L lunga).
# Di conseguenza, sempre nel caso dell'nMOS, la corrente di saturazione avrà equazione:
# I_dsat_n = (1 / 2) * k_n * a_n * (V_gs_n - V_t_n)^2 * (1 + lambda_n * V_ds_n)
# Faccio lo stesso ragionamento in modo analogo per il pMOS.
#
# Test fatti su Geogebra:
# https://www.geogebra.org/calculator/w9zjdbgj

# pMOS equation:
# I_d = k_p * a_p * V_min * (V_sg_p - abs(V_t_p) - V_min / 2) * (1 + abs(lambda_p) * V_sd_p) if V_sg_p > abs(V_t_p) else 0

# nMOS equation:
# I_d = k_n * a_n * V_min * (V_gs_n - V_t_n - V_min / 2) * (1 + lambda_n * V_ds_n) if V_gs_n > V_t_n else 0


# nMOS
k_n = 200 # µA / V^2
a_n = 1
lambda_n = 0.1
V_dsat_n = 0.4
V_t_n = 0.35


def nMOS_is_off(V_gs_n):
	return V_gs_n < V_t_n


def nMOS_is_lin(V_gs_n, V_ds_n):
	t = V_gs_n - abs(V_t_n)
	return not nMOS_is_off(V_gs_n) and V_ds_n < t


def nMOS_is_sat(V_gs_n, V_ds_n):
	t = V_gs_n - abs(V_t_n)
	return not nMOS_is_off(V_gs_n) and V_ds_n > t

# pMOS
k_p = 200
a_p = 1
lambda_p = 0.12
V_dsat_p = 0.4
V_t_p = 0.35


def pMOS_is_off(V_sg):
	return abs(V_sg) < abs(V_t_p)


def pMOS_is_lin(V_sg_p, V_sd_p):
	t = V_sg_p - abs(V_t_p)
	return not pMOS_is_off(V_sg_p) and V_sd_p < t


def pMOS_is_sat(V_sg_p, V_sd_p):
	t = V_sg_p - abs(V_t_p)
	return not pMOS_is_off(V_sg_p) and V_sd_p > t

# V_m
r = (k_p * V_dsat_p) / (k_n * V_dsat_n)

V_m = (r * V_dd) / (1 + r)
print("V_m: ~%.2fV" % V_m)


def cmos_inverter_I_d(V_in):
	V_gs_n = V_in - GND
	V_sg_p = V_dd - V_in
	
	# Coefficienti utili presenti nell'equazione di saturazione.
	C_n = (1 / 2) * k_n * a_n * (V_gs_n - V_t_n)**2
	C_p = (1 / 2) * k_p * a_p * (V_sg_p - abs(V_t_p))**2
	
	def pMOS_off_and_nMOS_on():
		if pMOS_is_off(V_sg_p) and not nMOS_is_off(V_gs_n):
			I_d = 0
			V_ds_n = 0
			V_sd_p = V_dd
		
			print("V_in=%.3fV => pMOS on & nMOS off : I_d=%.3fµA V_ds_n=%0.3fV V_sd_p=%0.3fV" % (V_in, I_d, V_ds_n, V_sd_p))
			return [I_d, V_ds_n, V_sd_p]
		
		return None

	def pMOS_on_and_nMOS_off():
		if not pMOS_is_off(V_sg_p) and nMOS_is_off(V_gs_n):
			I_d = 0
			V_ds_n = V_dd
			V_sd_p = 0
		
			print("V_in=%.3fV => pMOS on & nMOS off : I_d=%.3fµA V_ds_n=%0.3fV V_sd_p=%0.3fV" % (V_in, I_d, V_ds_n, V_sd_p))
			return [I_d, V_ds_n, V_sd_p]
		
		return None
	
	def pMOS_pinchoff_and_nMOS_pinchoff():
		# Nel caso in cui entrambi siano in pinchoff ottengo un sistema lineare che posso facilmente risolvere.

		m_n = C_n * lambda_n
		m_p = -C_p * abs(lambda_p)
		
		q_n = C_n
		q_p = C_p + C_p * abs(lambda_p) * V_dd
		
		A = np.array([[m_n, -1], [m_p, -1]])
		B = np.array([-q_n, -q_p])
		
		V_ds_n, I_d = np.linalg.solve(A, B)
		V_sd_p = V_dd - V_ds_n

		if nMOS_is_sat(V_gs_n, V_ds_n) and pMOS_is_sat(V_sg_p, V_sd_p):
			print("V_in=%.3fV => pMOS pinchoff & nMOS pinchoff : I_d=%.3fµA V_ds_n=%.3fV V_sd_p=%.3fV" % (V_in, I_d, V_ds_n, V_sd_p))
			return [I_d, V_ds_n, V_sd_p]
		
		return None
		
	def nMOS_lin_pMOS_pinchoff():
		# Se un transistor si trova in zona lineare ho a che fare con un'equazione cubica poiché considero anche il termine (1 + lambda_n * V_ds_n).
		# Il grafico di quest'equazione è una simil-parabola per x > 0 il cui vertice è circa a V_gs_n - V_t_n.

		# Se voglio trovare l'intersezione con la retta del pMOS in pinchoff,
		# mi serve un punto X_0 vicino alla soluzione reale da cui steppare per trovare la soluzione effettiva.

		# Il punto X_0 lo trovo nel seguente modo:
		# - prendo l'arco di parabola crescente da x = 0 a x = v_x ~= V_gs_n - V_t_n.
		# - divido quest'intervallo per due, in modo da prenderne il punto medio.
		# - y0 = nMOS_lin(x0)
		# - trovo la retta tangente al punto P=(y0, x0) rispetto alla funzione nMOS_lin
		# - trovo l'intersezione tra questa retta e quella del pMOS in pinchoff

		#print("pMOS lin : y = %.3f * x * (%.3f - x / 2) * (1 + %.3f * x)" % (k_n * a_n, (V_gs_n - V_t_n), lambda_n))

		def nMOS_lin_f(x):
			return k_n * a_n * x * ((V_gs_n - V_t_n) - x / 2) * (1 + lambda_n * x)

		def pMOS_pinchoff_f(x):
			return C_p * (1 + abs(lambda_p) * (V_dd - x))

		def sys(x):
			return (
				nMOS_lin_f(x[0]) - x[1],
				pMOS_pinchoff_f(x[0]) - x[1]
			)

		# Calcolo della tangente:
		x_0 = (V_gs_n - V_t_n) / 2
		y_0 = nMOS_lin_f(x_0)

		#print("X tan : X_t = (%.6f, %.6f)" % (x_0, y_0))

		eps = 0.00001  # Valore piccolo per poter calcolare lo slope della retta.
		m_tan = (nMOS_lin_f(x_0 + eps) - nMOS_lin_f(x_0)) / eps
		q_tan = -m_tan * x_0 + y_0

		#print("nMOS tan : y = %.3f * x + %.3f" % (m_tan, q_tan))

		# Calcolo della retta di pinchoff del pMOS:
		pMOS_m = -C_p * abs(lambda_p)
		pMOS_q = C_p + C_p * abs(lambda_p) * V_dd

		#print("pMOS pinchoff : y = %.3f * x + %.3f" % (pMOS_m, pMOS_q))

		# Risoluzione dell'intersezione in modo approssimato:
		A = np.array([[m_tan, -1], [pMOS_m, -1]])
		B = np.array([-q_tan, -pMOS_q])

		X_0 = np.linalg.solve(A, B)
		#print("nMOS tan x pMOS pinchoff : X_0 = (%.3f, %.3f)" % tuple(X_0))

		# Risoluzione dell'intersezione in modo preciso:
		[V_ds_n, I_d], info, ier, msg = scipy.fsolve(sys, X_0, full_output=True)
		V_sd_p = V_dd - V_ds_n

		#print("nMOS lin x pMOS pinchoff : X_p = (%.6f, %.6f)" % (V_ds_n, I_d))

		if ier != 1 or not nMOS_is_lin(V_gs_n, V_ds_n):
			return None  # Nessun'intersezione.

		print("V_in=%.3fV => pMOS pinchoff & nMOS lin : I_d=%.3fµA V_ds_n=%.3fV V_sd_p=%.3fV" % (V_in, I_d, V_ds_n, V_sd_p))

		return I_d, V_ds_n, V_sd_p
		
	def nMOS_pinchoff_and_pMOS_lin():
		# Procedimento uguale a quello descritto sopra.

		#print("pMOS lin : y = %.3f * (%.3f - x) * (%.3f - (%.3f - x) / 2) * (1 + %.3f * (%.3f - x))" % (k_p * a_p, V_dd, V_sg_p - abs(V_t_p), V_dd, abs(lambda_p), V_dd))

		def pMOS_lin_f(x):
			return k_p * a_p * (V_dd - x) * ((V_sg_p - abs(V_t_p)) - (V_dd - x) / 2) * (1 + abs(lambda_p) * (V_dd - x))

		def nMOS_pinchoff_f(x):
			return C_n * (1 + lambda_n * x)

		def sys(x):
			return (
				pMOS_lin_f(x[0]) - x[1],
				nMOS_pinchoff_f(x[0]) - x[1]
			)

		# Calcolo della tangente:
		# In questo caso l'intervallo va da V_dd a V_dd - (V_sg_p - abs(V_t_p)) poiché il riferimento è la V_ds_n.
		I_len = V_dd - (V_sg_p - abs(V_t_p))
		x_0 = V_dd - I_len / 2
		y_0 = pMOS_lin_f(x_0)

		#print("X tan : X_t = (%.6f, %.6f)" % (x_0, y_0))

		eps = 0.00001  # Valore piccolo per poter calcolare lo slope della retta.
		m_tan = (pMOS_lin_f(x_0 + eps) - pMOS_lin_f(x_0)) / eps
		q_tan = -m_tan * x_0 + y_0

		#print("pMOS tan : y = %.3f * x + %.3f" % (m_tan, q_tan))

		# Calcolo della retta di pinchoff dell'nMOS:
		nMOS_m = C_n * abs(lambda_n)
		nMOS_q = C_n

		#print("nMOS pinchoff : y = %.3f * x + %.3f" % (nMOS_m, nMOS_q))

		# Risoluzione dell'intersezione in modo approssimato:
		A = np.array([[m_tan, -1], [nMOS_m, -1]])
		B = np.array([-q_tan, -nMOS_q])

		X_0 = np.linalg.solve(A, B)

		#print("pMOS tan x nMOS pinchoff : X_0 = (%.3f, %.3f)" % tuple(X_0))

		# Risoluzione dell'intersezione in modo preciso:
		[V_ds_n, I_d], info, ier, msg = scipy.fsolve(sys, X_0, full_output=True)
		V_sd_p = V_dd - V_ds_n

		#print("nMOS lin x pMOS pinchoff : X_p = (%.3f, %.3f)" % (V_ds_n, I_d))

		if ier != 1 or not pMOS_is_lin(V_sg_p, V_sd_p):
			return None  # Nessun'intersezione.

		print("V_in=%.3fV => pMOS lin & nMOS pinchoff : I_d=%.3fµA V_ds_n=%.3fV V_sd_p=%.3fV" % (V_in, I_d, V_ds_n, V_sd_p))

		return I_d, V_ds_n, V_sd_p

	# A priori non riesco a sapere lo stato dei due transistor, non conosco la V_ds_n!
	# Quindi provo tutti i casi eccetto quello in cui entrambi sono in zona lineare o entrambi spenti (non possibile).
	
	sol = None
	
	if sol is None:
		sol = pMOS_on_and_nMOS_off()  # pMOS on e nMOS off
		
	if sol is None:
		sol = pMOS_off_and_nMOS_on()  # nMOS off e pMOS off

	if sol is None:
		sol = nMOS_lin_pMOS_pinchoff()  # nMOS lin e pMOS sat
	
	if sol is None:
		sol = nMOS_pinchoff_and_pMOS_lin()  # nMOS sat e pMOS lin

	if sol is None:
		sol = pMOS_pinchoff_and_nMOS_pinchoff()  # pMOS sat e nMOS sat
	
	if sol is None:
		print("ERRORE: V_in=%.3fV => UNKNOWN" % V_in)
		sol = [0, 0, 0]
	
	return sol


def main():
	import matplotlib.pyplot as plt

	V_in_arr = []
	I_d_arr = []

	divisions = 100 # Numero di step in cui suddividere l'intervallo [0, V_dd].

	for i in range(0, divisions):
		V_in = V_dd / divisions * i
		[I_d, V_ds_n, V_sd_p] = cmos_inverter_I_d(V_in)

		V_in_arr.append(V_in)
		I_d_arr.append(I_d)

	plt.plot(V_in_arr, I_d_arr)
	plt.title('I_d(V_in)')
	plt.xlabel('V_in [V]')
	plt.ylabel('I_d [µA]')
	plt.show()

if __name__ == "__main__":
	main()
