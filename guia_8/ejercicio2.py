import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import seaborn as sns

comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')
servicio = ctrl.Antecedent(np.arange(0, 11, 1), 'servicio')
propina = ctrl.Consequent(np.arange(0, 26, 1), 'propina')

#==================================================
#Comida
#==================================================
comida.automf(3, names=['pobre', 'promedio', 'buena'])

#==================================================
#Servicio
#==================================================
servicio.automf(3, names=['pobre', 'promedio', 'buena'])

#==================================================
#Propina
#==================================================
propina.automf(3, names=['baja', 'media', 'alta'])

#===================================================
#Reglas
#===================================================
#Regla 1: Si servicio = bueno o comida = buena entonces propina = alta
regla1 = ctrl.Rule(servicio['buena'] | comida['buena'], propina['alta'])

#Regla 2: Si servicio = promedio entonces propina = media
regla2 = ctrl.Rule(servicio['promedio'], propina['media'])

#Regla 3: Si servicio = pobre o comida = pobre entonces propina = baja
regla3 = ctrl.Rule(servicio['pobre'] | comida['pobre'], propina['baja'])

propina_ctrl = ctrl.ControlSystem([regla1, regla2, regla3])

estimacion_propina = ctrl.ControlSystemSimulation(propina_ctrl)

estimacion_propina.input['comida'] = 6.5
estimacion_propina.input['servicio'] = 9.8

estimacion_propina.compute()

print(f"Propina: {estimacion_propina.output['propina']:0.4}%")

comida.view(sim=estimacion_propina)
servicio.view(sim=estimacion_propina)
propina.view(sim=estimacion_propina)

P = np.zeros((100, 100))
idx_s = 0
idx_q = 0
for s in np.arange(0, 10, 0.1):
    for q in np.arange(0, 10, 0.1):
        
        estimacion_propina.input['servicio'] = s
        estimacion_propina.input['comida'] = q
        estimacion_propina.compute()
        
        P[idx_s, idx_q] = estimacion_propina.output['propina']
        
        idx_q += 1
        
    idx_q = 0
    idx_s += 1
    
fig, ax = plt.subplots(1, 1, figsize=(14, 12))
sns.heatmap(P, cmap='jet', square=True, ax=ax)
plt.xlabel('Comida (expresado en el rango [0, 100])', fontsize=16)
plt.ylabel('Servicio (expresado en el rango [0, 100])', fontsize=16)
plt.show()