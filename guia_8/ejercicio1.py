import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#Valores de calificacion
x_cal = np.arange(0, 11, 1)
x_serv = np.arange(0, 11, 1)
#Porcentajes entre 0% y 25% de propina
x_tip = np.arange(0, 26, 1)

#Generando funciones de membresia borrosa
#Calidad de la comida
cal_baja = fuzz.trimf(x_cal, [0, 0, 5])
cal_media = fuzz.trimf(x_cal, [0, 5, 10])
cal_alta = fuzz.trimf(x_cal, [5, 10, 10])
#Calidad del servicio
serv_baja = fuzz.trimf(x_serv, [0, 0, 5])
serv_media = fuzz.trimf(x_serv, [0, 5, 10])
serv_alta = fuzz.trimf(x_serv, [5, 10, 10])
#Propinas
tip_baja = fuzz.trimf(x_tip, [0, 0, 13])
tip_media = fuzz.trimf(x_tip, [0, 13, 25])
tip_alta = fuzz.trimf(x_tip, [13, 25, 25])

#Visualizacion de valores y funciones de membresia
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_cal, cal_baja, 'b', linewidth = 1.5, label = 'Baja')
ax0.plot(x_cal, cal_media, 'g', linewidth = 1.5, label = 'Media')
ax0.plot(x_cal, cal_alta, 'r', linewidth = 1.5, label = 'Alta')
ax0.set_title('Calidad de la comida')
ax0.legend()

ax1.plot(x_serv, serv_baja, 'b', linewidth = 1.5, label = 'Me escupio la comida')
ax1.plot(x_serv, serv_media, 'g', linewidth = 1.5, label = 'Meh, hay mejores')
ax1.plot(x_serv, serv_alta, 'r', linewidth = 1.5, label = 'Un/una tipazo/a')
ax1.set_title('Calidad del servicio')
ax1.legend()

ax2.plot(x_tip, tip_baja, 'b', linewidth = 1.5, label = 'Baja')
ax2.plot(x_tip, tip_media, 'g', linewidth = 1.5, label = 'Media')
ax2.plot(x_tip, tip_alta, 'r', linewidth = 1.5, label = 'Alta')
ax2.set_title('Porcentaje de propinas')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
plt.tight_layout()
plt.show()

#Evaluamos los valores en las funciones de membresia de cada conjunto
lvl_cal_baja = fuzz.interp_membership(x_cal, cal_baja, 6.5)
lvl_cal_media = fuzz.interp_membership(x_cal, cal_media, 6.5)
lvl_cal_alta = fuzz.interp_membership(x_cal, cal_alta, 6.5)

lvl_serv_baja = fuzz.interp_membership(x_serv, serv_baja, 9.8)
lvl_serv_media = fuzz.interp_membership(x_serv, serv_media, 9.8)
lvl_serv_alta = fuzz.interp_membership(x_serv, serv_alta, 9.8)

#Regla 1: mala comida o mal servicio
activacion_regla1 = np.fmax(lvl_cal_baja, lvl_serv_baja)

tip_activacion_baja = np.fmin(activacion_regla1, tip_baja)

#Regla 2: servicio medio
tip_activacion_media = np.fmin(lvl_serv_media, tip_media)

#Regla 3: servicio alto o buena comida -> alta propina
activacion_regla3 = np.fmax(lvl_cal_alta, lvl_serv_alta)
tip_activacion_alta = np.fmin(activacion_regla3, tip_alta)
tip0 = np.zeros_like(x_tip)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_tip, tip0, tip_activacion_baja, facecolor='b', alpha=0.7)
ax0.plot(x_tip, tip_baja, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_tip, tip0, tip_activacion_media, facecolor='g', alpha=0.7)
ax0.plot(x_tip, tip_media, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tip, tip0, tip_activacion_alta, facecolor='r', alpha=0.7)
ax0.plot(x_tip, tip_alta, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Actividad de membresia de la salida')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

agregado = np.fmax(tip_activacion_baja, np.fmax(tip_activacion_media, tip_activacion_alta))

tip = fuzz.defuzz(x_tip, agregado, 'centroid')
tip_activacion = fuzz.interp_membership(x_tip, agregado, tip)

fig, ax0 = plt.subplots(figsize=(8, 3))

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_tip, tip_baja, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_tip, tip_media, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_tip, tip_alta, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tip, tip0, agregado, facecolor='Orange', alpha=0.7)
ax0.plot([tip, tip], [0, tip_activacion], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('membresia agragada y resultado (linea)')

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()

print("El porcentaje de propina es ", tip, "%")