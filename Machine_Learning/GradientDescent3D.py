import os
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('Graphs'):
    os.makedirs('Graphs')

# Augmenter la résolution des échantillons pour les graphiques 3D et contours
x = np.arange(-0.5, 1.5, 0.005)  # Intervalle réduit pour plus de points
y = np.arange(-0.5, 1.5, 0.005)  # Intervalle réduit pour plus de points

X, Y = np.meshgrid(x, y)
Z = np.sin(5*X)*np.cos(5*Y)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, 
                       cmap='YlOrRd', 
                       linewidth=0, 
                       antialiased='True', 
                       rstride=3, 
                       cstride=3)

ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
ax.set_zlim([-1.5, 1.5])
plt.title("Surface Plot", size=14)
plt.savefig('Graphs/3D_surface_plot1.png', dpi=300, bbox_inches='tight')
plt.show()

# Augmenter la résolution pour le graphique de contour
x = np.arange(-0.5, 1.6, 0.02)  # Intervalle réduit pour plus de points
y = np.arange(-0.5, 1.6, 0.02)  # Intervalle réduit pour plus de points

X, Y = np.meshgrid(x, y)
Z = np.sin(5*X) * np.cos(5*Y)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
ax.contourf(X, Y, Z, 200, cmap='YlOrRd', alpha=0.3)  # Augmenter le nombre de niveaux de contour

ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
plt.title("Contour Plot", size=14)
plt.savefig('Graphs/Contour_plot1.png', dpi=300, bbox_inches='tight')
plt.show()

def gradient_f(v):
    x = v[0]
    y = v[1]
    gx = 5*np.cos(5*x) * np.cos(5*y)
    gy = -5*np.sin(5*x) * np.sin(5*y)
    return np.array([gx, gy])



v = np.array([2,5])
gradient_f(v)



def GD_f(lamda, epsilon):

    x = np.random.uniform(low=[-0.5, -0.5], high=[1.5, 1.5], size=2)
    points_GD = [x]

    while np.linalg.norm(gradient_f(x)) > epsilon:

        x = x - lamda * gradient_f(x)
        points_GD.append(x)

    return points_GD




points_GD = GD_f(0.01, 0.01)
points_GD




x = np.arange(-0.5, 1.6, 0.1)
y = np.arange(-0.5, 1.6, 0.1)

X, Y = np.meshgrid(x, y)
Z = np.sin(5*X) * np.cos(5*Y)


fig = plt.figure(figsize=(7, 7))

ax = fig.add_subplot(111)
ax.contourf(X, Y, Z, 100, cmap='YlOrRd', alpha=0.3)

#-----
gd_x = [v[0] for v in points_GD]
gd_y = [v[1] for v in points_GD]
ax.scatter(gd_x, gd_y, marker='.', c='black', label='GD trace')
#-----

ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-0.5, 1.5])
plt.title("Contour Plot", size=14)
plt.legend()
plt.savefig('Graphs/Contour_plot2.png', dpi=300, bbox_inches='tight')
plt.show



