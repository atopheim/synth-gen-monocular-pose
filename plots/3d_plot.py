from mpl_toolkits import mplot3d
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cls_name',dest='cls_name',default='intake')
parser.add_argument('--dataset',dest='dataset',default='HOMEMADE')
args = parser.parse_args()

# Redundancy
try:
	farthest = np.loadtxt('data/{}/{}/farthest.txt'.format(args.dataset,args.cls_name))
	corners = np.loadtxt('data/{}/{}/corners.txt'.format(args.dataset,args.cls_name))
	dense_pts = np.loadtxt('data/{}/{}/dense_pts.txt'.format(args.dataset,args.cls_name))
except:
	farthest = np.loadtxt('../data/{}/{}/farthest.txt'.format(args.dataset,args.cls_name))
	corners = np.loadtxt('../data/{}/{}/corners.txt'.format(args.dataset,args.cls_name))
	dense_pts = np.loadtxt('../data/{}/{}/dense_pts.txt'.format(args.dataset,args.cls_name))
	
fig = plt.figure()
ax = plt.axes(projection='3d')

x = []
y = []
z = []

a = []
b = []
c = []

d = []
e = []
f = []

for idx,xyz in enumerate(dense_pts):
	if idx%75 == 0:
		x.append(xyz[0])
		y.append(xyz[1])
		z.append(xyz[2])

for idx,xyz in enumerate(corners):
	if True:
		a.append(xyz[0])
		b.append(xyz[1])
		c.append(xyz[2])

for idx,xyz in enumerate(farthest):
	if True:
		d.append(xyz[0])
		e.append(xyz[1])
		f.append(xyz[2])

print("Max: ",max(x),max(y),max(z))
print("Min: ",min(x),min(y),min(z))

ax.scatter3D(x, y, z, c=z, cmap='binary');
ax.scatter3D(a, b, c, c=c, cmap='coolwarm_r');
ax.scatter3D(d, e, f, c=f, cmap='jet');

plt.show()
