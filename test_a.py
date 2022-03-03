import numpy as np
import trimesh

# attach to logger so trimesh messages will be printed to console
v = np.array(
    [
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
        [1.4, 0, 0],
        [1, 31, 0],
        [0, 10, 0],
        [1, 1.3, 1],
        [1, 0.9, 0],
        [1, 1, 0],
        [0, 12, 9],
        [1, 23, 10],
    ]
)
f = np.array(
    [
        [0, 1, 3],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3],
    ]
)

print(v.shape)
print(f.shape)
# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(vertices=v, faces=f)

mesh.show()
