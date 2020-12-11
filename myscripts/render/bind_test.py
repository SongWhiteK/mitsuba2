from time import time
import numpy as np
import torch
from mitsuba.heightmap import HeightMap

array = torch.randn([6, 6])

map_list = HeightMap(array, 3)

inpos = torch.tensor([1,2])
mesh_id = torch.tensor([1,2,3,4,5])

a = map_list.get_height_map(inpos, mesh_id)

print(a)