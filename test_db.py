from tables import *
import numpy as np


class RegionFeatures(IsDescription):
    region_features = Float32Col((128, 200))
    start_indexes = Int8Col()
    end_indexes = Int8Col()


h5file = open_file("region-features/region_features.h5", mode="w", title="Region Features Database")
group = h5file.create_group("/", 'detector', 'Detector information')
table = h5file.create_table(group, 'readout', RegionFeatures, "Readout example")
row_pointer = table.row

for i in range(100000000000):
    x = np.zeros((128, 200), dtype=np.float)
    row_pointer['region_features'] = x
    row_pointer.append()

print("This is the value for batch number {}: {}".format(10, row_pointer['region_features'][10]))
