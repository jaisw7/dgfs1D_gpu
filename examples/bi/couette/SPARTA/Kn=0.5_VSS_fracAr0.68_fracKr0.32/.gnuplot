
#tail -n +10 prop_grid.100000000.dat | sort -g -k1,1 > multiSpecie.dat

plot 'singleSpecie.dat' u ($2):($8), 'multiSpecie.dat' u ($2):(($6*$8+$11*$13)/($6+$11))
