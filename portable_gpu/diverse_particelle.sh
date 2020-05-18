#/bin/bash


	for i in config_108.fcc config_256.fcc config_500.fcc config_864.fcc config_1372.fcc config_2048.fcc  config_2916.fcc config_4000.fcc;
	do
		   echo $i
		   cp config/$i config.fcc	
		   ./MolDyn_NVE.x 0 10000 ## prima configurazione random
	done

