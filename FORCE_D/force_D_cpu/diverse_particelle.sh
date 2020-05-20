#/bin/bash

	## preparazione delle cartelle necessarie

#	mkdir gas_Ar
#	mkdir generic_liquid_10000
#	mkdir generic_solid_10000
#	mkdir generic_liquid_100000
#	mkdir generic_solid_100000
#	mkdir liquid_Ar
#	mkdir solid_Ar
#	mkdir raggiungimento_eq

#	cd raggiungimento_eq 
#	mkdir gas_Ar
#	mkdir generic_liquid
#	mkdir generic_solid
#	mkdir liquid_Ar
#	mkdir solid_Ar
	
#	cd ..

####### mostro il raggiugimento della temperatura target ##################


	for i in config_108.fcc config_256.fcc config_500.fcc config_864.fcc config_1372.fcc config_2048.fcc  config_2916.fcc config_4000.fcc;
	do
		   echo $i
		   cp config/$i config.fcc	
		   ./MolDyn_NVE.x 0 10000 ## prima configurazione random
	done

	#mv output* raggiungimento_eq/generic_solid

	#./clean.sh
