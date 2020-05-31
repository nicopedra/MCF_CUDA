set terminal png
set output 'Performance_gpu.png'
set autoscale
unset key
	set title 'Time in function of number of particles' font "Helvetica Bold,18"
	set ylabel "Time t (s)" font "Helvetica Bold,18" offset 2,0
	set xlabel "number of particles" font "Helvetica Bold,18" 
plot 'simulation_gpu.time' u 1:($2/1000) w l lt rgb "#FF6619"

reset

set terminal png
set output 'Performance_cpu.png'
set autoscale
unset key
	set title 'Time in function of number of particles' font "Helvetica Bold,18"
	set ylabel "Time t (s)" font "Helvetica Bold,18" offset 2,0
	set xlabel "number of particles" font "Helvetica Bold,18" 
plot 'simulation_cpu.time' u 1:($2/1000) w l lt rgb "#FF6619"

