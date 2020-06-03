#include "MolDyn_NVE.h"

//#define equilibration

using namespace std;

int main(int argc, char** argv){

  int tentativo = 1;
  temp = 0.8;
  m_temp=0;
  int N;
  accettazione = 0.001;
  double errore = abs(m_temp-temp); 
  Particles P;
  Input(&P); 

  //assign symbol to costant memory
  HANDLE_ERROR(cudaMemcpyToSymbol(gpu_npart, &npart, sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(gpu_box, &box, sizeof(double)));
  HANDLE_ERROR(cudaMemcpyToSymbol(gpu_binsize, &bin_size, sizeof(double)));
  HANDLE_ERROR(cudaMemcpyToSymbol(gpu_delta, &delta, sizeof(double)));
  HANDLE_ERROR(cudaMemcpyToSymbol(gpu_rcut, &rcut, sizeof(double)));

#ifdef equilibration
  cout << "equilibration phase! " << endl;
  while ( errore > accettazione ) { //until equilibration
        cout <<"################################################################" << endl;
cout << "                tentativo numero: " << tentativo << endl;
        cout <<"################################################################" << endl;
        if (tentativo == 1)
                restart = 0;
        else
                restart = 1;

#else
  if (argc!=2) {cerr << "insert restart parameter" << endl;
              return -1;}

  restart = atoi(argv[1]);
#endif
  cout << "restart = " << restart << endl;

  P.TotalTime = 0;
  Initialization(&P);  //Initialize initial configurations

  Measure(&P);

  cout << "Initial potential energy (with tail corrections) = " << stima_pot_gpu << endl;
  cout << "Pressure (with tail corrections) = " << stima_press_gpu << endl;   
  cout << "Ekin = " << stima_kin_gpu << endl << endl;  //questa deve venire 1.2, perchè ho riscalato le velocità per avere
					              // temperatura = 0.8

  HANDLE_ERROR( cudaEventCreate( &P.start ));
  HANDLE_ERROR( cudaEventCreate( &P.stop ));
  HANDLE_ERROR( cudaEventRecord( P.start, 0 ) );
  cout<< "\n\n";

  N = 100; //number of blocks for data_blocking analysis

 //doing MD steps
  for(int istep=1; istep <= nstep; ++istep) {
     Move_gpu(&P); //move with verlet-algorithm
     if (istep%10 == 0) Measure(&P); //measure physical properties
     if (istep%iprint == 0) cout << "Number of time-steps: " << istep << endl; 
  }
  
  //save instantaneous results on file
  print_properties(); 
  //doing data analysis and save on file the results
  data_blocking_MD(N); 
  
  //error between mean_temperature obtained from data analysis and target temperature
  errore = abs(m_temp-temp);
  cout << "ora l'errore tra la temperatura del sistema e quella target è: " << errore << endl;
  //save target configuration in old.0 and old.final
  system ("make copy");
  //overwrite config.0 and config.final
  ConfFinal(&P);   

  HANDLE_ERROR( cudaEventRecord( P.stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( P.stop ) );
  HANDLE_ERROR( cudaEventElapsedTime( &P.TotalTime,
                                        P.start, P.stop ) );
  ofstream Time("simulation.time",ios::app);
  Time << npart << "\t" << P.TotalTime << endl;
  Time.close();
  printf( "Time:  %3.1f ms\n",P.TotalTime);

#ifdef equilibration
  tentativo++;
  }
#else
  cout << endl;
  cout <<"################################################################" << endl;
  cout << "REMEMBER: if want to save final and penultimate configurations" << endl;
  cout <<"in file old.0 (last one) and old.final(penultimate) do command-> make copy" << endl;
  cout <<"##################################################################" << endl;
  cout << endl;
#endif
  exit(&P);

  return 0;
}
