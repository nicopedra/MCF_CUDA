#include "MolDyn_NVE.h"

//nvprof --cpu-profiling on --cpu-profiling-mode flat ./MolDyn_NVE.x 0

//USING VERLET LIST
//È UTILE QUANDO CI SONO MOLTE PARTICELLE E QUANDO LA DENSITÀ È ALTA
//SE LA DENSITÀ È BASSA ALLORA OVVIAMENTE DEVO AGGIORNARE SPESSO LA LISTA
//INOLTRE È CONVENIENTE QUANDO IL NUMERO DI PRIMI VICINI È MOLTO MINORE DEL NUMERO
//DI PARTICELLE TOTALE: n_nn = 4/3 PI rm^3 rho, stimando per argon circa 65

//#define equilibration

using namespace std;

int main(int argc, char** argv){

  int tentativo = 1;
  temp = 0.8;
  m_temp=0;
  cont_ag = 0;
 // int nconf;
  int N=100; //blocks for blocking
  accettazione = 0.001;
  float errore = abs(m_temp-temp); 
  clock_t start,stop;
 //reading input parameters and allocate memory
  Input();
 
#ifdef equilibration

  cout << "equilibration phase! " << endl;
  while ( errore > accettazione ) { //utile per equilibrazione
        cout <<"################################################################" << endl;
cout << "                tentativo numero: " << tentativo << endl;
        cout <<"################################################################" << endl;
        if (tentativo == 1)
                restart = 1;
        else
                restart = 1;


#else
  if (argc!=2) {cerr << "insert restart parameter" << endl;
              return -1;}

  restart = atoi(argv[1]);

#endif

  cout << "restart = " << restart << endl;
  Initialization();//starting configurations
  //nconf = 1;
  start = clock();
  cout << "starting measuring execution time " << endl;

  for(int istep=1; istep <= nstep; ++istep) {
     if ( check_aggiornamento() ) {
			aggiorno_primi_vicini(); 
			//cont_ag++;
			//cout << "aggiornamento: " <<cont_ag<< endl;
     }
     Move();           //Move particles with Verlet algorithm
     if(istep%iprint == 0) cout << "Number of time-steps: " << istep << endl;
     if(istep%10 == 0){
        Measure();     //Properties measurement
//        ConfXYZ(nconf);//Write actual configuration in XYZ format 
       // nconf += 1;
     }
  }


  stop = clock();
  cout << "stop time" << endl;
  print_properties();
  data_blocking_MD(N);

  ofstream Time("simulation.time",ios::app);
  float elapsedTime = (float)(stop-start)/(float)CLOCKS_PER_SEC*1000.0f;
  printf("Time passed: %3.1f ms\n", elapsedTime);
  Time << npart << "\t" << elapsedTime << endl;
  Time.close();

  errore = abs(m_temp-temp);
  cout << "ora l'errore tra la temperatura del sistema e quella target è: " << errore << endl;

  system ("make copy");
  ConfFinal();
#ifdef equilibration
  tentativo++;
  }
#endif
  //free memory
  exit(); 
  cout << endl;
  cout <<"################################################################" << endl;
  cout << "REMEMBER: if want to save final and penultimate configurations" << endl;
  cout <<"in file old.0 (last one) and old.final(penultimate) do command-> make copy" << endl;
  cout <<"##################################################################" << endl;
  cout << endl;

  return 0;
}

