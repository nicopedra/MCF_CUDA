#include <stdlib.h>     // srand, rand: to generate random number
#include <iostream>     
#include <fstream>      
#include <cmath>        
#include <vector>

using namespace std;

//reading input parameters
ifstream ReadInput,ReadConf,ReadPrecedentConf;
string file_start;

//parameters, observables
const int m_props=5;
const int nbins=100;
int n_props;
//for measure istantaneous properties
double stima_pot, stima_kin, stima_etot, stima_temp,stima_press;
//tail correction due to rcutoff
double vtail;
double ptail;
//to save mean temperature
double m_temp;
double accettazione;
//for gdir
double bin_size;
vector<vector<double>> gdir(nbins);
//for instantaneous values and then do data analysis
vector<vector<double>> properties(m_props);
//parameter to restart
int restart;

//configuration
double *x,*y,*z,*xold,*yold,*zold,*vx,*vy,*vz;

// thermodynamical state
int npart;
double energy,temp,vol,rho,box,rcut;

// parameters for simulation
int nstep, iprint, seed;
double delta;

//####################### functions ###############################
void Input(void);

void exit();

void Move(void);

void ConfFinal(void);

//void ConfXYZ(int);

void Measure(void);

double Force(int, int);

double Pbc(double);

void print_properties();

void Print(vector<double>,string);

double error(vector<double>,vector<double>,int);

void data_blocking(int,vector<double>,double,string);

double mean(vector<double>,int,int = 0);

void data_blocking_MD(int); 

vector<double> last_data_from_datablocking(int,vector<double>);


//########################### IMPLEMENTAZIONI ##################################

void Initialization() {

if(restart == 1) {
	  file_start = "config.0";
	  cout << "reading configurations from precedent simulation: " << endl;
	  ReadConf.open(file_start);
	  ReadConf >> npart;
  }
  else    {
	  file_start = "config.fcc";
	  cout << "using method of random velocities: " << endl;
	  ReadConf.open(file_start);
	  ReadConf >> npart;
  }

 cout << "Read initial configuration from file "+file_start << endl << endl;
  for (int i=0; i<npart; ++i){
    ReadConf >> x[i] >> y[i] >> z[i];
    x[i] = x[i] * box;
    y[i] = y[i] * box;
    z[i] = z[i] * box;
  }
  ReadConf.close();

if(restart == 1) {
	ReadPrecedentConf.open("config.final");
	for (int i=0; i<npart; ++i){
    	 ReadPrecedentConf >> xold[i] >> yold[i] >> zold[i];
    	 xold[i] = xold[i] * box;
    	 yold[i] = yold[i] * box;
    	 zold[i] = zold[i] * box;
  	}
  	ReadPrecedentConf.close();
	double sumv2=0.0,fs;
	Move();
	for (int i=0; i<npart; ++i){
	vx[i] = Pbc(x[i] - xold[i])/(delta);
    	vy[i] = Pbc(y[i] - yold[i])/(delta);
    	vz[i] = Pbc(z[i] - zold[i])/(delta);
	sumv2 += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
	}
	sumv2 /= (double)npart;
	double T = sumv2/3.;//from the equipartion theorem
	fs = sqrt(temp/T);//scale factor
	
	for (int i=0; i<npart; ++i){
     		vx[i] *= fs;
     		vy[i] *= fs;
     		vz[i] *= fs;

     		xold[i] = Pbc(x[i] - vx[i] * delta);
     		yold[i] = Pbc(y[i] - vy[i] * delta);
     		zold[i] = Pbc(z[i] - vz[i] * delta);
        }
     }

else {
//Prepare initial velocities
   cout << "Prepare random velocities with center of mass velocity equal to zero " << endl << endl;
   double sumv[3] = {0.0, 0.0, 0.0};
   for (int i=0; i<npart; ++i){
     vx[i] = rand()/double(RAND_MAX) - 0.5; //centrate in 0
     vy[i] = rand()/double(RAND_MAX) - 0.5;
     vz[i] = rand()/double(RAND_MAX) - 0.5;

     sumv[0] += vx[i];
     sumv[1] += vy[i];
     sumv[2] += vz[i];
   } //servono per calcolare la posizione al tempo precedente rispetto a quello iniziale
   for (int idim=0; idim<3; ++idim) sumv[idim] /= (double)npart;
   double sumv2 = 0.0, fs;
   for (int i=0; i<npart; ++i){ //così evito drift rispetto al centro di massa del sistema
     vx[i] = vx[i] - sumv[0];
     vy[i] = vy[i] - sumv[1];
     vz[i] = vz[i] - sumv[2];

     sumv2 += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
   }
   sumv2 /= (double)npart;
   fs = sqrt(3 * temp / sumv2);   // fs = velocity scale factor 
   for (int i=0; i<npart; ++i){
     vx[i] *= fs;
     vy[i] *= fs;
     vz[i] *= fs;

     xold[i] = Pbc(x[i] - vx[i] * delta);
     yold[i] = Pbc(y[i] - vy[i] * delta);
     zold[i] = Pbc(z[i] - vz[i] * delta);
   }
}

Measure();

cout << "Initial potential energy (with tail corrections) = " << stima_pot << endl;
cout << "Pressure (with tail corrections) = " << stima_press << endl << endl;
cout << "initial Ekin = " << stima_kin << endl << endl; //deve venire 1.2 per il solido

};



void Input(void){ //Prepare all stuff for the simulation
  //ifstream ReadInput,ReadConf,ReadPrecedentConf;

  cout << "Classic Lennard-Jones fluid        " << endl;
  cout << "Molecular dynamics simulation in NVE ensemble  " << endl << endl;
  cout << "Interatomic potential v(r) = 4 * [(1/r)^12 - (1/r)^6]" << endl << endl;
  cout << "The program uses Lennard-Jones units " << endl;

  //string file_start;
  if(restart == 1) {
	  file_start = "config.0";
	  ReadConf.open(file_start);
	  ReadConf >> npart;
  }
  else    {
	  file_start = "config.fcc";
	  ReadConf.open(file_start);
	  ReadConf >> npart;
  }
  ReadConf.close();
//allocate memory
  x = new double [npart];
  y = new double [npart];
  z = new double [npart];
  xold = new double [npart];
  yold = new double [npart];
  zold = new double [npart];
  vx = new double [npart];
  vy = new double [npart];
  vz = new double [npart];

  seed = 1;    //Set seed for random numbers
  srand(seed); //Initialize random number generator
  ReadInput.open("input.dat"); //Read input

  ReadInput >> temp;
  cout << "target temperature = " << temp << endl;
  cout << "Number of particles = " << npart << endl;
  ReadInput >> rho;
  cout << "Density of particles = " << rho << endl;
  vol = (double)npart/rho;
  cout << "Volume of the simulation box = " << vol << endl;
  box = pow(vol,1.0/3.0);
  cout << "Edge of the simulation box = " << box << endl;//unità sigma

  ReadInput >> rcut;
  cout << "cutoff r= " << rcut << endl;
  ReadInput >> delta;//delta t, lo step temporale, abbastanza piccolo per conservare l'energia
  ReadInput >> nstep;
  ReadInput >> iprint;//ogni quanto stampare a video il punto della simulazione a cui sono arrivato
  cout << "The program integrates Newton equations with the Verlet method " << endl;
  cout << "Time step = " << delta << endl;
  cout << "Number of steps = " << nstep << endl << endl;
  ReadInput.close();

//Prepare array for measurements   //they're just indices
  n_props = 5; //Number of observables, already add pressure

  //correzioni di tail al potenziale e all ptail
  vtail = (8.0*M_PI*rho)/(9.0*pow(rcut,9)) - (8.0*M_PI*rho)/(3.0*pow(rcut,3));
  ptail = (32.0*M_PI*rho)/(9.0*pow(rcut,9)) - (16.0*M_PI*rho)/(3.0*pow(rcut,3));
  cout << "vtail: " << vtail << endl;
  cout << "ptail: " << ptail << endl;
  bin_size = (box*0.5)/nbins; 
  cout << "size of each bin: " << bin_size << endl;
};


void exit() {

	delete [] x;
	delete [] y;
	delete [] z;
	delete [] xold;
	delete [] yold;
	delete [] zold;
	delete [] vx;
	delete [] vy;
	delete [] vz;

}

void Move(void){ //Move particles with Verlet algorithm
  double xnew, ynew, znew, fx[npart], fy[npart], fz[npart];

  for(int i=0; i<npart; ++i){ //Force acting on particle i
    fx[i] = Force(i,0);
    fy[i] = Force(i,1);
    fz[i] = Force(i,2);
  }

  for(int i=0; i<npart; ++i){ //Verlet integration scheme

    xnew = Pbc( 2.0 * x[i] - xold[i] + fx[i] * pow(delta,2) );
    ynew = Pbc( 2.0 * y[i] - yold[i] + fy[i] * pow(delta,2) );
    znew = Pbc( 2.0 * z[i] - zold[i] + fz[i] * pow(delta,2) );

    vx[i] = Pbc(xnew - xold[i])/(2.0 * delta);
    vy[i] = Pbc(ynew - yold[i])/(2.0 * delta);
    vz[i] = Pbc(znew - zold[i])/(2.0 * delta);

    xold[i] = x[i];
    yold[i] = y[i];
    zold[i] = z[i];

    x[i] = xnew;
    y[i] = ynew;
    z[i] = znew;
  }
  return;
}

double Force(int ip, int idir){ //Compute forces as -Grad_ip V(r)
  double f=0.0;
  double dvec[3], dr;

  for (int i=0; i<npart; ++i){
    if(i != ip){//su tutte le particelle tranne la stessa
      dvec[0] = Pbc( x[ip] - x[i] );  // distance ip-i in pbc
      dvec[1] = Pbc( y[ip] - y[i] );
      dvec[2] = Pbc( z[ip] - z[i] );

      dr = dvec[0]*dvec[0] + dvec[1]*dvec[1] + dvec[2]*dvec[2];
      dr = sqrt(dr);

      if(dr < rcut){
        f += dvec[idir] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); // -Grad_ip V(r)
      }
    }
  }
  
  return f;
}

// per misurare le proprietà fisiche 

void Measure(){ //Properties measurement
  int bin;
  double v, w, t, vij, wij;
  double dx, dy, dz, dr;
  vector<double> hist(nbins*10,0.0);

  v = 0.0; //reset observables
  w = 0.0;
  t = 0.0;

//cycle over pairs of particles
  for (int i=0; i<npart-1; ++i){
    for (int j=i+1; j<npart; ++j){

     dx = Pbc( x[i] - x[j] );
     dy = Pbc( y[i] - y[j] );
     dz = Pbc( z[i] - z[j] );

     dr = dx*dx + dy*dy + dz*dz;
     dr = sqrt(dr);

     bin = int(dr/bin_size);
     hist[bin] = hist[bin]+2;

     if(dr < rcut){
       vij = 4.0/pow(dr,12) - 4.0/pow(dr,6);
       wij = 16.0/pow(dr,12) - 8.0/pow(dr,6);


//Potential energy
       v += vij;
       w += wij;
     }
    }          
  }

//Kinetic energy
  for (int i=0; i<npart; ++i) t += 0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);


// g(r)
  double deltaVr;
  for (int i=0;i<nbins;i++) {
	   deltaVr = rho*npart*4.*M_PI/3.*(pow((i+1)*bin_size,3)-pow((i)*bin_size,3));
	   gdir[i].push_back(hist[i]/deltaVr);
   }

    stima_pot = v/(double)npart+vtail; //Potential energy per particle
    stima_kin = t/(double)npart; //Kinetic energy per particle
    stima_temp = (2.0 / 3.0) * t/(double)npart; //Temperature
    stima_etot = (t+v)/(double)npart+vtail; //Total energy per particle
    stima_press = rho*stima_temp+(w+ptail*(double)npart)/vol;

    //saving here to do data_blockig later
    properties[0].push_back(stima_pot);
    properties[1].push_back(stima_kin);
    properties[2].push_back(stima_temp);
    properties[3].push_back(stima_etot);
    properties[4].push_back(stima_press);

    return;
}

void print_properties() {

  string name = "output_epot"+ to_string(nstep)+".dat";
  Print(properties[0],name);
  name = "output_ekin"+ to_string(nstep)+".dat";
  Print(properties[1],name);
  name = "output_temp"+ to_string(nstep)+".dat";
  Print(properties[2],name);
  name = "output_etot"+ to_string(nstep)+".dat";
  Print(properties[3],name);
  name = "output_press"+ to_string(nstep)+".dat";
  Print(properties[4],name);
}

void Print(vector<double> v, string name) {
   ofstream fd; fd.open(name,ios::app);
   for (auto& el : v) fd << el << endl;
   fd.close();
}

void ConfFinal(void){ //Write final configuration

  ofstream WriteOldConf("config.final",ios::out);
  cout << "Print penultimate configuration in config.final " << endl << endl;
  for (int i=0; i<npart; ++i){
    WriteOldConf << xold[i]/box << "   " <<  yold[i]/box << "   " << zold[i]/box << endl;
  } 
  WriteOldConf.close();

  ofstream WriteConf("config.0",ios::out);
  cout << "Print final configuration to file config.0 " << endl << endl;
  WriteConf << npart << endl;
  for (int i=0; i<npart; ++i){
    WriteConf << x[i]/box << "   " <<  y[i]/box << "   " << z[i]/box << endl;
  }
  WriteConf.close();
  return;
}

/*
void ConfXYZ(int nconf){ //Write configuration in .xyz format
  fstream WriteXYZ;

  WriteXYZ.open("traj.xyz",ios::app);
  if(nconf==1) {
  WriteXYZ << npart << endl;
  WriteXYZ << "This is only a comment!" << endl;
  }
  for (int i=0; i<npart; ++i){
    WriteXYZ << "LJ  " << Pbc(x[i]) << "   " <<  Pbc(y[i]) << "   " << Pbc(z[i]) << endl;
  }
  WriteXYZ.close();
}
*/

double Pbc(double r) {  //Algorithm for periodic boundary conditions with side L=box
    return r - box * rint(r/box);
}

vector<double> last_data_from_datablocking(int N,vector<double> simulation_value) {

 vector<double> err_prog;
 vector<double> sum_prog(N,0.);
 vector<double> simulation_value2;
 vector<double> su2_prog(N,0.);

 for (int i=0;i<N;i++) simulation_value2.push_back(simulation_value[i]*simulation_value[i]);

 for (int i=0; i<N; i++) {
         for (int j=0; j<i+1; j++) {
                 sum_prog[i] += simulation_value[j];
                 su2_prog[i] += simulation_value2[j];
         }
         sum_prog[i]/=(i+1);
         su2_prog[i]/=(i+1);
         err_prog.push_back(error(sum_prog,su2_prog,i));
 }
vector<double> data = {sum_prog[N-1],err_prog[N-1]};

        return data;
};


void data_blocking_MD(int N) {

int L = (nstep/10.)/N; //cause I measure properties each 10 steps
vector<string> names = {"ave_epot","ave_ekin","ave_temp","ave_etot","ave_press"};
int j=0;
vector<double> v_mean;
vector<double> data(2);
 for (auto & el : names) {
	for (int i=0;i<N;i++) 
		 v_mean.push_back( mean(properties[j], (i+1)*L, i*L ));
	 if ( j== 2) data = last_data_from_datablocking(N,v_mean);
	 data_blocking(N,v_mean,0,el+to_string(nstep)+".out");
	 properties[j].clear();
	 j++;
	 v_mean.clear();
 }
 accettazione = data[1];
 m_temp = data[0];
 // per controllare a che temperatura sono arrivato
 cout << "temperatura di ora: " << data[0] << " , con incertezza: " << data[1]<< endl;
 v_mean.clear();

 // radial correlation function
 
 string gdir_name = "output.gave.out";
 ofstream Gave(gdir_name,ios::out);
 
 for (int i=0;i<nbins;i++) {
	 for (j=0;j<N;j++)
		 v_mean.push_back(mean(gdir[i],(j+1)*L,j*L));
	 data = last_data_from_datablocking(N,v_mean);
	 gdir[i].clear();
	 v_mean.clear();
	 Gave << (bin_size*0.5+bin_size*i) << "\t"  << data[0] << "\t" << data[1] << endl;
 }
 Gave.close();
};

void data_blocking(int N,vector<double> simulation_value, double real_value, string file) {

 vector<double> err_prog;
 vector<double> sum_prog(N,0.);
 vector<double> simulation_value2;
 vector<double> su2_prog(N,0.);

 for (int i=0;i<N;i++) simulation_value2.push_back(simulation_value[i]*simulation_value[i]);

 for (int i=0; i<N; i++) {
         for (int j=0; j<i+1; j++) {
                 sum_prog[i] += simulation_value[j];
                 su2_prog[i] += simulation_value2[j];
         }
         sum_prog[i]/=(i+1);
         su2_prog[i]/=(i+1);
         err_prog.push_back(error(sum_prog,su2_prog,i));
 }

         fstream fd;
         fd.open(file,ios::out);
         for (int i=0; i<N;i++) fd << sum_prog[i]-real_value<<" "<< err_prog[i] << endl;
         fd.close();

};

double error(vector<double> AV, vector<double> AV2, int i) {
        if (i==0) return 0;
        else return sqrt( (AV2[i]-AV[i]*AV[i]) / double(i) );
};

double mean(vector<double> v,int last_index, int first_index) {
	double sum = 0;
	for (int i=first_index; i<last_index; i++) sum += v[i];
        return sum/(last_index-first_index);
}; 


