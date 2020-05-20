#include <stdlib.h>     // srand, rand: to generate random number
#include <iostream>     // cin, cout: Standard Input/Output Streams Library
#include <fstream>      // Stream class to both read and write from/to files.
#include <cmath>        // rint, pow
#include <vector>
#include <cstdio>
//#include <stdio.h>

using namespace std;

//parameters, observables
const int m_props=5;
const int nbins=100;
int n_props;
float vtail;
float ptail;
float m_temp;
float accettazione;
float bin_size;
vector<vector<float>> gdir(nbins);
vector<vector<float>> properties(m_props);
float stima_pot, stima_kin, stima_etot, stima_temp,stima_press;//per test cpu
int restart;
//############### dimension for a specific molecule #############
  float sigma;//nm
  float eps_kb;//kelvin
  float eps = eps_kb*(1.38e-23);//Joule
  float m;//amu
//##############################################
//######## Argon ##########
  //sigma = 0.34e-9;//m
  //eps_kb = 120;//kelvin
  //eps = eps_kb*(1.38e-23);//Joule
  //m = 39.948;//amu
//######################
//configuration
const int m_part=108;

// thermodynamical state
int npart;
float energy,temp,vol,rho,box,rcut;

//float *x,*y,*z,*xold,*yold,*zold,*vx,*vy,*vz;
vector<float> x,y,z,xold,yold,zold,vx,vy,vz,Fx,Fy,Fz;
//float *Fx,*Fy,*Fz;

// simulation
int nstep, iprint, seed;
float delta;

//################### functions ####################

void Input();

void ConfFinal();

//void ConfXYZ(int);
float Pbc(float);

template <typename T>
void Print(vector<T>,string);

float error(vector<float>,vector<float>,int);

void data_blocking(int,vector<float>,float,string);

vector<float> last_data_from_datablocking(int,vector<float>);

float mean_v(vector<float>,int,int = 0);

void data_blocking_MD(int);

void print_conf();

void print_old_conf();

void print_properties_cpu(); 

void Measure_cpu();

void Move_cpu();

void Force_cpu();  

void print_Force(); 

void print_Force_file(); 

void Exit();

void print_velocities();

//########################### IMPLEMENTAZIONI #################################

template< typename T >
void swap( T& a, T& b ) {
    T t = a;
    a = b;
    b = t;
}


void Input(){ //Prepare all stuff for the simulation
  ifstream ReadInput,ReadConf,ReadPrecedentConf;

  cout << "Classic Lennard-Jones fluid        " << endl;
  cout << "Molecular dynamics simulation in NVE ensemble  " << endl << endl;
  cout << "Interatomic potential v(r) = 4 * [(1/r)^12 - (1/r)^6]" <<"\n"<<"\n";
  cout << "The program uses Lennard-Jones units " << endl;

  string file_start;
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
 

  seed = 1;    //Set seed for random numbers
  srand(seed); //Initialize random number generator
  ReadInput.open("input.dat"); //Read input

  ReadInput >> temp;
  cout << "target temperature = " << temp << endl;
  cout << "Number of particles = " << npart << endl;
  ReadInput >> rho;
  cout << "Density of particles = " << rho << endl;
  vol = (float)npart/rho;
  cout << "Volume of the simulation box = " << vol << endl;
  box = pow(vol,1.0/3.0);
  cout << "Edge of the simulation box = " << box << endl;//unità sigma

  ReadInput >> rcut;
  cout << "cutoff r: " << rcut << endl;
  ReadInput >> delta;
  ReadInput >> nstep;
  ReadInput >> iprint; //ogni quanto stampare a che punto sono della simulazione

  cout << "The program integrates Newton equations with the Verlet method " << endl;
  cout << "Time step = " << delta << endl;
  cout << "Number of steps = " << nstep << endl << endl;
  ReadInput.close();

  vtail = (8.0*M_PI*rho)/(9.0*pow(rcut,9)) - (8.0*M_PI*rho)/(3.0*pow(rcut,3));
  ptail = (32.0*M_PI*rho)/(9.0*pow(rcut,9)) - (16.0*M_PI*rho)/(3.0*pow(rcut,3));
  cout << "vtail: " << vtail << endl;
  cout << "ptail: " << ptail << endl;
  bin_size = (box*0.5)/nbins; 
  cout << "size of each bin: " << bin_size << endl;  

//Prepare array for measurements   //they're just indices
  n_props = 5; //Number of observables, already add pressure

Fx.resize(npart*npart,0.0);
Fy.resize(npart*npart,0.0);
Fz.resize(npart*npart,0.0);

float xv,yv,zv;
	
//Read initial configuration
  cout << "Read initial configuration from file "+file_start << endl << endl;
  for (int i=0; i<npart; ++i){
    ReadConf >> xv >> yv >> zv;
    x.push_back(xv*box); y.push_back(yv*box); z.push_back(zv*box);
  }
  ReadConf.close();
if(restart == 1) {
	ReadPrecedentConf.open("config.final");
	for (int i=0; i<npart; ++i){
	ReadPrecedentConf >> xv >> yv >> zv;
        xold.push_back(xv*box); yold.push_back(yv*box); zold.push_back(zv*box);
  	}
	cout << endl;
  	ReadPrecedentConf.close();
	float sumv2=0.0,fs;
        Force_cpu();
        vx.resize(npart); vy.resize(npart); vz.resize(npart);
	Move_cpu();
	for (int i=0; i<npart; ++i){
        vx.push_back(Pbc(x[i] - xold[i])/(2.0 * delta));
    	vy.push_back(Pbc(y[i] - yold[i])/(2.0 * delta));
    	vz.push_back(Pbc(z[i] - zold[i])/(2.0 * delta));        
	sumv2 += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
	}
	cout << endl;
	sumv2 /= (float)npart;
	float T = sumv2/3.;//from the equipartion theorem
	fs = sqrt(temp/T);//scale factor
	cout << "scale factor: " << endl;
	cout << fs << endl;
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
     vx.push_back(rand()/double(RAND_MAX) - 0.5); //centrate in 0
     vy.push_back(rand()/double(RAND_MAX) - 0.5);
     vz.push_back(rand()/double(RAND_MAX) - 0.5);

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

     xold.push_back(Pbc(x[i] - vx[i] * delta));
     yold.push_back(Pbc(y[i] - vy[i] * delta));
     zold.push_back(Pbc(z[i] - vz[i] * delta));
   }
}

 Force_cpu();
 print_Force();
 print_Force_file(); 
 Measure_cpu();

 cout << "Initial potential energy (with tail corrections) = " << stima_pot << endl;
 cout << "Pressure (with tail corrections) = " << stima_press << endl << endl;
 cout << "Pressure (with tail corrections) = " << stima_kin << endl << endl;   

}


void Exit() {

x.clear();
y.clear();
z.clear();
xold.clear();
yold.clear();
zold.clear();
vx.clear();
vy.clear();
vz.clear();
Fx.clear();
Fy.clear();
Fz.clear();

};


void Force_cpu() {

   float dvec[3];
   float dr;
   for (int i = 0 ; i<npart ;i++) {
	for (int j = 0 ; j < i;j++) {
		dvec[0] = Pbc(x[i]-x[j]) ;
		dvec[1] = Pbc(y[i]-y[j]) ;
		dvec[2] = Pbc(z[i]-z[j]) ;
		dr = sqrt(dvec[0]*dvec[0] + dvec[1]*dvec[1] + dvec[2]*dvec[2]); 
		if (dr>0 && dr<rcut) {
			Fx[i*npart+j] = dvec[0]*(48.0/pow(dr,14) - 24.0/pow(dr,8));
			Fy[i*npart+j] = dvec[1]*(48.0/pow(dr,14) - 24.0/pow(dr,8));
			Fz[i*npart+j] = dvec[2]*(48.0/pow(dr,14) - 24.0/pow(dr,8));			
		}
		else {
			Fx[i*npart+j] = 0;
			Fy[i*npart+j] = 0;
			Fz[i*npart+j] = 0;
		}

		Fx[j*npart+i] = -Fx[i*npart+j];
		Fy[j*npart+i] = -Fy[i*npart+j];
		Fz[j*npart+i] = -Fz[i*npart+j];
	}
   }
};

void print_Force_file() {


 ofstream printfx("FORCEX.txt",ios::out);
 for (int i=0; i<npart; ++i){
	for (int j = 0; j<npart;j++) {
    		printfx << Fx[i*npart+j] << "   ";
	}
  printfx << endl;
 }
 printfx.close();

 ofstream printfy("FORCEY.txt",ios::out);
 for (int i=0; i<npart; ++i){
	for (int j = 0; j<npart;j++) {
    		printfy << Fy[i*npart+j] << "   ";
	}
  printfy << endl;
 }
printfy.close();

ofstream printfz("FORCEZ.txt",ios::out);
 for (int i=0; i<npart; ++i){
	for (int j = 0; j<npart;j++) {
    		printfz << Fz[i*npart+j] << "   ";
	}
  printfz << endl;
 }
printfz.close();
 
};

void print_Force() {

cout<< "\n";
 cout << "Print matrix force:" << endl;
 cout << "FORCE X" << endl;
 for (int i=0; i<npart; ++i){
	for (int j = 0; j<npart;j++) {
    		cout << Fx[i*npart+j] << "   ";
	}
  cout << endl;
 }

 cout << "\n";
 cout << "FORCE Y" << endl;

 for (int i=0; i<npart; ++i){
	for (int j = 0; j<npart;j++) {
    		cout << Fy[i*npart+j] << "   ";
	}
  cout << endl;
 }

cout << "\n";
cout << "FORCE Z" << endl;


 for (int i=0; i<npart; ++i){
	for (int j = 0; j<npart;j++) {
    		cout << Fz[i*npart+j] << "   ";
	}
  cout << endl;
 }

cout<< "\n";
 
};

void Move_cpu() {

	float fx,fy,fz;
	float xnew,ynew,znew;
	//float dvec[3]; float dr;

	for (int ip=0;ip<npart;ip++) {
                fx = 0; fy = 0; fz=0;
                for (int i=0;i<npart;i++) {
		/*		
                        if (i != ip) {

                                dvec[0] = Pbc(x[ip]-x[i]);
                                dvec[1] = Pbc(y[ip]-y[i]);
                                dvec[2] = Pbc(z[ip]-z[i]);
                                dr = sqrt(dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2]);

                                if (dr < rcut && dr> 0) {
                                        fx += dvec[0] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); ;
                                        fy += dvec[1] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); ;
                                        fz += dvec[2] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); ;
                                }
                        }
		*/
		fx += Fx[ip*npart+i];
		fy += Fy[ip*npart+i];
		fz += Fz[ip*npart+i];
                }

        xnew = Pbc(2.0*x[ip]-xold[ip]+fx*pow(delta,2));
        ynew = Pbc(2.0*y[ip]-yold[ip]+fy*pow(delta,2));
        znew = Pbc(2.0*z[ip]-zold[ip]+fz*pow(delta,2));


        vx[ip] = Pbc(xnew-xold[ip])/(2.0*delta);
        vy[ip] = Pbc(ynew-yold[ip])/(2.0*delta);
        vz[ip] = Pbc(znew-zold[ip])/(2.0*delta);

        xold[ip] = x[ip];
        yold[ip] = y[ip];
        zold[ip] = z[ip];

        x[ip] = xnew;
        y[ip] = ynew;
        z[ip] = znew;

        }
}



void Measure_cpu() { 

  int bin;
  double v, w, t, vij, wij;
  double dx, dy, dz, dr;
  vector<float> hist(nbins*10,0.0);

  v = 0.0; //reset observables
  w = 0.0;
  t = 0.0;

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
                v += vij;
                w += wij;
           }
      }          
   }
   for (int i=0; i<npart; ++i) t += 0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
       
   for (int i=0;i<nbins;i++) {
	   float deltaVr = rho*npart*4.*M_PI/3.*(pow((i+1)*bin_size,3)-pow((i)*bin_size,3));
	   gdir[i].push_back(hist[i]/deltaVr);
   }
                                                                         
   stima_pot = v/(double)npart+vtail; //Potential energy per particle
   stima_kin = t/(double)npart; //Kinetic energy per particle
   stima_temp = (2.0 / 3.0) * t/(double)npart; //Temperature
   stima_etot = (t+v)/(double)npart+vtail; //Total energy per particle
   stima_press = rho*stima_temp+(w+ptail*npart)/vol;

   properties[0].push_back(stima_pot);
   properties[1].push_back(stima_kin);
   properties[2].push_back(stima_temp);
   properties[3].push_back(stima_etot);
   properties[4].push_back(stima_press);

}

void print_properties_cpu() {

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

template <typename T>
void Print(vector<T> v, string name) {
   ofstream fd; fd.open(name,ios::app);
   for (auto& el : v) fd << el << endl;
   fd.close();
}


void print_velocities() {

cout<< "\n";
 cout << "Print actual velocities:" << endl;

 for (int i=0; i<npart; ++i){
    cout << vx[i] << "   " <<  vy[i] << "   " << vz[i] << endl;
 }

cout<< "\n";
}



void print_old_conf() {

cout<< "\n";
 cout << "Print actual old configuration" << endl;

 for (int i=0; i<npart; ++i){
    cout << xold[i]/box << "   " <<  yold[i]/box << "   " << zold[i]/box << endl;
 }

cout<< "\n";
}

void print_conf() {

   cout<< "\n";
 cout << "Print actual configuration" << endl;
 for (int i=0; i<npart; ++i){
    cout << x[i]/box << "   " <<  y[i]/box << "   " << z[i]/box << endl;
 }

cout <<"\n";

}

void ConfFinal(){ //Write final configuration

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
}


float Pbc(float r) {  //Algorithm for periodic boundary conditions with side L=box
    return r - box * rintf(r/box);
}

void data_blocking_MD(int N) {

int L = (nstep/10.)/N; //cause I measure properties each 10 steps
vector<string> names = {"ave_epot","ave_ekin","ave_temp","ave_etot","ave_press"};
int j=0;
vector<float> v_mean;
vector<float> data(2);
 for (auto & el : names) {
	for (int i=0;i<N;i++) 
		 v_mean.push_back( mean_v(properties[j], (i+1)*L, i*L ));
	 if ( j== 2) data = last_data_from_datablocking(N,v_mean);
	 data_blocking(N,v_mean,0,el+to_string(nstep)+".out");
	 properties[j].clear();
	 j++;
	 v_mean.clear();
 }
 accettazione = 2*data[1];
 m_temp = data[0];
 cout << "temperatura di ora: " << data[0] << " , con incertezza: " << data[1]<< endl;
 v_mean.clear();
 // radial correlation function
 
 string gdir_name = "output.gave.out";
 ofstream Gave(gdir_name,ios::out);
 
 for (int i=0;i<nbins;i++) {
	 for (j=0;j<N;j++)
		 v_mean.push_back(mean_v(gdir[i],(j+1)*L,j*L));
	 data = last_data_from_datablocking(N,v_mean);
	 gdir[i].clear();
	 v_mean.clear();
	 Gave << (bin_size*0.5+bin_size*i) << "\t"  << data[0] << "\t" << data[1] << endl;
 }
 Gave.close();
};

vector<float> last_data_from_datablocking(int N,vector<float> simulation_value) {

 vector<float> err_prog;
 vector<float> sum_prog(N,0.);
 vector<float> simulation_value2;
 vector<float> su2_prog(N,0.);

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
vector<float> data = {sum_prog[N-1],err_prog[N-1]};

        return data;
};

void data_blocking(int N,vector<float> simulation_value, float real_value, string file) {

 vector<float> err_prog;
 vector<float> sum_prog(N,0.);
 vector<float> simulation_value2;
 vector<float> su2_prog(N,0.);

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

         ofstream fd;
         fd.open(file,ios::out);
         for (int i=0; i<N;i++) fd << sum_prog[i]-real_value<<" "<< err_prog[i] << endl;
         fd.close();

};

float error(vector<float> AV, vector<float> AV2, int i) {
        if (i==0) return 0;
        else return sqrt( (AV2[i]-AV[i]*AV[i]) /(float)i );
};

float mean_v(vector<float> v,int last_index, int first_index) {
	float sum = 0;
	for (int i=first_index; i<last_index; i++) sum += v[i];
        return sum/(last_index-first_index);
}; 


