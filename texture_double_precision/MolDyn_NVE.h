#include <stdlib.h>     // srand, rand: to generate random number
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include "lock.h"

#define bl 128 //512
#define th_verlet 128 //128
#define th_measure 128 //1024 
#define th_kinetic 128 //1024 
using namespace std;

//for Initialization
ifstream ReadInput,ReadConf,ReadPrecedentConf;
string file_start;

//parameters, observables

//number of properties
const int m_props=5;
const int nbins=100;
//tail corrections
double vtail;
double ptail;
//mean temperature from data analysis
double m_temp;
double accettazione;
//useful for measurement
double bin_size;
vector<vector<double>> gdir(nbins);
double stima_pot_gpu, stima_kin_gpu, stima_etot_gpu, stima_temp_gpu,stima_press_gpu;
vector<vector<double>> properties_gpu(m_props);

int restart;

	texture<int2,1> texx;
	texture<int2,1> texy;
	texture<int2,1> texz;
	texture<int2,1> texxold;
	texture<int2,1> texyold;
	texture<int2,1> texzold;
	texture<int2,1> texvx;
	texture<int2,1> texvy;
	texture<int2,1> texvz;

	texture<int2,1> texFx;
	texture<int2,1> texFy;
	texture<int2,1> texFz;

double* Fx,*Fy,*Fz;
//fetch but in double precision
static __inline__ __device__ double fetch_double(texture<int2, 1> t, int i)

{

	int2 v = tex1Dfetch(t,i);

	return __hiloint2double(v.y, v.x);

}

// thermodynamical state
int npart;
double energy,temp,vol,rho,box,rcut;

//using to calculate the physical properties on the gpu
double *dev_w,*dev_v,*dev_k; //virial, potential, kinetic 
double* dev_hist;// g(r)

// simulation
int nstep, iprint, seed;
double delta;

//allocate costant memory
__constant__ int gpu_npart;
__constant__ double gpu_binsize;
__constant__ double gpu_rcut;
__constant__ double gpu_delta;
__constant__ double gpu_box;

//structures
struct Particles {

	double* x;
	double* y;
	double* z;
	double* xold;
	double* yold;
	double* zold;
	double* vx;
	double* vy;
	double* vz;
	float TotalTime;
	cudaEvent_t start,stop;

};

//################### functions ####################
//reading parameters from input.dat
//allocating memory on gpu
void Input(Particles*);

//Initialization of initial configurations
void Initialization(Particles*);

//free memory and unbind
void exit(Particles*);

//printing results on file
void ConfFinal(Particles*);

//Periodic boundary condition
double Pbc(double);

template <typename T>
void Print(vector<T>,string);

double error(vector<double>,vector<double>,int);

//data analysis method
void data_blocking(int,vector<double>,double,string);

//last data from the analysis
vector<double> last_data_from_datablocking(int,vector<double>);

double mean_v(vector<double>,int,int = 0);

//data analysis
void data_blocking_MD(int);

void print_conf(Particles*);

void print_old_conf(Particles*);

void print_velocities(Particles*);

//useful if restart = 1, to do the first step 
void first_move(double*,double*,double*,double*,double*,double*,double*,double*,double*);

double Force_cpu(double*,double*,double*,int,int);

//################# Global functions ########################

__global__ void force_gpu(double*,double*,double*);

__global__ void verlet_gpu(double*,double*,double*,double*,double*,double*,double*,double*,double*);

//Periodic boundary condition
__device__ double Pbc_gpu(double); 

void Move_gpu(Particles*);
 
void print_device_properties();

__global__ void  measure_pot_virial(Lock lock,double*,double*,double*);

__global__ void  measure_kinetic(Lock lock,double*);

void Measure(Particles*);

//########################### IMPLEMENTAZIONI #################################

//atomicAdd but in double precision.. not avaible in this capability device of Cuda
#if __CUDA_ARCH__ < 600
__device__ double AtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
      } while (assumed != old);
            
             return __longlong_as_double(old);
   		}
#endif

//useful to understand allocating memory errors!
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


void Input(Particles* P){ //Prepare all stuff for the simulation

  cout << "using " << bl << " blocksPerGrid" << endl;
  cout << "using " << th_measure << " threadsPerBlock for measure properties"<< endl;
  cout << "using " << th_verlet << " threadsPerBlock for verlet algorithm"<< endl;


  cout << "Classic Lennard-Jones fluid        " << endl;
  cout << "Molecular dynamics simulation in NVE ensemble  " << endl << endl;
  cout << "Interatomic potential v(r) = 4 * [(1/r)^12 - (1/r)^6]" <<"\n"<<"\n";
  cout << "The program uses Lennard-Jones units " << endl;
  
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
  vol = (double)npart/rho;
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
 
  cudaSetDevice(1);

//allocate memory on global memory
 HANDLE_ERROR( cudaMalloc( (void**)&P->x,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->y,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->z,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->xold,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->yold,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->zold,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->vx,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->vy,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->vz,
                              npart*sizeof(double) ) );

//bind texture memory
 HANDLE_ERROR( cudaBindTexture( NULL, texx,
                                   P->x,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texy,
                                   P->y,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texz,
                                   P->z,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texxold,
                                   P->xold,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texyold,
                                   P->yold,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texzold,
                                   P->zold,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texvx,
                                   P->vx,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texvy,
                                   P->vy,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texvz,
                                   P->vz,
                                   npart*sizeof(double) ) );

 HANDLE_ERROR( cudaMalloc( (void**)&Fx,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&Fy,
                              npart*sizeof(double) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&Fz,
                              npart*sizeof(double) ) );

 HANDLE_ERROR( cudaBindTexture( NULL, texFx,
                                   Fx,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texFy,
                                   Fy,
                                   npart*sizeof(double) ) );
 HANDLE_ERROR( cudaBindTexture( NULL, texFz,
                                   Fz,
                                   npart*sizeof(double) ) );

//allocate global memory on the gpu
 HANDLE_ERROR( cudaMalloc( (void**)&dev_w, bl*sizeof(double)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_v, bl*sizeof(double)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_k, sizeof(double)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_hist, nbins*2*sizeof(double)  ) );

   return;
}


void Initialization(Particles* P) {

ReadConf.close();

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

double* x = new double[npart];
double* y = new double[npart];
double* z = new double[npart];
double* xold = new double[npart];
double* yold = new double[npart];
double* zold = new double[npart];
double* vx = new double[npart];
double* vy = new double[npart];
double* vz = new double[npart];
		
//Read initial configuration
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
	cout << endl;
  	ReadPrecedentConf.close();
	double sumv2=0.0,fs;
	first_move(x,y,z,xold,yold,zold,vx,vy,vz);
	for (int i=0; i<npart; ++i){
	vx[i] = Pbc(x[i] - xold[i])/(delta);
    	vy[i] = Pbc(y[i] - yold[i])/(delta);
    	vz[i] = Pbc(z[i] - zold[i])/(delta);
	sumv2 += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
	}
	cout << endl;
	sumv2 /= (double)npart;
	double T = sumv2/3.;//from the equipartion theorem
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
//copy configurations on the gpu
   HANDLE_ERROR( cudaMemcpy( P->x, x,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) ); 
   HANDLE_ERROR( cudaMemcpy( P->y, y,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy( P->z, z,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy( P->xold, xold,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) ); 
   HANDLE_ERROR( cudaMemcpy( P->yold, yold,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) );  
   HANDLE_ERROR( cudaMemcpy( P->zold, zold,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) );  
   HANDLE_ERROR( cudaMemcpy( P->vx, vx,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) ); 
   HANDLE_ERROR( cudaMemcpy( P->vy, vy,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) );  
   HANDLE_ERROR( cudaMemcpy( P->vz, vz,
                              npart*sizeof(double),
                              cudaMemcpyHostToDevice ) );
  
//free memory on host device
 delete [] x;
 delete [] y;
 delete [] z;
 delete [] xold;
 delete [] yold;
 delete [] zold;
 delete [] vx;
 delete [] vy;
 delete [] vz;

return;

}

void first_move(double* x,double*y,double*z,double*xold,double*yold,double*zold,double*vx,double* vy,double* vz){ //Move particles with Verlet algorithm
  double xnew, ynew, znew, fx[npart], fy[npart], fz[npart];

  for(int i=0; i<npart; ++i){ //Force acting on particle i
    fx[i] = Force_cpu(x,y,z,i,0);
    fy[i] = Force_cpu(x,y,z,i,1);
    fz[i] = Force_cpu(x,y,z,i,2);
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

double Force_cpu(double* x,double*y,double*z,int ip, int idir){ //Compute forces as -Grad_ip V(r)
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

//unbind and free memory
void exit(Particles* P) {

    HANDLE_ERROR(cudaUnbindTexture( texx ) );
    HANDLE_ERROR(cudaUnbindTexture( texy ) );
    HANDLE_ERROR(cudaUnbindTexture( texz ) );
    HANDLE_ERROR(cudaUnbindTexture( texxold ) );
    HANDLE_ERROR(cudaUnbindTexture( texyold ) );
    HANDLE_ERROR(cudaUnbindTexture( texzold ) );
    HANDLE_ERROR(cudaUnbindTexture( texvx ) );
    HANDLE_ERROR(cudaUnbindTexture( texvy ) );
    HANDLE_ERROR(cudaUnbindTexture( texvz ) );
    HANDLE_ERROR(cudaUnbindTexture( texFx ) );
    HANDLE_ERROR(cudaUnbindTexture( texFy ) );
    HANDLE_ERROR(cudaUnbindTexture( texFz ) );




    HANDLE_ERROR( cudaFree( P->x ) );
    HANDLE_ERROR( cudaFree( P->y ) );
    HANDLE_ERROR( cudaFree( P->z ) );
    HANDLE_ERROR( cudaFree( P->xold ) );
    HANDLE_ERROR( cudaFree( P->yold ) );
    HANDLE_ERROR( cudaFree( P->zold ) );
    HANDLE_ERROR( cudaFree( P->vx ) );
    HANDLE_ERROR( cudaFree( P->vy ) );
    HANDLE_ERROR( cudaFree( P->vz ) );
    HANDLE_ERROR( cudaFree( Fx ) );
    HANDLE_ERROR( cudaFree( Fy ) );
    HANDLE_ERROR( cudaFree( Fz ) );




   HANDLE_ERROR(cudaFree(dev_w));
   HANDLE_ERROR(cudaFree(dev_v));
   HANDLE_ERROR(cudaFree(dev_k));
   HANDLE_ERROR(cudaFree(dev_hist));

 
}

void Move_gpu(Particles *P) {

 double* fx,*fy,*fz;
 fx = Fx;
 fy = Fy;
 fz = Fz;

 force_gpu<<<bl,th_verlet>>>(fx,fy,fz);

 double*xold,*yold,*zold,*x,*y,*z,*vx,*vy,*vz;
 xold = P->xold;
 yold = P->yold;
 zold = P->zold;
 x = P->x;
 y = P->y;
 z = P->z;
 vx = P->vx;
 vy = P->vy;
 vz = P->vz;

 verlet_gpu<<<bl,th_kinetic>>>(xold,yold,zold,x,y,z,vx,vy,vz);

 //cudaDeviceSynchronize();

}

//method of atom decomposition ---> each block calculate force for its own particle (ip = blockIdx.x)
// threads in each block calculate the corresponding force acting on the block-particle
__global__ void force_gpu(double*fx,double*fy,double*fz){

	__shared__ double f0[th_verlet];
	__shared__ double f1[th_verlet];
	__shared__ double f2[th_verlet];
	double temp0,temp1,temp2;
	double dvec[3];
	double dr;
	int tid;
	int cacheIndex = threadIdx.x;
	int ip = blockIdx.x;
	while (ip < gpu_npart ) {
		tid = threadIdx.x;
		temp0 =0;
		temp1 =0;
		temp2 =0;
		while( tid<gpu_npart ) {
			dvec[0] = Pbc_gpu(fetch_double(texx,ip)-fetch_double(texx,tid)); 
			dvec[1] = Pbc_gpu(fetch_double(texy,ip)-fetch_double(texy,tid)); 
			dvec[2] = Pbc_gpu(fetch_double(texz,ip)-fetch_double(texz,tid));
			dr=sqrt(dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2]);
			if (dr<gpu_rcut && dr>0) {
				temp0 += dvec[0] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); 
				temp1 += dvec[1] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); 
				temp2 += dvec[2] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); 
			}
			tid += blockDim.x;
		}
		f0[cacheIndex] = temp0;		
		f1[cacheIndex] = temp1;	
		f2[cacheIndex] = temp2;	
		__syncthreads();
		tid = blockDim.x/2;
		while (tid !=0) {
			if (cacheIndex < tid) {
				f0[cacheIndex] += f0[cacheIndex+tid];
				f1[cacheIndex] += f1[cacheIndex+tid];
				f2[cacheIndex] += f2[cacheIndex+tid];


			}
			__syncthreads();
			tid /= 2;
		}
	
	if (cacheIndex == 0) {

    		fx[ip] = f0[0];
		fy[ip] = f1[0];
		fz[ip] = f2[0];
		}
	
		ip += gridDim.x;
	}		
}

//aggiornamento delle posizioni tramite verlet
__global__ void verlet_gpu(double*xold,double*yold,double*zold,double*x,double*y,double*z,double*vx,double*vy,double*vz){

	double xnew,ynew,znew;
	int ip = threadIdx.x+blockDim.x*blockIdx.x;

	while (ip < gpu_npart ) {

		xnew = Pbc_gpu( 2.0 * fetch_double(texx,ip) - fetch_double(texxold,ip) +  fetch_double(texFx,ip)*gpu_delta*gpu_delta);
		ynew = Pbc_gpu( 2.0 * fetch_double(texy,ip) - fetch_double(texyold,ip) +  fetch_double(texFy,ip)*gpu_delta*gpu_delta);
		znew = Pbc_gpu( 2.0 * fetch_double(texz,ip) - fetch_double(texzold,ip) +  fetch_double(texFz,ip)*gpu_delta*gpu_delta);

		vx[ip] = Pbc_gpu(xnew - fetch_double(texxold,ip)) / (2.0 * gpu_delta);
    		vy[ip] = Pbc_gpu(ynew - fetch_double(texyold,ip)) / (2.0 * gpu_delta);
    		vz[ip] = Pbc_gpu(znew - fetch_double(texzold,ip)) / (2.0 * gpu_delta);

    		xold[ip] = fetch_double(texx,ip);
    		yold[ip] = fetch_double(texy,ip);
    		zold[ip] = fetch_double(texz,ip);

    		x[ip] = xnew;
   		y[ip] = ynew;
    		z[ip] = znew;
	
		ip += gridDim.x*blockDim.x;
	}		
}


__global__ void  measure_kinetic(Lock lock,double *k) {

//it's simply a scalar product
	__shared__ double kin[th_kinetic]; 
	double a=0,b=0,c=0; 
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	while (i<gpu_npart) {
		a += fetch_double(texvx,i)*fetch_double(texvx,i);
		b += fetch_double(texvy,i)*fetch_double(texvy,i);
		c += fetch_double(texvz,i)*fetch_double(texvz,i);
		i += blockDim.x*gridDim.x;
	}
	int cacheIndex = threadIdx.x;
        kin[cacheIndex] = a+b+c;
	__syncthreads();
	int tid = blockDim.x/2.;
	while(tid !=0) {
		if(cacheIndex<tid) kin[cacheIndex]+=kin[cacheIndex+tid];
	__syncthreads();
	tid /= 2;
	}
	if(cacheIndex==0) {
		lock.lock();//to let work just one thread per block and in sequence
		*k += kin[0];
		lock.unlock();
	}

}

__global__ void  measure_properties(Lock lock,double *v,double *w,double* hist) {

	__shared__ double pot[th_measure];
	__shared__ double vir[th_measure];
	double temp0=0;
	double temp1=0;
	double dr;
	int tid;
	int bin;
	int cacheIndex = threadIdx.x;
	int i = blockIdx.x;
	int j;
	while (i<gpu_npart-1) {
		j = i+1+threadIdx.x;//take the consecutive, not to count twice the same
				     //contribute
		__syncthreads();
		while (j<gpu_npart) {
			dr = sqrt( Pbc_gpu(fetch_double(texx,i)-fetch_double(texx,j))*Pbc_gpu(fetch_double(texx,i)-fetch_double(texx,j))+
				   Pbc_gpu(fetch_double(texy,i)-fetch_double(texy,j))*Pbc_gpu(fetch_double(texy,i)-fetch_double(texy,j))+
				   Pbc_gpu(fetch_double(texz,i)-fetch_double(texz,j))*Pbc_gpu(fetch_double(texz,i)-fetch_double(texz,j))  );

			bin = int(dr/gpu_binsize);
			//filling the histogram
			AtomicAdd( &hist[bin],2);//this is necessary, to not overwrite counts
                        if ( dr<gpu_rcut ) {
				temp0 += 4.0/pow(dr,12) - 4.0/pow(dr,6);
                                temp1 += 16.0/pow(dr,12) - 8.0/pow(dr,6);
			}
			j+=blockDim.x;
		}
	i+=gridDim.x;
	}
	pot[cacheIndex] = temp0;
	vir[cacheIndex] = temp1;
	__syncthreads();
	tid = blockDim.x/2.;
	while (tid!=0) {
		if(cacheIndex<tid) {
			pot[cacheIndex] += pot[cacheIndex+tid];
			vir[cacheIndex] += vir[cacheIndex+tid];
		}
	__syncthreads();
	tid /= 2.;
	}
	if (cacheIndex == 0) {
		v[blockIdx.x] = pot[0];
		w[blockIdx.x] = vir[0];
	}

}

void Measure(Particles* P) {

 Lock lock;

	//setting all variables on gpu side equals to zero
	HANDLE_ERROR( cudaMemset(dev_hist,0, nbins*2*sizeof(double)  ) );
	HANDLE_ERROR( cudaMemset(dev_k,0, sizeof(double)  ) );

	measure_properties<<<bl,th_measure>>>(lock,dev_v,dev_w,dev_hist);
	measure_kinetic<<<bl,th_kinetic>>>(lock,dev_k);
	//cudaDeviceSynchronize();

	//copy on cpu-side, to do the data analysis
	double v[bl];
	double w[bl];
	double k;
	double hist[nbins*2];
	double deltaVr;

        HANDLE_ERROR( cudaMemcpy(hist,dev_hist,nbins*2*sizeof(double),cudaMemcpyDeviceToHost) );
	for (int i=0;i<nbins;i++) {
	    deltaVr = rho*npart*4.*M_PI/3.*(pow((i+1)*bin_size,3)-pow((i)*bin_size,3));
	   gdir[i].push_back(hist[i]/deltaVr);
        }

	HANDLE_ERROR( cudaMemcpy(w,dev_w,bl*sizeof(double),cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(v,dev_v,bl*sizeof(double),cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&k,dev_k,sizeof(double),cudaMemcpyDeviceToHost) );

	double total_v = 0;
	double total_w = 0;
	for (int k=0;k<bl;k++){
			total_w += w[k];  
			total_v += v[k];	
	}
	
	double t = 0.5*k;

        stima_pot_gpu = total_v/(double)npart+vtail; //Potential energy per particle
        stima_kin_gpu = t/(double)npart; //Kinetic energy per particle
        stima_temp_gpu = (2.0 / 3.0) *t/(double)npart; //Temperature
        stima_etot_gpu = (t+total_v)/(double)npart+vtail; //Total energy per particle
        stima_press_gpu = rho*stima_temp_gpu+ (total_w + ptail*(double)npart ) / vol;

        properties_gpu[0].push_back(stima_pot_gpu);
        properties_gpu[1].push_back(stima_kin_gpu);
        properties_gpu[2].push_back(stima_temp_gpu);
        properties_gpu[3].push_back(stima_etot_gpu);
        properties_gpu[4].push_back(stima_press_gpu);


}

void print_properties() {

  string name = "output_epot"+ to_string(nstep)+".dat";
  Print(properties_gpu[0],name);
  name = "output_ekin"+ to_string(nstep)+".dat";
  Print(properties_gpu[1],name);
  name = "output_temp"+ to_string(nstep)+".dat";
  Print(properties_gpu[2],name);
  name = "output_etot"+ to_string(nstep)+".dat";
  Print(properties_gpu[3],name);
  name = "output_press"+ to_string(nstep)+".dat";
  Print(properties_gpu[4],name);
}

template <typename T>
void Print(vector<T> v, string name) {
   ofstream fd; fd.open(name,ios::app);
   for (auto& el : v) fd << el << endl;
   fd.close();
}


void print_velocities(Particles* P) {

double * vx = new double [npart];
double * vy = new double [npart];
double * vz = new double [npart];

  HANDLE_ERROR( cudaMemcpy(vx,P->vx,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(vy,P->vy,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(vz,P->vz,npart*sizeof(double),cudaMemcpyDeviceToHost));

cout<< "\n";
 cout << "Print actual velocities:" << endl;

 for (int i=0; i<npart; ++i){
    cout << vx[i] << "   " <<  vy[i] << "   " << vz[i] << endl;
 }

cout<< "\n";
 

delete [] vx;
delete [] vy;
delete [] vz;
}



void print_old_conf(Particles* P) {
  double * xold = new double[npart];
  double * yold = new double[npart];
  double * zold = new double[npart];

  HANDLE_ERROR( cudaMemcpy(xold,P->xold,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(yold,P->yold,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(zold,P->zold,npart*sizeof(double),cudaMemcpyDeviceToHost));

cout<< "\n";
 cout << "Print actual old configuration" << endl;

 for (int i=0; i<npart; ++i){
    cout << xold[i]/box << "   " <<  yold[i]/box << "   " << zold[i]/box << endl;
 }

cout<< "\n";
  delete [] xold;
  delete [] yold;
  delete [] zold;
 
  return;


}

void print_conf(Particles* P) {

  double * x = new double[npart];
  double * y = new double[npart];
  double * z = new double[npart];

  HANDLE_ERROR( cudaMemcpy(x,P->x,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(y,P->y,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(z,P->z,npart*sizeof(double),cudaMemcpyDeviceToHost));
 cout<< "\n";
 cout << "Print actual configuration" << endl;
 for (int i=0; i<npart; ++i){
    cout << x[i]/box << "   " <<  y[i]/box << "   " << z[i]/box << endl;
 }

cout <<"\n";

  delete [] x;
  delete [] y;
  delete [] z;
 
  return;

}

void ConfFinal(Particles*P){ //Write final configuration

  double * xold = new double[npart];
  double * yold = new double[npart];
  double * zold = new double[npart];

  HANDLE_ERROR( cudaMemcpy(xold,P->xold,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(yold,P->yold,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(zold,P->zold,npart*sizeof(double),cudaMemcpyDeviceToHost));

  ofstream WriteOldConf("config.final",ios::out);
  cout << "Print penultimate configuration in config.final " << endl << endl;
  for (int i=0; i<npart; ++i){
    WriteOldConf << xold[i]/box << "   " <<  yold[i]/box << "   " << zold[i]/box << endl;
  } 
  WriteOldConf.close();
 
  delete [] xold;
  delete [] yold;
  delete [] zold;

  double * x = new double[npart];
  double * y = new double[npart];
  double * z = new double[npart];

  HANDLE_ERROR( cudaMemcpy(x,P->x,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(y,P->y,npart*sizeof(double),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(z,P->z,npart*sizeof(double),cudaMemcpyDeviceToHost));


  ofstream WriteConf("config.0",ios::out);
  
  cout << "Print final configuration to file config.0 " << endl << endl;
  WriteConf << npart << endl;
  for (int i=0; i<npart; ++i){
    WriteConf << x[i]/box << "   " <<  y[i]/box << "   " << z[i]/box << endl;
  }
  WriteConf.close();

  delete [] x;
  delete [] y;
  delete [] z;

}

//cpu-side
double Pbc(double r) {  //Algorithm for periodic boundary conditions with side L=box
    return r - box * rintf(r/box);
}
//gpu-side
__device__ double Pbc_gpu(double r) {  //Algorithm for periodic boundary conditions with side L=box
    return r - gpu_box * rintf(r/gpu_box);
}

void data_blocking_MD(int N) {

int L = (nstep/10.)/N; //cause I measure properties each 10 steps
vector<string> names = {"ave_epot","ave_ekin","ave_temp","ave_etot","ave_press"};
int j=0;
vector<double> v_mean;
vector<double> data(2);
 for (auto & el : names) {
	for (int i=0;i<N;i++) 
		 v_mean.push_back( mean_v(properties_gpu[j], (i+1)*L, i*L ));
	 if ( j== 2) data = last_data_from_datablocking(N,v_mean);
	 data_blocking(N,v_mean,0,el+to_string(nstep)+".out");
	 properties_gpu[j].clear();
	 j++;
	 v_mean.clear();
 }
 //must be within two sigma
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

         ofstream fd;
         fd.open(file,ios::out);
         for (int i=0; i<N;i++) fd << sum_prog[i]-real_value<<" "<< err_prog[i] << endl;
         fd.close();

};

double error(vector<double> AV, vector<double> AV2, int i) {
        if (i==0) return 0;
        else return sqrt( (AV2[i]-AV[i]*AV[i]) /(double)i );
};

double mean_v(vector<double> v,int last_index, int first_index) {
	double sum = 0;
	for (int i=first_index; i<last_index; i++) sum += v[i];
        return sum/(last_index-first_index);
}; 


