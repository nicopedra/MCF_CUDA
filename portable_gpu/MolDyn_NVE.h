#include <stdlib.h>     // srand, rand: to generate random number
#include <iostream>     // cin, cout: Standard Input/Output Streams Library
#include <fstream>      // Stream class to both read and write from/to files.
#include <cmath>        // rint, pow
#include <vector>
#include <cstdio>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "lock.h"

#define bl 512//512
#define th_verlet 128 //128
#define th_measure 1024 //1024
#define th_kinetic 1024 //1024
using namespace std;
//for initialization
ifstream ReadInput,ReadConf,ReadPrecedentConf;
string file_start;

//parameters, observables
const int m_props=5;
const int nbins=100;
int n_props;
//tail correction due to rcut
float vtail;
float ptail;
//mean temp
float m_temp;
float accettazione;
// gdir and other properties
float bin_size;
vector<vector<float>> gdir(nbins);
float stima_pot_gpu, stima_kin_gpu, stima_etot_gpu, stima_temp_gpu,stima_press_gpu;
vector<vector<float>> properties_gpu(m_props);
//restart parameter
int restart;
//############### dimension for a specific molecule #############

// thermodynamical state
int npart;
float energy,temp,vol,rho,box,rcut;

//su gpu
float *dev_w,*dev_v,*dev_k; //per calcolare le energie in measure
float* dev_hist;//per la g(r)

// simulation
int nstep, iprint, seed;
float delta;

//position and force action on particles
float *x,*y,*z,*xold,*yold,*zold,*vx,*vy,*vz;
float* fx,*fy,*fz;

//structures
//questa struttura Particelle contiene tutte le variabili relative alle particelle
struct Particles {
	
	int deviceID;
        int size;
        int offset;

	float* x;
	float* y;
	float* z;
	float* xold;
	float* yold;
	float* zold;
	float* vx;
	float* vy;
	float* vz;
	float* fx;
	float* fy;
	float* fz;

	float TotalTime;
	cudaEvent_t start,stop;

};


__constant__ int gpu_npart;
__constant__ float gpu_binsize;
__constant__ float gpu_rcut;
__constant__ float gpu_delta;
__constant__ float gpu_box;


//################### functions ####################

void Input(Particles*);

void Initialization(Particles*);

//free memory
void exit(Particles*);

void ConfFinal(Particles*);

//void ConfXYZ(int);
float Pbc(float);

template <typename T>
void Print(vector<T>,string);

float error(vector<float>,vector<float>,int);

void data_blocking(int,vector<float>,float,string);

vector<float> last_data_from_datablocking(int,vector<float>);

float mean_v(vector<float>,int,int = 0);

void data_blocking_MD(int);

void print_conf(Particles*);

void print_old_conf(Particles*);
 
void first_move(float*,float*,float*,float*,float*,float*,float*,float*,float*);

float Force_cpu(float*,float*,float*,int,int);

//################# FUNZIONI GLOBAL ########################

__global__ void force_gpu(float*,float*,float*,float*,float*,float*);

__global__ void verlet_gpu(float*,float*,float*,float*,float*,float*,float*,float*,float*,float*,float*,float*);

__device__ float Pbc_gpu(float); 

void Move_gpu(Particles*);
 
void print_device_properties();

__global__ void  measure_properties(Lock lock,float*,float*,float*,float*,float*,float*);

__global__ void  measure_kinetic(Lock lock,float*,float*,float*,float*);

void print_velocities(Particles*);

void Measure(Particles*);

//########################### IMPLEMENTAZIONI #################################

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

//##################### INPUT #########################################

void Input(Particles* P){ //Prepare all stuff for the simulation

  cout << "using " << bl << " blocksPerGrid" << endl;
  cout << "using " << th_measure << " threadsPerBlock for measure properties"<< endl;
  cout << "using " << th_verlet << " threadsPerBlock for verlet algorithm"<< endl;
  
  cout << endl;


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

//allocate memory on host side
    HANDLE_ERROR( cudaHostAlloc( (void**)&x,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&y,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&z,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&xold,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&yold,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&zold,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&vx,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&vy,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
   HANDLE_ERROR( cudaHostAlloc( (void**)&vz,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
//pointing to that memory! unified memory
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->x, x, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->y, y, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->z, z, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->xold, xold, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->yold, yold, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->zold, zold, 0 ) );		
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->vx, vx, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->vy, vy, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer( &P->vz, vz, 0 ) );

//same as above, forces
 HANDLE_ERROR( cudaHostAlloc( (void**)&fx,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
 HANDLE_ERROR( cudaHostAlloc( (void**)&fy,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );
 HANDLE_ERROR( cudaHostAlloc( (void**)&fz,
                              npart*sizeof(float),
                              cudaHostAllocWriteCombined |
                                     cudaHostAllocMapped ) );

 HANDLE_ERROR( cudaHostGetDevicePointer(&P->fx, fx, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer(&P->fy, fy, 0 ) );
 HANDLE_ERROR( cudaHostGetDevicePointer(&P->fz, fz, 0 ) );


//allocate global memory on the gpu
 HANDLE_ERROR( cudaMalloc( (void**)&dev_w, bl*sizeof(float)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_v, bl*sizeof(float)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_k, sizeof(float)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_hist, nbins*2*sizeof(float)  ) );

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
	float sumv2=0.0,fs;
	first_move(x,y,z,xold,yold,zold,vx,vy,vz);
	for (int i=0; i<npart; ++i){
	vx[i] = Pbc(x[i] - xold[i])/(delta);
    	vy[i] = Pbc(y[i] - yold[i])/(delta);
    	vz[i] = Pbc(z[i] - zold[i])/(delta);
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
   float sumv[3] = {0.0, 0.0, 0.0};
   for (int i=0; i<npart; ++i){
     vx[i] = rand()/float(RAND_MAX) - 0.5; //centrate in 0
     vy[i] = rand()/float(RAND_MAX) - 0.5;
     vz[i] = rand()/float(RAND_MAX) - 0.5;

     sumv[0] += vx[i];
     sumv[1] += vy[i];
     sumv[2] += vz[i];
   } //servono per calcolare la posizione al tempo precedente rispetto a quello iniziale
   for (int idim=0; idim<3; ++idim) sumv[idim] /= (float)npart;
   float sumv2 = 0.0, fs;
   for (int i=0; i<npart; ++i){ //così evito drift rispetto al centro di massa del sistema
     vx[i] = vx[i] - sumv[0];
     vy[i] = vy[i] - sumv[1];
     vz[i] = vz[i] - sumv[2];

     sumv2 += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i];
   }
   sumv2 /= (float)npart;
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

return;

}

void first_move(float* x,float*y,float*z,float*xold,float*yold,float*zold,float*vx,float* vy,float* vz){ //Move particles with Verlet algorithm
  float xnew, ynew, znew, fx[npart], fy[npart], fz[npart];

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

float Force_cpu(float* x,float*y,float*z,int ip, int idir){ //Compute forces as -Grad_ip V(r)
  float f=0.0;
  float dvec[3], dr;

  for (int i=0; i<npart; ++i){
    if(i != ip){
      dvec[0] = Pbc( x[ip] - x[i] );  
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

//chiamo alla fine per liberare la memoria sul device e Host
void exit(Particles* P) {

    HANDLE_ERROR( cudaFreeHost( x ) );
    HANDLE_ERROR( cudaFreeHost( y ) );
    HANDLE_ERROR( cudaFreeHost( z ) );
    HANDLE_ERROR( cudaFreeHost( xold ) );
    HANDLE_ERROR( cudaFreeHost( yold ) );
    HANDLE_ERROR( cudaFreeHost( zold ) );
    HANDLE_ERROR( cudaFreeHost( vx ) );
    HANDLE_ERROR( cudaFreeHost( vy ) );
    HANDLE_ERROR( cudaFreeHost( vz ) );
    HANDLE_ERROR( cudaFreeHost( fx ) );
    HANDLE_ERROR( cudaFreeHost( fy ) );
    HANDLE_ERROR( cudaFreeHost( fz ) );

   cudaFree(dev_w);
   cudaFree(dev_v);
   cudaFree(dev_k);
   cudaFree(dev_hist);

}

void Move_gpu(Particles *P) {

//forces
 force_gpu<<<bl,th_verlet>>>(P->fx,P->fy,P->fz,P->x,P->y,P->z);
//moving particles with Verlet algorithm
 verlet_gpu<<<bl,th_kinetic/2>>>(P->xold,P->yold,P->zold,P->x,P->y,P->z,P->vx,P->vy,P->vz,P->fx,P->fy,P->fz);

}

//method of atom decomposition ---> each block calculate force for its own particle (ip = blockIdx.x)
// threads in each block calculate the corresponding force acting on the block-particle
__global__ void force_gpu(float*fx,float*fy,float*fz,float*x,float*y,float*z){

	__shared__ float f0[th_verlet];
	__shared__ float f1[th_verlet];
	__shared__ float f2[th_verlet];
	float temp0,temp1,temp2;
	float dvec[3];
	float dr;
	int tid;
	int cacheIndex = threadIdx.x;
	int ip = blockIdx.x;
	while (ip < gpu_npart ) {
		tid = threadIdx.x;
		temp0 =0;
		temp1 =0;
		temp2 =0;
		while( tid<gpu_npart ) {
			dvec[0] = Pbc_gpu(x[ip]-x[tid]); 
			dvec[1] = Pbc_gpu(y[ip]-y[tid]); 
			dvec[2] = Pbc_gpu(z[ip]-z[tid]);
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
__global__ void verlet_gpu(float*xold,float*yold,float*zold,float*x,float*y,float*z,float*vx,float*vy,float*vz,float*fx,float*fy,float*fz){

	float xnew,ynew,znew;
	int ip = threadIdx.x+blockDim.x*blockIdx.x;

	while (ip < gpu_npart ) {

		xnew = Pbc_gpu( 2.0 * x[ip] - xold[ip] + fx[ip]*gpu_delta*gpu_delta);
		ynew = Pbc_gpu( 2.0 * y[ip] - yold[ip] + fy[ip]*gpu_delta*gpu_delta);
		znew = Pbc_gpu( 2.0 * z[ip] - zold[ip] + fz[ip]*gpu_delta*gpu_delta);

		vx[ip] = Pbc_gpu(xnew - xold[ip]) / (2.0 * gpu_delta);
    		vy[ip] = Pbc_gpu(ynew - yold[ip]) / (2.0 * gpu_delta);
    		vz[ip] = Pbc_gpu(znew - zold[ip]) / (2.0 * gpu_delta);

    		xold[ip] = x[ip];
    		yold[ip] = y[ip];
    		zold[ip] = z[ip];

    		x[ip] = xnew;
   		y[ip] = ynew;
    		z[ip] = znew;
	
		ip += gridDim.x*blockDim.x;
	}		
}

__global__ void  measure_kinetic(Lock lock,float *k,float*vx,float*vy,float*vz) {

//it's simply a scalar product
	__shared__ float kin[th_kinetic]; 
	float a=0,b=0,c=0; 
	int i = threadIdx.x+blockIdx.x*blockDim.x;
	while (i<gpu_npart) {
		a += vx[i]*vx[i];
		b += vy[i]*vy[i];
		c += vz[i]*vz[i];
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

__global__ void  measure_properties(Lock lock,float *v,float *w,float* hist,float*x,float*y,float*z) {

	__shared__ float pot[th_measure];
	__shared__ float vir[th_measure];
	float temp0=0;
	float temp1=0;
	float dr;
	int bin;
	int cacheIndex = threadIdx.x;
        int tid;
	int i = blockIdx.x;
	int j;
	while (i<gpu_npart-1) {
		j = i+1+threadIdx.x; //take the consecutive, not to count twice the same
				     //contribute
		__syncthreads();
		while (j<gpu_npart) {
			dr = sqrt( Pbc_gpu(x[i]-x[j])*Pbc_gpu(x[i]-x[j])+
				   Pbc_gpu(y[i]-y[j])*Pbc_gpu(y[i]-y[j])+
				   Pbc_gpu(z[i]-z[j])*Pbc_gpu(z[i]-z[j]));

			bin = int(dr/gpu_binsize);
			//filling the histogram
			atomicAdd( &hist[bin],2);//this is necessary, to not overwrite counts
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
	HANDLE_ERROR( cudaMemset(dev_hist,0, nbins*2*sizeof(float)  ) );
	HANDLE_ERROR( cudaMemset(dev_k,0, sizeof(float)  ) );

	measure_properties<<<bl,th_measure>>>(lock,dev_v,dev_w,dev_hist,x,y,z);
	measure_kinetic<<<bl,th_kinetic>>>(lock,dev_k,vx,vy,vz);
	//cudaDeviceSynchronize();

	//copy on cpu-side, to do the data analysis
	float v[bl];
	float w[bl];
	float k;
	float hist[nbins*2];
	float deltaVr;

        HANDLE_ERROR( cudaMemcpy(hist,dev_hist,nbins*2*sizeof(float),cudaMemcpyDeviceToHost) );
	for (int i=0;i<nbins;i++) {
	    deltaVr = rho*npart*4.*M_PI/3.*(pow((i+1)*bin_size,3)-pow((i)*bin_size,3));
	   gdir[i].push_back(hist[i]/deltaVr);
        }

	HANDLE_ERROR( cudaMemcpy(w,dev_w,bl*sizeof(float),cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(v,dev_v,bl*sizeof(float),cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&k,dev_k,sizeof(float),cudaMemcpyDeviceToHost) );

	float total_v = 0;
	float total_w = 0;
	for (int k=0;k<bl;k++){
			total_w += w[k];  
			total_v += v[k];	
	}
	
	float t = 0.5*k;

        stima_pot_gpu = total_v/(float)npart+vtail; //Potential energy per particle
        stima_kin_gpu = t/(float)npart; //Kinetic energy per particle
        stima_temp_gpu = (2.0 / 3.0) *t/(float)npart; //Temperature
        stima_etot_gpu = (t+total_v)/(float)npart+vtail; //Total energy per particle
        stima_press_gpu = rho*stima_temp_gpu+ (total_w + ptail*(float)npart ) / vol;

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

float * Vx = new float [npart];
float * Vy = new float [npart];
float * Vz = new float [npart];

  HANDLE_ERROR( cudaMemcpy(Vx,P->vx,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Vy,P->vy,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Vz,P->vz,npart*sizeof(float),cudaMemcpyDeviceToHost));

cout<< "\n";
 cout << "Print actual velocities:" << endl;

 for (int i=0; i<npart; ++i){
    cout << Vx[i] << "   " <<  Vy[i] << "   " << Vz[i] << endl;
 }

cout<< "\n";
 

delete [] Vx;
delete [] Vy;
delete [] Vz;
}



void print_old_conf(Particles* P) {
  float * Xold = new float[npart];
  float * Yold = new float[npart];
  float * Zold = new float[npart];

  HANDLE_ERROR( cudaMemcpy(Xold,P->xold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Yold,P->yold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Zold,P->zold,npart*sizeof(float),cudaMemcpyDeviceToHost));

cout<< "\n";
 cout << "Print actual old configuration" << endl;

 for (int i=0; i<npart; ++i){
    cout << Xold[i]/box << "   " <<  Yold[i]/box << "   " << Zold[i]/box << endl;
 }

cout<< "\n";
  delete [] Xold;
  delete [] Yold;
  delete [] Zold;
 



}

void print_conf(Particles* P) {

  float * X = new float[npart];
  float * Y = new float[npart];
  float * Z = new float[npart];

  HANDLE_ERROR( cudaMemcpy(X,P->x,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Y,P->y,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Z,P->z,npart*sizeof(float),cudaMemcpyDeviceToHost));
 cout<< "\n";
 cout << "Print actual configuration" << endl;
 for (int i=0; i<npart; ++i){
    cout << X[i]/box << "   " <<  Y[i]/box << "   " << Z[i]/box << endl;
 }

cout <<"\n";

  delete [] X;
  delete [] Y;
  delete [] Z;
 
  return;

}

void ConfFinal(Particles*P){ //Write final configuration, per ricominciare dalla simulazione precedente

  float * Xold = new float[npart];
  float * Yold = new float[npart];
  float * Zold = new float[npart];

  HANDLE_ERROR( cudaMemcpy(Xold,P->xold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Yold,P->yold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Zold,P->zold,npart*sizeof(float),cudaMemcpyDeviceToHost));

  ofstream WriteOldConf("config.final",ios::out);
  cout << "Print penultimate configuration in config.final " << endl << endl;
  for (int i=0; i<npart; ++i){
    WriteOldConf << Xold[i]/box << "   " <<  Yold[i]/box << "   " << Zold[i]/box << endl;
  } 
  WriteOldConf.close();
 
  delete [] Xold;
  delete [] Yold;
  delete [] Zold;

  float * X = new float[npart];
  float * Y = new float[npart];
  float * Z = new float[npart];

  HANDLE_ERROR( cudaMemcpy(X,P->x,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Y,P->y,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(Z,P->z,npart*sizeof(float),cudaMemcpyDeviceToHost));


  ofstream WriteConf("config.0",ios::out);
  
  cout << "Print final configuration to file config.0 " << endl << endl;
  WriteConf << npart << endl;
  for (int i=0; i<npart; ++i){
    WriteConf << X[i]/box << "   " <<  Y[i]/box << "   " << Z[i]/box << endl;
  }
  WriteConf.close();

  delete [] X;
  delete [] Y;
  delete [] Z;

}


float Pbc(float r) {  //Algorithm for periodic boundary conditions with side L=box
    return r - box * rintf(r/box);
}

__device__ float Pbc_gpu(float r) {  //Algorithm for periodic boundary conditions with side L=box
    return r - gpu_box * rintf(r/gpu_box);
}

void data_blocking_MD(int N) {

int L = (nstep/10.)/N; //cause I measure properties each 10 steps
vector<string> names = {"ave_epot","ave_ekin","ave_temp","ave_etot","ave_press"};
int j=0;
vector<float> v_mean;
vector<float> data(2);
 for (auto & el : names) {
	for (int i=0;i<N;i++) 
		 v_mean.push_back( mean_v(properties_gpu[j], (i+1)*L, i*L ));
	 if ( j== 2) data = last_data_from_datablocking(N,v_mean);
	 data_blocking(N,v_mean,0,el+to_string(nstep)+".out");
	 properties_gpu[j].clear();
	 j++;
	 v_mean.clear();
 }

 accettazione = data[1];
 m_temp = data[0];
 //fatta l'analisi dati controllo a che temperatura si trova il sistema
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


