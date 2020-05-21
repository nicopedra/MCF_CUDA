#include <stdlib.h>     // srand, rand: to generate random number
#include <iostream>     // cin, cout: Standard Input/Output Streams Library
#include <fstream>      // Stream class to both read and write from/to files.
#include <cmath>        // rint, pow
#include <vector>
#include <cstdio>
//#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "lock.h"

#define th 128
#define bl 64

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
float stima_pot_gpu, stima_kin_gpu, stima_etot_gpu, stima_temp_gpu,stima_press_gpu;
vector<vector<float>> properties_gpu(m_props);

int restart;
//############### dimension for a specific molecule #############

// thermodynamical state
int npart;
float energy,temp,vol,rho,box,rcut;

float *dev_w,*dev_v,*dev_k; //per calcolare le energie in measure
float* dev_hist;//per la g(r)

// simulation
int nstep, iprint, seed;
float delta;

//structures
//questa struttura Particelle contiene tutte le variabili relative alle particelle
struct Particles {

	float* dev_x;
	float* dev_y;
	float* dev_z;
	float* dev_xold;
	float* dev_yold;
	float* dev_zold;
	float* dev_vx;
	float* dev_vy;
	float* dev_vz;
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

void exit(Particles*);

void ConfFinal(Particles*);

//void ConfXYZ(int);
float Pbc(float);

void print_properties();

template <typename T>
void Print(vector<T>,string);

float error(vector<float>,vector<float>,int);

void data_blocking(int,vector<float>,float,string);

vector<float> last_data_from_datablocking(int,vector<float>);

float mean_v(vector<float>,int,int = 0);

void data_blocking_MD(int);

void print_conf(Particles*);

void print_old_conf(Particles*);
 
void first_move(float*,float*,float*,float*,float*,float*,float*,float*,float*);//primo step, se rileggo configurazioni precedenti 

//################# FUNZIONI GLOBAL ########################

__global__ void verlet_gpu(float*,float*,float*,float*,float*,float*,float*,float*,float*);

__device__ float Pbc_gpu(float); 

void Move_gpu(Particles*);
 
void print_device_properties();

__global__ void  measure_pot_virial(Lock lock,float*,float*,float*,float*,float*,float*);
__global__ void  measure_kinetic(Lock lock,float*,float*,float*,float*);

void print_velocities(Particles*);

void Measure(Particles*);

__global__ void prova_gpu (float*,float*,float);//per testare la Pbc_gpu 

//########################### IMPLEMENTAZIONI #################################

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

template< typename T >
void swap( T& a, T& b ) {
    T t = a;
    a = b;
    b = t;
}


void print_device_properties() {

 int count;
 cudaDeviceProp prop;
 HANDLE_ERROR(cudaGetDeviceCount(&count));
 cout <<"how many devices: "<< count << endl;

 for (int i=0;i<count;i++) {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop,i) );
        cout <<"name: "<< prop.name << endl;
        cout <<"total global memory: "<< prop.totalGlobalMem <<"bytes \n";
        cout <<"memory per block: "<< prop.sharedMemPerBlock <<"bytes \n";
        cout <<"numb of threads in warp: "<< prop.warpSize << endl;
        cout <<"max threads per block: "<< prop.maxThreadsPerBlock << endl;
        cout <<"threads allowed "<<1<<" dim: "<< prop.maxThreadsDim[0] << "\n";
        cout <<"blocks allowed in "<<1<<" dim: "<< prop.maxGridSize[0] <<"\n";
        cout <<"threads allowed 2 dim: "<< prop.maxThreadsDim[1] <<"\n";
        cout <<"blocks allowed in 2 dim: "<< prop.maxGridSize[1] << endl;
        cout <<"threads allowed in 3 dim: "<< prop.maxThreadsDim[2] << endl;
        cout <<"blocks allowed in 3 dim: "<< prop.maxGridSize[2] << endl;
        cout <<"cudamem+kernel? "<< prop.deviceOverlap << endl;
        printf("Compute capability: %d.%d\n", prop.major,prop.minor);
        cout << "\n\n";
 }

};

//##################### INPUT #########################################


void Input(Particles* P){ //Prepare all stuff for the simulation
  ifstream ReadInput,ReadConf,ReadPrecedentConf;
  cout << "using " << bl << " blocksPerGrid" << endl;
  cout << "using " << th << " threadsPerBlock" << endl;
  cout << endl;
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

  n_props = 5; //Number of observables

 //alloca la memoria che mi serve su device

 HANDLE_ERROR( cudaSetDevice(1)); 
 
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_x,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_y,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_z,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_xold,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_yold,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_zold,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_vx,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_vy,
                              npart*sizeof(float) ) );
 HANDLE_ERROR( cudaMalloc( (void**)&P->dev_vz,
                              npart*sizeof(float) ) );

 HANDLE_ERROR( cudaMalloc( (void**)&dev_w, sizeof(float)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_v, sizeof(float)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_k, sizeof(float)  ) );
 HANDLE_ERROR( cudaMalloc( (void**)&dev_hist, nbins*10*sizeof(float)  ) );


// variabili da caricare che mi servono per inizializzare quelle da usare su device
float* x = new float[npart];
float* y = new float[npart];
float* z = new float[npart];
float* xold = new float[npart];
float* yold = new float[npart];
float* zold = new float[npart];
float* vx = new float[npart];
float* vy = new float[npart];
float* vz = new float[npart];
		
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

// copio i valori iniziale sulla  memoria allocata precedentemente su device 
   HANDLE_ERROR( cudaMemcpy( P->dev_x, x,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) ); 
   HANDLE_ERROR( cudaMemcpy( P->dev_y, y,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy( P->dev_z, z,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy( P->dev_xold, xold,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) ); 
   HANDLE_ERROR( cudaMemcpy( P->dev_yold, yold,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) );  
   HANDLE_ERROR( cudaMemcpy( P->dev_zold, zold,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) );  
   HANDLE_ERROR( cudaMemcpy( P->dev_vx, vx,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) ); 
   HANDLE_ERROR( cudaMemcpy( P->dev_vy, vy,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) );  
   HANDLE_ERROR( cudaMemcpy( P->dev_vz, vz,
                              npart*sizeof(float),
                              cudaMemcpyHostToDevice ) );  


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


//per inizializzare le coordinate delle particelle quando leggo una configurazione precedente

void first_move(float* x,float*y,float*z,float*xold,float*yold,float*zold,float*vx,float* vy,float* vz) {
	float fx,fy,fz;
	float xnew,ynew,znew;
	float dvec[3]; float dr;

	for (int ip=0;ip<npart;ip++) {
                fx = 0; fy = 0; fz=0;
                for (int i=0;i<npart;i++) {

                        if (i != ip) {

                                dvec[0] = Pbc(x[ip]-x[i]);
                                dvec[1] = Pbc(y[ip]-y[i]);
                                dvec[2] = Pbc(z[ip]-z[i]);
                                dr = sqrt(dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2]);

                                if (dr < rcut){
                                        fx += dvec[0] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); ;
                                        fy += dvec[1] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); ;
                                        fz += dvec[2] * (48.0/pow(dr,14) - 24.0/pow(dr,8)); ;
                                }
                        }
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
return;

}

//chiamo alla fine per liberare la memoria sul device
void exit(Particles* P) {

    HANDLE_ERROR( cudaFree( P->dev_x ) );
    HANDLE_ERROR( cudaFree( P->dev_y ) );
    HANDLE_ERROR( cudaFree( P->dev_z ) );
    HANDLE_ERROR( cudaFree( P->dev_xold ) );
    HANDLE_ERROR( cudaFree( P->dev_yold ) );
    HANDLE_ERROR( cudaFree( P->dev_zold ) );
    HANDLE_ERROR( cudaFree( P->dev_vx ) );
    HANDLE_ERROR( cudaFree( P->dev_vy ) );
    HANDLE_ERROR( cudaFree( P->dev_vz ) );

   cudaFree(dev_w);
   cudaFree(dev_v);
   cudaFree(dev_k);
   cudaFree(dev_hist);

}

void Move_gpu(Particles *P) {

 verlet_gpu<<<bl,th>>>(P->dev_xold,P->dev_yold,P->dev_zold,P->dev_x,P->dev_y,P->dev_z,P->dev_vx,P->dev_vy,P->dev_vz);

}

__global__ void prova_gpu (float* a,float* p) {

 *a = Pbc_gpu(*p);

}


__global__ void verlet_gpu(float*xold,float*yold,float*zold,float*x,float*y,float*z,float*vx,float*vy,float*vz){//,float*fx,float*fy,float*fz) {

	__shared__ float f0[th];
	__shared__ float f1[th];
	__shared__ float f2[th];
	//__shared__ float xnew,ynew,znew;
	float xnew,ynew,znew;
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
		while( tid < gpu_npart ) {
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

		xnew = Pbc_gpu( 2.0 * x[ip] - xold[ip] + f0[0] *gpu_delta*gpu_delta );
		ynew = Pbc_gpu( 2.0 * y[ip] - yold[ip] + f1[0] *gpu_delta*gpu_delta );
		znew = Pbc_gpu( 2.0 * z[ip] - zold[ip] + f2[0] *gpu_delta*gpu_delta );

		vx[ip] = Pbc_gpu(xnew - xold[ip]) / (2.0 * gpu_delta);
    		vy[ip] = Pbc_gpu(ynew - yold[ip]) / (2.0 * gpu_delta);
    		vz[ip] = Pbc_gpu(znew - zold[ip]) / (2.0 * gpu_delta);

    		xold[ip] = x[ip];
    		yold[ip] = y[ip];
    		zold[ip] = z[ip];

    		x[ip] = xnew;
   		y[ip] = ynew;
    		z[ip] = znew;

		}

		ip += gridDim.x;
	}		
}

__global__ void  measure_kinetic(Lock lock,float*vx,float*vy,float*vz,float *k) {

	

	__shared__ float kin[th];

	int cacheIndex = threadIdx.x;
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	float a=0,b=0,c=0;
	
	while (i < gpu_npart ) {
		a += vx[i]*vx[i];
		b += vy[i]*vy[i];
		c += vz[i]*vz[i];
		i += blockDim.x*gridDim.x;
	}

        kin[cacheIndex] = a+b+c;
	__syncthreads();

	int tid = blockDim.x/2;
	while(tid !=0) {

		if(cacheIndex < tid) 
			kin[cacheIndex]+=kin[cacheIndex+tid];

	__syncthreads();

	tid /= 2;

	}

	if(cacheIndex==0) {
		lock.lock();
		*k += kin[0];
		lock.unlock();
	}

}

__global__ void  measure_pot_virial(Lock lock,float* x,float*y,float*z,float *v,float *w,float* hist) {

		
	__shared__ float pot[th];
	__shared__ float vir[th];
	float temp0=0;
	float temp1=0;
	float dvec[3];
	float dr;
	int tid;
	int bin;
	int cacheIndex = threadIdx.x;
	int i = blockIdx.x;
	int j;
	while (i<gpu_npart-1) {
		j = i+1+threadIdx.x;
		while (j<gpu_npart) {

			dvec[0] = Pbc_gpu(x[i]-x[j]);
                        dvec[1] = Pbc_gpu(y[i]-y[j]);
                        dvec[2] = Pbc_gpu(z[i]-z[j]);
                        dr=sqrt(dvec[0]*dvec[0]+dvec[1]*dvec[1]+dvec[2]*dvec[2]);
			bin = int(dr/gpu_binsize);
			atomicAdd( &hist[bin],2);

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
		lock.lock();
		*v += pot[0];
		*w += vir[0];
		lock.unlock();
	}

}

void Measure(Particles* P) {

 Lock lock;

	HANDLE_ERROR( cudaMemset(dev_hist,0, nbins*10*sizeof(float)  ) );
	HANDLE_ERROR( cudaMemset(dev_k,0, sizeof(float)  ) );
	HANDLE_ERROR( cudaMemset(dev_v,0, sizeof(float)  ) );
	HANDLE_ERROR( cudaMemset(dev_w,0, sizeof(float)  ) );

	measure_kinetic<<<bl,th>>>(lock,P->dev_vx,P->dev_vy,P->dev_vz,dev_k);

	measure_pot_virial<<<bl,th>>>(lock,P->dev_x,P->dev_y,P->dev_z,dev_v,dev_w,dev_hist);

	float v,w,k;
	float hist[nbins*10];
	float deltaVr;
        HANDLE_ERROR( cudaMemcpy(hist,dev_hist,nbins*10*sizeof(float),cudaMemcpyDeviceToHost) );
	for (int i=0;i<nbins;i++) {
	    deltaVr = rho*npart*4.*M_PI/3.*(pow((i+1)*bin_size,3)-pow((i)*bin_size,3));
	   gdir[i].push_back(hist[i]/deltaVr);
        }

	HANDLE_ERROR( cudaMemcpy(&w,dev_w,sizeof(float),cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&v,dev_v,sizeof(float),cudaMemcpyDeviceToHost) );
	HANDLE_ERROR( cudaMemcpy(&k,dev_k,sizeof(float),cudaMemcpyDeviceToHost) );

	float t = 0.5*k;

        stima_pot_gpu = v/(float)npart+vtail; //Potential energy per particle
        stima_kin_gpu = t/(float)npart; //Kinetic energy per particle
        stima_temp_gpu = (2.0 / 3.0) *t/(float)npart; //Temperature
        stima_etot_gpu = (t+v)/(float)npart+vtail; //Total energy per particle
        stima_press_gpu = rho*stima_temp_gpu+ (w + ptail*(float)npart ) / vol;

	//carico per fare analisi dati con la CPU
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

float * vx = new float [npart];
float * vy = new float [npart];
float * vz = new float [npart];

  HANDLE_ERROR( cudaMemcpy(vx,P->dev_vx,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(vy,P->dev_vy,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(vz,P->dev_vz,npart*sizeof(float),cudaMemcpyDeviceToHost));

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
  float * xold = new float[npart];
  float * yold = new float[npart];
  float * zold = new float[npart];

  HANDLE_ERROR( cudaMemcpy(xold,P->dev_xold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(yold,P->dev_yold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(zold,P->dev_zold,npart*sizeof(float),cudaMemcpyDeviceToHost));

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

  float * x = new float[npart];
  float * y = new float[npart];
  float * z = new float[npart];

  HANDLE_ERROR( cudaMemcpy(x,P->dev_x,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(y,P->dev_y,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(z,P->dev_z,npart*sizeof(float),cudaMemcpyDeviceToHost));
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

void ConfFinal(Particles*P){ //Write final configuration, per ricominciare dalla simulazione precedente

  float * xold = new float[npart];
  float * yold = new float[npart];
  float * zold = new float[npart];

  HANDLE_ERROR( cudaMemcpy(xold,P->dev_xold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(yold,P->dev_yold,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(zold,P->dev_zold,npart*sizeof(float),cudaMemcpyDeviceToHost));

  ofstream WriteOldConf("config.final",ios::out);
  cout << "Print penultimate configuration in config.final " << endl << endl;
  for (int i=0; i<npart; ++i){
    WriteOldConf << xold[i]/box << "   " <<  yold[i]/box << "   " << zold[i]/box << endl;
  } 
  WriteOldConf.close();
 
  delete [] xold;
  delete [] yold;
  delete [] zold;

  float * x = new float[npart];
  float * y = new float[npart];
  float * z = new float[npart];

  HANDLE_ERROR( cudaMemcpy(x,P->dev_x,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(y,P->dev_y,npart*sizeof(float),cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaMemcpy(z,P->dev_z,npart*sizeof(float),cudaMemcpyDeviceToHost));


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

 accettazione = 2*data[1];
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


