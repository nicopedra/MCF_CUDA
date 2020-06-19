## Molecular Dynamics CUDA
  
Dentro *texture_gpu* sono presenti i file relativi all'implementazione in CUDA del codice di MD  
presente nella cartella *ex_cpu* utilizzando la memoria *texture*.    
Dentro *texture_double_precision* sono presenti i file relativi all'implementazione in CUDA del codice di MD  
presente nella cartella *ex_cpu_double* utilizzando la memoria *texture* ma con precisione double.    
Dentro *global_gpu* sono presenti i file relativi all'implementazione in CUDA del codice di MD  
presente nella cartella *ex_cpu* utilizzando interamente la memoria *global*.    
Dentro *portable_gpu* sono presenti i file relativi all'implementazione in CUDA del codice di MD  
presente nella cartella *ex_cpu* utilizzando la memoria *unified* supportata dal device.      
  
È presente inoltre una cartella List\_method, dentro:  
- cartella ex\_cpu contenente l'implementazione della *verlet list*  
- verlet\_list contenente l'implementazione in CUDA del metodo Verlet List, usando però una matrice di booleani per indicare i primi vicini  
  

dentro ogni cartella sono presenti anche le seguenti:  
- input : diversi file di input a seconda di cosa si vuole simulare  
- config : diverse configurazioni di reticolo fcc con diverso numero di particelle. La simulazione numero 0 va sempre fatta partire con la configurazione config.fcc  
- old\_config : contiene vecchie configurazioni di solido liquido e gas con 108 particelle  
  
La cosa importante da tenere a mente è l'utilizzo di registri per thread relativi ad ogni kernel. Se questo numero supera quello concesso allora il kernel non viene eseguito.  
Per questo motivo ho avuto limitazioni di utilizzo per numero di threads.  
Inoltre un'altra cosa da tenere a mente è che per ogni kernel esiste un numero ottimale di threads per blocco, che **non è** necessariamente il massimo concesso (1024).  
Questo numero è molto spesso legato al massimo numero di thread che contemporaneamente possono agire in un kernel. E questo numero è a sua volta legato al numero di registri utilizzati.  
Se ad esempio per un kernel il massimo numero di registri utilizzati per thread è pari a 32, e nel kernel un singolo thread usa 30 registri allora utilizzare 1024 threads per blocco peggiorerà la performance, mentre usarne solo 128 la migliorerà estremamente.   
In definitiva per ogni kernel lanciato è stato prescelto un numero di threads da lanciare.   
 
Ho simulato al massimo un sistema contenente 4000 particelle
  
Nel jupyter notebook sono presenti alcuni risultati e confronti ottenuti   

I test fatti sono stati eseguiti su un solido (input.solid) eseguendo una singola simulazione con nstep = 10000.  
dentro input.dat (il file letto dalla funzione input per inizializzare i parametri) viene spiegato chi sono i vari parametri   
  
esecuzione del codice:   
- *make* per compilare  
- *./MolDyn\_NVE.x parametro\_restart* per eseguire  
nota: il *parmetro\_restart* per i test è sempre stato messo pari a 0, ovvero la configurazione iniziale è quella contenuta in config.fcc, un reticolo fcc. Se posto uguale a 1 legge la configurazione della simulazione precedente (utile per la fase di equilibrazione).  
  
In fase di compilazione tramite la flag -Xptxas è possibile vedere quanta memoria viene utilizzata per ogni singolo kernel e quanti registri per ogni singolo thread all'interno del kernel.  
Il profile del codice è stato eseguito con nvprof.  
 
In tutte le cartelle sono anche presenti i file di profiling del codice, nel file *profile*   
  
È anche presente la cartella *generate_lattice_coord*, che contiene un codice che genera le coordinate per un reticolo fcc o bcc dando il passo reticolare come input.  
