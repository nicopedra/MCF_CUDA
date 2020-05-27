## Molecular Dynamics CUDA
  
Dentro *texture_gpu* sono presenti i file relativi all'implementazione in CUDA del codice di MD  
presente nella cartella *ex_cpu* utilizzando la memoria *texture*.    
Dentro *texture_double_precision* sono presenti i file relativi all'implementazione in CUDA del codice di MD  
presente nella cartella *ex_cpu* utilizzando la memoria *texture* ma con precisione double.    
    
dentro ogni cartella sono presenti anche le seguenti:  
- input : diversi file di input a seconda di cosa si vuole simulare  
- config : diverse configurazioni di reticolo fcc con diverso numero di particelle. La simulazione numero 0 va sempre fatta partire con la configurazione config.fcc  
- old\_config : contiene vecchie configurazioni di solido liquido e gas con 108 particelle  
  
Ho constato di avere certi limiti di utilizzo per numero di threads:     
fino a 512 (1024 solo per il calcolo dell'energia cinetica) 
 
La combinazione finale usata è:  
- numero di blocchi = 512 
- numero di threads per blocco = 256   
 
Ho simulato al massimo un sistema contenente 4000 particelle (sulla cpu solo 2916 perchè con 4000 avrei dovuto aspettare troppo tempo per vedere dei risultati).  

I test fatti sono stati eseguiti su un solido (input.solid) eseguendo una singola simulazione con nstep = 10000.  
dentro input.dat (il file letto dalla funzione input per inizializzare i parametri) viene spiegato chi sono i vari parametri   
  
esecuzione del codice:   
- *make* per compilare  
- *./MolDyn\_NVE.x parametro\_restart* per eseguire  
nota: il *parmetro\_restart* per i test è sempre stato messo pari a 0, ovvero la configurazione iniziale è quella contenuta in config.fcc, un reticolo fcc. Se posto uguale a 1 legge la configurazione della simulazione precedente (utile per la fase di equilibrazione).  
  
In tutte le cartelle sono anche presenti i file di profiling del codice.  
nel file *profile*
