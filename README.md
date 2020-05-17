## Molecular Dynamics CUDA
  
  
Dentro *global_gpu* sono presenti i file relativi all'implementazione in CUDA del codice di MD  
presente nella cartella *ex_cpu*.   
  
dentro queste due cartelle sono presenti anche le seguenti:  
- input : diversi file di input a seconda di cosa si vuole simulare  
- config : diverse configurazioni di reticolo fcc con diverso numero di particelle. La simulazione numero 0 va sempre fatta partire con la configurazione config.fcc  
- old\_config : contiene vecchie configurazioni di solido liquido e gas con 108 particelle  
  
  
Ho scelto di usare la memoria global. (avevo anche implementato un codice usando la memoria texture. Ma le considerazioni seguenti sono esattamente le medesime per entrambi i codici).      
Ho constato di avere certi limiti di utilizzo per numero di threads e blocchi:    
- utilizzando un solo blocco (bl = 1) ottengo risultati compatibili con il codice cpu utilizzando un numero di threads massimo di 512 (th = 12)
- mantenendo il numero di threads per blocco a 512 ottengo risultati compatibili con il codice cpu fino a un numero massimo di blocchi pari a 15.  
- mantenendo fisso il numero di threads a 2 (th = 2) ottengo risultati compatibili con il codice cpu arrivando a un numero massimo di blocchi pari a 120 (bl = 120)  
- variando ora il numero di threads mantenendo fisso bl a 120, ottengo risultati compatibili con il codice cpu avendo un numero massimo di threads per blocco pari a 64.  
  
Ho simulato al massimo un sistema contenente 4000 particelle (sulla cpu solo 2916 perchè con 4000 avrei dovuto aspettare troppo tempo per vedere dei risultati).  

I test fatti sono stati eseguiti su un solido (input.solid) eseguendo una singola simulazione con nstep = 10000.  
dentro input.dat (il file letto dalla funzione input per inizializzare i parametri) viene spiegato chi sono i vari parametri   
  
  
esecuzione del codice:   
- *make* per compilare  
- *./MolDyn\_NVE.x parametro\_restart* per eseguire  
nota: il *parmetro\_restart* per i test è sempre stato messo pari a 0, ovvero la configurazione iniziale è quella contenuta in config.fcc, un reticolo fcc. Se posto uguale a 1 legge la configurazione della simulazione precedente (utile per la fase di equilibrazione).  
  
nelle rispettive cartelle *global\_gpu* e *ex\_cpu* sono anche presenti i file di profiling del codice.  
nel file *profile*
