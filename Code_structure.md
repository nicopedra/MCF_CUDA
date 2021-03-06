## STRUCTURE

### Input()  

La prima parte del codice chiama la funzione Input() che legge i parametri necessari all'esecuzione del codice leggendo il file input.dat    
I parametri (tutti in unità di Lennard-Jones) sono:  
- temperatura target
- densità 
- raggio di cut-off
- delta t
- numero di step per la simulazione
- dopo quanti passi stampare in output il punto a cui sono arrivato con l'esecuzione  
  
Da linea di comando si assegna il valore 0 o 1 al parametro restart. Se posto uguale a 0 allora la configurazione iniziale (x,y,z) è un reticolo fcc con coordinate normalizzate contenute nel file config.fcc, e la configurazione al tempo precedente viene assegnata generando delle velocità random (con centro di massa di velocità pari a 0) e ricavando le posizioni (xold,yold,zold). Questa operazione viene eseguita non in Input() ma in Initialization().   
Se posto uguale a 1 si leggono queste configurazioni rispettivamente dai file config.0 e config.final (cioè l'ultima e la penultima configurazione ottenuta dalla simulazione precedente).   
La prima riga dei file config.fcc e config.0 contiene il numero di particelle, parametro npart, di cui si vuole simulare la dinamica.  

Dopo aver letto i parametri viene allocata sulla gpu la memoria necessaria.  
- Le variabili utilizzate per salvare i risultati relativi alle proprietà fisiche sono sia lette sia scritte utilizzando la **global memory**.   
- Le variabili utilizzate per far evolvere la dinamica delle particelle (x,y,z,xold,yold,zold,vx,vy,vz) vengono scritte tramite l'utilizzo della memoria global, ma vengono lette usando la **texture memory**. Per cui dopo aver allocato la memoria necessaria per queste variabili viene eseguito il comando cudaBindtexture per poter leggere il loro contenuto tramite il comando API di Cuda tex1Dfetch().  
- durante l'esecuzione del codice i parametri letti in precedenza rimangono costanti. Inizialmente avevo deciso di passarli direttamente alle funzioni kernel. Successivamente ho deciso di usare la **constant memory**. Per cui dopo aver letto i parametri e aver chiamato la funzione Input() assegno alle rispettive varilabili costanti i valori dei parametri, chiamando il comando cudaMemcpyToSymbol.

### Initialization

Dopo la funzione *Input()* viene chiamata la funzione *Initialization()*. Questa (a seconda del valore assegnato al parametro restart) apre i file config. e inizializza i valori iniziali alle coordinate e alle velocità {x,y,z,xold,yold,zold,vx,vy,vz} (le velocità vengono ricavate dalle posizioni). Le velocità vengono poi riscalate per avere come temperatura iniziale proprio quella target (per permettere al sistema di raggiungerla). Con queste nuove velocità si ricavano le coordinate xold,yold,zold che le particelle devono avere per possedere quei valori di velocità.    
A questo punto tramite il comando cudaMemcpy si copiano i valori ricavati sulla cpu alla gpu. Da ora in poi queste coordinate verranno modificate sempre tramite funzioni global eseguite sulla gpu.    

### Ciclo su nstep MD

Ora si esegue un ciclo *for* su tutti gli step di simulazione che si vogliono eseguire.  
- La prima operazione del ciclo è la chiamata alla funzione Move(). Questa funzione chiama la funzione __global__ verlet_gpu() che evolve le coordinate tramite l'integratore di Verlet.  
Tale funzione segue la logica dell'*Atom Decomposition*, ovvero ogni blocco, con i suoi threads, si occupa di calcolare la forza relativa a una singola particella e a eseguire l'evoluzione di tale particella. (per cui l'indice di blocco rappresenta la coordinate della particella che subisce la forza dovuta alla presenza di tutte le altre particelle (rappresentate dagli indici di thread di quel blocco) ). Per far parlare tra di loro i diversi threads di ogni singolo blocco è stata utilizzata la **shared memory**.   
- La seconda operazione è calcolare ogni 10 step le grandezze fisiche istantanee chiamando la funzione Measure: 
	+ energia cinetica per particella (teorema di equipartizione)
	+ energia potenziale per particella (con correzione vtail dovuta al raggio di cut off)
	+ energia totale per particella (che deve restare costante durante una singola simulazione, insieme NVE *microcanonico*)
	+ temperatura del sistema
	+ pressione (con correzione ptail)
	+ pair radial correlation function g(r)   
 
* Per calcolare queste grandezze vengono chiamate due funzioni __global__:
   
	+ La prima relativa al calcolo dell'energia cinetica totale (è semplicemente un prodotto scalare)   
	+ La seconda calcola il viriale e l'energia potenziale delle particelle, e riempie l'istrogramma relativo alla g(r). Questa funzione è molto simile come metodo alla funzione *verlet_gpu()* poichè è necessario calcolare la distanza tra una particella e tutte le altre. Per cui l'indice della prima particella è l'indice di blocco, mentre l'indice della seconda particella è l'indice di thread. Anche per queste funzione è stato fondamentale utilizzare la **shared memory**.  
È stato necessario inoltre utilizzare la *struct Lock*, che tramite l'utilizzo della sua variabile *mutex* permette l'esecuzione di un thread per volta (operazione necessaria per sommare tra loro le somme parziali per il prodotto scalare e il calcolo dell'energia potenziale e il viriale). Per riempire l'istogramma è stato necessario invece usare la funzione API di Cuda *atomicAdd()* (per non sovrascrivere conteggi e quindi rischiare dicontare di meno).  

- Man mano le grandezze misurate vengono raccolte dentro un vector<>, poi necessario all'analisi dati

### Data Analysis

Uscito dal ciclo vengono chiamate le funzioni *print_properties()* e *data_blocking()*. Queste due funzioni eseguono l'analisi dati delle grandezze misurate e raccolte durante la simulazione. L'analisi dati viene interamente eseguita su cpu. 

### Exit()

L'ultima parte del codice salva nei file config.0 e config.final le configurazioni finale e penultima. Per poi essere lette successivamente con la simulazione successiva (e così via).    
Successivamente viene chiamata funzione Exit() che libera tutta la memoria allocata sulla gpu e chiama la funzione *cudaunbindtexture*.

## EQUILIBRATION 

All'inizio del codice viene definita (o non definita) tramite *define* la variabile *equilibration*. Se questa è definita allora viene eseguita l'equilibrazione:  
- un ciclo *while* che controlla la differenza tra la temperatura media raggiunta dal sistema e la temperatura target. Se tale differenza (in valore assoluto) è minore di 2 sigma (ovvero di due volte l'errore ottenuto dall'analisi dati) allora l'equilibrazione si considera conclusa.    
- Ogni ciclo while ricomincia ponendo il parametro *restart* uguale a 1. In questo modo le configurazioni iniziali vengono assegnate (all'inizio del ciclo) copiando le configurazioni assunte dalla simulazione precedente (ovvera quella del ciclo precedente) e vengono di nuovo riscalate le velocità (per far raggiungere al mio sistema la temperatura target).

## MEMORIE UTILIZZATE

Ho svolto tre diversi codici utilizzando tre diverse memorie:  
- texture memory
- global memory 
- unified memory  

Sarà interessante testare le varie performance basandomi sull'utilizzo di queste memorie. La cosa interessante è che il numero di registri usati per thread è più grande nella global memory e nella unified memory rispetto alla texture memory.


    

