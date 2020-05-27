## STRUCTURE

### Input()  

La prima parte del codice chiama una funzione Input() che legge i parametri necessari all'esecuzione del codice leggendo il file input.dat    
I parametri (tutti in unità di Lennard-Jones) sono:  
- temperatura target
- densità 
- raggio di cut-off $r_{cutoff}$
- delta t
- numero di step per la simulazione
- dopo quanti passi stampare in output il punto a cui sono arrivato con l'esecuzione  
Da linea di comando si assegna il valore $0$ o $1$ al parametro $restart$. Se posto uguale a 0 allora la configurazione iniziale è un reticolo fcc, e la configurazione al tempo precedente viene assegnata generando delle velocità random (con centro di massa di velocità pari a 0) e ricavando le posizioni.  
Se posto uguale a $1$ si leggono queste configurazioni rispettivamente dai file $config.0$ e $config.final$ (cioè l'ultima e la penultima configurazione ottenuta dalla simulazione precedente). 
