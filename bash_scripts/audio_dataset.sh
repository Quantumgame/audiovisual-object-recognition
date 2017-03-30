#!/bin/bash

while read material
do
  while read forma
  do

    echo 'Da vez:' $forma '-' $material
    sleep 10
    
    for i in $(seq 0 99)
    do

      arecord cubo-solido/cubo-solido-$i.wav -D plughw:1 -c 4 -d 3 -f S16_LE -r 16000 & #plughw:1 -c 4 -d 2 -r 16000
      #arecord wav-hq/$forma-$material-$i-microfone.wav -D plughw:0 -c 1 -d 3 -f S16_LE -r 44100 & #plughw:1 -c 4 -d 2 -r 16000
      
      echo 'Nova iteracao em 12s'
      sleep 2
      clear
      sleep 2
      # rosparam set knock true &
    
      #voltar a posicao inicial     
        
    done
  done < $2
done < $1