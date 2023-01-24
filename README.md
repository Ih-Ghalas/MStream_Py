# MStream_py

## Overview 
This is an implementation of MStream [[1]](#1) in python using the River library. 
MStream is used for fast anomaly detection in multi aspects stream. 
It can detect unusual group anomalies as they occur, in a dynamic manner. MSTREAM has the following properties: 
(a) it detects anomalies in multi-aspect data including both categorical and numeric attributes; 
(b) it is online, thus processing each record in constant time and constant memory; 
(c) it can capture the correlation between multiple aspects of the data.

<img width="1380" alt="1e923500-fdf3-11ea-85d7-19ea9c4332cd" src="https://user-images.githubusercontent.com/91777714/214274702-9dc09ba6-f009-4499-a750-686096814b4d.png">

## Dataset
You can find the datasets in the forlder data.
The preprocessed dateset of KDDCUP99 is dataset.csv
The preprocessed dataset of  UNSW-NB 15 data2.csv

## Contributers 

Ihssane Ghalas 
Chaimae Akharaze
Maroua El-Arni


## References
<a id="1">[1]</a> 
MSTREAM: Fast Anomaly Detection in Multi-Aspect Streams
Siddharth Bhatia, Arjit Jain, Pan Li, Ritesh Kumar, Bryan Hooi
