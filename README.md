# nn_from_scratch
A neural network library only using the C++ standard library. Based on notation and python implementation from http://neuralnetworksanddeeplearning.com/

## Build
To build, cd into the repository, then:
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```
## Positive or negative demo
A very basic demo that trains a two-neuron network to classify numbers as positive or negative.
```bash
./scripts/positive_or_negative
```
Output:
```
Randomly initialising network with layer sizes [1, 2]
Initial evaluation: 10 / 20
Epoch 0: 11 / 20
...
Epoch 9: 20 / 20
Enter a float: 12
Input: [12]
Output: [1, 3.58939e-12]
Prediction: positive!
Enter a float: -0.1 
Input: [-0.1]
Output: [0.504695, 0.505574]
Prediction: negative!
```

## MNIST demo
Download and unzip the MNIST test and training datasets from http://yann.lecun.com/exdb/mnist/, then run this demo with:
```bash
./scripts/mnist ~/Downloads/train-images.idx3-ubyte ~/Downloads/train-labels.idx1-ubyte ~/Downloads/t10k-images.idx3-ubyte ~/Downloads/t10k-labels.idx1-ubyte
```
Output:
```
Reading /home/jamie/Downloads/train-images.idx3-ubyte
Reading /home/jamie/Downloads/train-labels.idx1-ubyte
Reading /home/jamie/Downloads/t10k-images.idx3-ubyte
Reading /home/jamie/Downloads/t10k-labels.idx1-ubyte
Randomly initialising network with layer sizes [784, 16, 16, 10]
Initial evaluation: 1010 / 10000
Epoch 0: 8957 / 10000
... (some minutes pass)
Epoch 29: 9384 / 10000
                            
                            
                            
       ▄▄▄                  
      ▀▀▀▀███████████▄      
                   ██       
                 ▄██        
                ▄██         
                ██          
              ▄██           
             ██▀            
           ▄██              
           ███              
           ▀▀               
Actual: 7, Network: 7
                            
            ▄▄▄▄            
        ▄████████           
        ██▀    ██▀          
              ███           
            ▄███            
           ███▀             
          ███▀              
        ▄███▀               
        ███           ▄▄▄▄  
        ████████████████▀▀  
             ▀▀▀▀▀          
                            
                            
Actual: 2, Network: 2
                            
                            
                 █          
                ██          
               ▄█           
               ██           
              ██            
             ▄█▀            
             ██             
            ██▀             
           ▄██              
           ██               
                            
                            
Actual: 1, Network: 1
                            
                            
             ███            
           ▄████▄           
         ▄█████████▄        
        ▄█████▀  ▀██▄       
        ███▀       ███      
       ▄██         ███▀     
       ███      ▄▄███▀      
       ███  ▄▄▄█████▀       
        ██████████▀▀        
         ▀▀███▀▀            
                            
                            
Actual: 0, Network: 0
                            
                            
           ▄                
          ▄█       ▀█       
         ▄█▀       ▄█       
        ██        ▄██       
       ██        ▄██        
       ██        ██▀        
       ▀██▄▄▄▄█████         
                ▀██         
                ▀██         
                ██▀         
                ▀           
                            
Actual: 4, Network: 4
                            
                            
                 ▄          
                ███         
               ███          
              ███▀          
              ███           
             ███            
            ▄██▀            
            ███             
           ▄██              
           ███              
            ▀               
                            
Actual: 1, Network: 1
                            
                            
          ▄▄                
        ▄██▀        ▄█▀     
       ▄█▀         ▄█▀      
      ███        ▄▄█▀       
      ▀████▄▄▄▄████▀        
         ▀▀▀▀▀▀ ██▀         
               ██           
              ██▀           
             ▄██            
             ███▄█▀         
              ▀▀            
                            
Actual: 4, Network: 4
                            
                            
                            
           ▄██              
         ▄█████▄            
         ██▀▀▀███▄          
        ▄█▄   ████          
         ██▄▄██▀██▄         
          ▀██▀  ▀██▄        
                  ██        
                   ▀█▄      
                    ▀█▄     
                     ▀█     
                            
Actual: 9, Network: 9
...
```

## TODO
* Unit tests!
