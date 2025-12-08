Run below commands for the AES GCM 128/256 encryption test
$ mkdir build
$ cd build
$ cmake ../
$ make
$ cd ..
// Test AES GCM 128 Sequence program 
$ python3 test_AES_128_sequence.py

// Test AES GCM 128 Parallel CUDA program 
$ python3 test_AES_128_parallel_cuda.py

// Test AES GCM 256 Sequence program 
$ python3 test_AES_256_sequence.py

// Test AES GCM 256 Parallel CUDA program 
$ python3 test_AES_256_parallel_cuda.py



Note:
Make sure libwb files and folders are present as per the Assigments
