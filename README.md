# BSP-S3: MPI and SVM

This repository stores code produced during the research done for my 3rd bachelor semester project.

## MPI program <command_line_arguments>
- Hello_World_MPI : "Hello World" with MPI
- pi_monte_carlo_mpi <number_of_point_samples> : approximation of pi
- matrix_mult <size_of_matrix> : matrix multiplication
- createLargeMatrix <size> : creates a binary file containing size*size elements
- readLargeMatrix <size> : prints the elements contained in a binary file of size sze*size
- addLargeMatrices <size> <file1> <file2> : reads two matrices from binary files, adds them together and stores the result in another binary file
  
## SVM
- python3 rsvm.py -<t/a/p> (<-o filename>) <train/test file>

If you want to test the SVM model execute the following:
The only restriction on the dataset is that the column containing the label has to be called 'label'
```console
python3 rsvm.py .t trainIris.csv
python3 rsvm.py -a testIris.csv
```
When making predictions the program is going to output a file called 'model_name'_results that contains the predictions made by the model. Predictions can be made using:
```
python3 rsvm.py -p testIris.csv
```
