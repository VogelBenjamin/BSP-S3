#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
./a.out <matrixRowSize> <filename>
*/

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    int matrixRowSize = atoi(argv[1]);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int input[10] = {0,1,2,3,4,5,6,7,8,9};
    char *fileName = argv[2];
    int nints, iterationsNeeded;
    int *buffer;

    buffer = (int*)malloc(matrixRowSize*sizeof(int));
    for (int i = 0 ; i < matrixRowSize; i++)
    {
        buffer[i] = input[i%10];
    }

    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset;

    nints = matrixRowSize;
    iterationsNeeded = (matrixRowSize / size) + 1;
    
    //printf("iterationsNeeded: %d\n",iterationsNeeded);
    MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    for (int i = 0; i < iterationsNeeded ; i++)
    {
        if ((rank+i*size) < matrixRowSize)
        {
            offset = (rank+i*size) * nints * sizeof(int);
            MPI_File_write_at(fh, offset , buffer, matrixRowSize, MPI_INT, &status);
            
            //printf("process %d offset: %d\n",rank, (rank+i*size));
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);
    free(buffer);
    MPI_Finalize();
    return 0;
}