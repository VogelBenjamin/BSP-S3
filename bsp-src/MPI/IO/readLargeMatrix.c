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

    char *fileName = argv[2];
    int nints, iterationsNeeded;
    int *buffer, *totalBuff, *sendBuffer, *check;
    int sendBuffSize = matrixRowSize*sizeof(int)*iterationsNeeded;
    
    nints = matrixRowSize;
    iterationsNeeded = (matrixRowSize / size) + 1;

    sendBuffer = (int*)malloc(sendBuffSize);

    buffer = (int*)malloc(matrixRowSize*sizeof(int));

    check = (int*)malloc(iterationsNeeded*sizeof(int));

    if (rank == 0)
    {
        totalBuff = (int*)malloc(matrixRowSize*matrixRowSize*sizeof(int));
    }
    for (int i = 0; i < iterationsNeeded; i++)
    {
        check[i]=-1;
    }

    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset;

    MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    int rs;
    for (int i = 0; i < iterationsNeeded ; i++)
    {
        rs = rank+i*size;
        if ( rs < matrixRowSize)
        {
            offset = rs * nints * sizeof(int);
            MPI_File_read_at(fh, offset, buffer, matrixRowSize, MPI_INT, &status);
            MPI_Send(buffer, matrixRowSize, MPI_INT, 0, rank+i*size, MPI_COMM_WORLD);
            printf("process %d offset: %d\n",rank,rs);
        }
        
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    { 
        for (int i = 0 ; i < matrixRowSize; i++)
        {
            MPI_Recv(totalBuff+i*matrixRowSize, matrixRowSize, MPI_INT, MPI_ANY_SOURCE, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int k = 0 ; k < matrixRowSize; k++)
            {
                printf("%d ", totalBuff[matrixRowSize*i+k]);
            }
            printf("\n");
        }
        free(totalBuff);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    free(buffer);
    free(sendBuffer);
    MPI_Finalize();
    return 0;
}