#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define OUTPUT_NAME "outMatrix.bin"

void sumBuffer(int* buff1, int* buff2, int* outPutBuff, int size);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    int matrixRowSize = atoi(argv[1]);
    char *fileN1, *fileN2;
    int rs, nints, iterationsNeeded;
    int *buffer1, *buffer2, *outBuffer;

    MPI_File fh1, fh2, outFH;
    MPI_Status status;
    MPI_Offset offset;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    nints = matrixRowSize;
    iterationsNeeded = (matrixRowSize / size) + 1;
    fileN1 = argv[2];
    fileN2 = argv[3];

    buffer1 = (int*)malloc(matrixRowSize*sizeof(buffer1));
    buffer2 = (int*)malloc(matrixRowSize*sizeof(buffer2));
    outBuffer = (int*)malloc(matrixRowSize*sizeof(outBuffer));

    MPI_File_open(MPI_COMM_WORLD, fileN1, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh1);
    MPI_File_open(MPI_COMM_WORLD, fileN2, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh2);
    MPI_File_open(MPI_COMM_WORLD, OUTPUT_NAME, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &outFH);

    for (int i = 0; i < iterationsNeeded ; i++)
    {
        rs = rank+i*size;
        if ( rs < matrixRowSize)
        {
            offset = rs * nints * sizeof(int);
            MPI_File_read_at(fh1, offset, buffer1, matrixRowSize, MPI_INT, &status);
            MPI_File_read_at(fh2, offset, buffer2, matrixRowSize, MPI_INT, &status);

            sumBuffer(buffer1, buffer2, outBuffer, matrixRowSize);
            MPI_File_write_at(outFH, offset, outBuffer, matrixRowSize, MPI_INT, &status);
            //printf("process %d offset: %d\n",rank,rs);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_File_close(&fh1);
    MPI_File_close(&fh2);
    MPI_File_close(&outFH);
    free(buffer1);
    free(buffer2);
    free(outBuffer);
    MPI_Finalize();
    return 0;
}

void sumBuffer(int* buff1, int* buff2, int* outPutBuff, int size)
{
    for (int i = 0 ; i < size; i++)
    {
        outPutBuff[i] = buff1[i]+buff2[i];
    }
    return;
}