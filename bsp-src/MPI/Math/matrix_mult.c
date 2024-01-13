#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

double* initiate_matrix(int size);
void print_matrix(double *matrix, int size);
void matrix_mult(double *m1, double *m2, double *result, int size, int rank, int total_process);

int main(int argc, char** argv)
{
    int rank, size, m_size;
    double *matrix1, *matrix2, *result;

    // set up mpi environment and fetch processes ID/ size of communicator
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2 && rank == 0)
    {
        printf("Matrix_Size is missing\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return -1;
    }
    else if (atoi(argv[1]) > size && rank == 0)
    {
        printf("Not enough Processes allocated\n");
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
        return -1;
    }

    // allocate memory for the computations
    m_size = atoi(argv[1]);

    if (rank == 0)
    {
        matrix1 = initiate_matrix(m_size);
        matrix2 = initiate_matrix(m_size);
        result = (double*)malloc(sizeof(result)*m_size*m_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // matrix multiplication
    matrix_mult(matrix1, matrix2, result, m_size, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // print result
    if (rank == 0)
    {
        print_matrix(result,m_size);
    }

    MPI_Finalize();
    return 0;
}

void matrix_mult(double *m1, double *m2, double *result, int matrix_size, int rank, int total_process)
{

    if (rank >= matrix_size) {return;}

    // determine the ranks of processes which whom information is exchanged.
    int next_rank = (rank + 1) % matrix_size;
    int prev_rank = (rank != 0) ? rank-1 : matrix_size - 1; 

    // storage for matrix rows
    double r1[matrix_size];
    double r2[matrix_size];
    double r3[matrix_size];    
    
    MPI_Status status;

    // initialize r3
    for (int i = 0 ;i < matrix_size ; i++)
    {
        r3[i] = 0;
    }
    // distribute the rows of both matrices among processes
    MPI_Scatter(m1, matrix_size, MPI_DOUBLE, &r1, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(m2, matrix_size, MPI_DOUBLE, &r2, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // compute the rows of the result matrix
    for (int i = rank ; i < matrix_size+rank; i++)
    {   
        for (int j = 0 ; j < matrix_size ; j++)
        {
            r3[j] += r1[i%matrix_size]*r2[j];
        }
        // rotate access to the rows of the second matrix
        MPI_Sendrecv_replace(&r2, matrix_size, MPI_DOUBLE, prev_rank, 2, next_rank, 2, MPI_COMM_WORLD, &status);
    }
    
    //retrieve the rows from the processes and store data in the result array
    MPI_Send(&r3, matrix_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        for (int i = 0; i < matrix_size ; i++)
        {
            // fetch data
            MPI_Recv(&r3, matrix_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);  
            for (int j = 0; j < matrix_size ; j++)
            {   
                // store data
                result[matrix_size*i+j] = r3[j];
            }
        }
    }
    
}

void print_matrix(double *matrix, int size)
{
    for (int i = 0 ; i < size*size ; i++)
    {
        if (i % size == 0 && i != 0){
            printf("\n");
        }
        printf("%lf ", matrix[i]);   
    }
    printf("\n");
}

double *initiate_matrix(int size){
    double *matrix = (double*)malloc(sizeof(matrix)*size*size);
    
    for (int i = 0 ; i < size ; i++)
    {
        for (int j = 0 ; j < size ; j++)
        {
            matrix[i*size+j] = i+j;
        }
    }

    //print_matrix(matrix,size);
    return matrix;
}
