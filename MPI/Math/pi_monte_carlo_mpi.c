#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <time.h>
#include <mpi.h>

struct point {
    double x;
    double y;
};

double compute_distance(struct point *p1, struct point *p2);

int main(int argc, char **argv){
    int size_Of_Cluster, process_Rank;

    MPI_Init(&argc,&argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size_Of_Cluster);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_Rank);
    
    size_t i;
    time_t t;
    double x,y;
    double result, pi_partial, pi;
    int accuracy = atoi(argv[1]);
    int within = 1;
    int outside = 1;
    struct point center;
    struct point *new_point;
    center.x = 0;
    center.y = 0;
    if(process_Rank != 0){
        for (i = 0 ; i < accuracy ; i++){
            new_point = (struct point*)malloc(sizeof(new_point)*accuracy);
            
            x = (double)rand()/RAND_MAX;
            y = (double)rand()/RAND_MAX;

            new_point->x = x;
            new_point->y = y;

            result = compute_distance(new_point,&center);
            if (result >= 1){
                outside += 1;
            } 
            else {
                within += 1;
            }
            free(new_point);
        }
        pi_partial = ((double)within/(double)accuracy)*4;
        
    }
    MPI_Reduce(&pi_partial, &pi, size_Of_Cluster, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_Rank == 0)
    {
        printf("pi approximation is: %lf\n", pi/(size_Of_Cluster-1));
    }
    MPI_Finalize();

    return 0;
}

double compute_distance(struct point *p1, struct point *p2){
    return pow(p1->x - p2->x,2) + pow(p1->y - p2->y,2);
}