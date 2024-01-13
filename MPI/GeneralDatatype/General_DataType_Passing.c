#include <stdio.h>
#include <mpi.h>

struct Person {
    char name[20];
    int age;
    double salary;
};

int main(int argc, char** argv)
{
    int rank,size;
    struct Person people[3] = {{"Huey", 1, 4000},{"Dewey", 2, 5000},{"Louie", 3, 6000}};
    struct Person people2[3];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype MPI_PERSON;
    struct Person dummy;
    int num_of_types[3] = {20, 1, 1};
    MPI_Aint displacements[3];
    MPI_Aint start_address;
    MPI_Get_address(&dummy, &start_address);
    MPI_Get_address(&dummy.name[0], &displacements[0]);
    MPI_Get_address(&dummy.age, &displacements[1]);
    MPI_Get_address(&dummy.salary, &displacements[2]);

    displacements[0] = MPI_Aint_diff(displacements[0], start_address);
    displacements[1] = MPI_Aint_diff(displacements[1], start_address);
    displacements[2] = MPI_Aint_diff(displacements[2], start_address);
    MPI_Datatype types[3] = { MPI_CHAR, MPI_INT, MPI_DOUBLE};
    MPI_Type_create_struct(3, num_of_types, displacements, types, &MPI_PERSON);
    MPI_Type_commit(&MPI_PERSON);

    switch(rank)
    {
        case 0:
        {
            // Send the message
            MPI_Send(&people, 3, MPI_PERSON, 1, 0, MPI_COMM_WORLD);
            break;
        }
        case 1:
        {
            // Receive the message
            MPI_Recv(&people2, 3, MPI_PERSON, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < 3 ; i++)
            {
                printf("Process %d: receive Info of %s, aged %d, earning %lf\n", rank, people2[i].name, people2[i].age, people2[i].salary);
            }
            break;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

