// This code is loosely based on https://rosettacode.org/wiki/Reduced_row_echelon_form#C
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <sodium.h>

#include <immintrin.h>

#define TALLOC(n,typ) calloc(n,sizeof(typ))


#define EL_Type unsigned char

#define bitLength 8*sizeof(EL_Type)

typedef struct sMtx {
    int     dim_x, dim_y;
    EL_Type *m_stor;
    EL_Type **mtx;
} *Matrix, sMatrix;

typedef struct sRvec {
    int     dim_x;
    EL_Type *m_stor;
} *RowVec, sRowVec;


int isMember(int *array, int size, int candidate) {

    if (size==0) {
        return 0;
    }

    for (int i = 0; i<size; i++) {
        if (array[i]==candidate) {
            return 1;
        }
    }
    return 0;
}



void distinctRandomNumbers(int number, int range, int *array) {

    for (int i=0; i<number; i++) {
        array[i] = -1;
    }

    for (int i=0; i<number; i++) {
        int rnd;
        do {
            rnd = randombytes_uniform(range);
        } while (isMember(array,i,rnd)==1);

        array[i] = rnd;

    }


}


Matrix NewMatrix( int x_dim, int y_dim )
{
    int n;
    Matrix m;

    x_dim = x_dim/(bitLength);

    //Malloc is thread save starting from Glibc 2.2 ragma omp critical
    {
        m = TALLOC( 1, sMatrix);
    }
    n = x_dim * y_dim;
    m->dim_x = x_dim;
    m->dim_y = y_dim;

    //Malloc is thread save starting from Glibc 2.2 ragma omp critical
    {
        m->m_stor = TALLOC(n, EL_Type);
        m->mtx = TALLOC(m->dim_y, EL_Type *);
    }

    //pragma omp parallel for private(n) schedule(dynamic,1000)
    for(n=0; n<y_dim; n++) {
        m->mtx[n] = m->m_stor+n*x_dim;
    }
    return m;
}

void MtxSetRow(Matrix m, int irow, EL_Type *v)
{
    int ix;
    EL_Type *mr;
    mr = m->mtx[irow];

    //pragma omp parallel for private(ix) schedule(dynamic,1000)
    for(ix=0; ix<m->dim_x; ix++)
        mr[ix] = v[ix];
}

Matrix InitMatrix(Matrix m, int k)
{
    int col;
    int x_dim = m->dim_x;
    int y_dim = m->dim_y;

    //pragma omp parallel for private(col) schedule(guided,8192)
    for (col = 0; col<x_dim*(bitLength); col++) {
        int locations[k];
        distinctRandomNumbers(k, y_dim, locations);
        int row;
        for ( row = 0; row<y_dim; row++) {
            if (isMember(locations, k, row)==1) {
                EL_Type bucket = m->mtx[row][col/(bitLength)];

                unsigned char mask = 1<<(bitLength-(col % bitLength)-1);
                bucket = bucket | mask;

                m->mtx[row][col/(bitLength)] = bucket;
            }
        }
    }


    return m;
}


void RemoveRandomRows(Matrix m, int toRemove) {
    int locations[toRemove];
    distinctRandomNumbers(toRemove, m->dim_y, locations);

    int i;
    int col,row;
    for (i = 0; i<toRemove; i++) {
        for (col = 0; col<m->dim_x; col++) {
            row = locations[i];
            m->mtx[row][col] = 0;
        }
    }


}


void MtxDisplay( Matrix m )
{
    int iy, ix;
    for (iy=0; iy<m->dim_y; iy++) {
        for (ix=0; ix<m->dim_x; ix++) {
            EL_Type bucket = m->mtx[iy][ix];
            for (int j = 0; j<bitLength; j++) {
                printf("%u ", (bucket>>(bitLength - j - 1))&1);
            }

        }
        //        printf("    %u\n", m->mtx[iy][0]);
        printf("\n");
    }
    printf("\n");
}

void MtxMulAndAddRows(Matrix m, int ixrdest, int ixrsrc)
{

    EL_Type *drow, *srow;
    drow = m->mtx[ixrdest];
    srow = m->mtx[ixrsrc];

    int totalBits = bitLength * m->dim_x;

    for (int i = 0; i<totalBits/256; i++) {
        __m256i sbucket = _mm256_loadu_si256((__m256i *) &srow[i*(32)]);
        __m256i dbucket = _mm256_loadu_si256((__m256i *) &drow[i*(32)]);
        dbucket = _mm256_xor_si256 (sbucket, dbucket);
        memcpy((char*) &drow[i*32], (char*) &dbucket, 32);
    }


}

void MtxSwapRows( Matrix m, int rix1, int rix2)
{
  EL_Type *r1, *r2, temp;
   int ix;

  /*if (rix1 != rix2) {
        EL_Type *t;
        t = (m->mtx[rix1]);
        m->mtx[rix1] = m->mtx[rix2];
        m->mtx[rix2] = t;

    }*/

    if (rix1 != rix2) {
        r1 = m->mtx[rix1];
        r2 = m->mtx[rix2];
        for (ix=0; ix<m->dim_x; ix++) {
      temp = r1[ix];
      r1[ix]=r2[ix];
      r2[ix]=temp;
    }

    }
}


//define MtxGet( m, rix, cix ) m->mtx[rix][cix]
#define MtxGet(m,rix,cix) (((m->mtx[rix][cix/bitLength]) >> (bitLength - 1 - (cix%bitLength)) )&1)


void MtxToReducedREForm(Matrix m)
{
    int lead;
    int rix, iix;
    int rowCount = m->dim_y;

    lead = 0;
    for (rix=0; rix<rowCount; rix++) {
        if (lead >= m->dim_x * bitLength)
            return;
        iix = rix;
        while (0 == MtxGet(m, iix,lead)) {
            iix++;
            if (iix == rowCount) {
                iix = rix;
                lead++;
                if (lead == m->dim_x * bitLength)
                    return;
            }
        }


        MtxSwapRows(m, iix, rix );

#pragma omp parallel for if(rowCount>8192)
        for (iix=0; iix<rowCount; iix++) {
            if ( iix != rix ) {
                EL_Type lv = MtxGet(m, iix, lead );
                if (lv==1) {
                    MtxMulAndAddRows(m,iix, rix) ;
                }
            }
        }

        lead++;
    }
}

//Computes the number of unique solutions
int solutions (Matrix m) {
    int row;
    int col;


    //pragma omp parallel for private(row,col) schedule(guided,8192) collapse(2)
    for ( row = 0; row<m->dim_y; row++) {
      int count = 0;
        for (col = 0; col<m->dim_x*bitLength; col++) {
            if (MtxGet(m,row,col)==1) {
	      //pragma omp atomic
	      if (count ==0) {
		count ++;
	      }
	      else {
		return -1;
	      }

            }
        }
    }

    return 0;
}


int main(int argc, char **argv)
{
  //omp_set_nested(1);

    if (argc!=6) {
        printf("Parameters: n c k rowsToRemove logOfRuns\nFor k=5, c>1.1243.\n");
        exit(1);
    }
    if (sodium_init() == -1) {
        return 1;
    }

    int n = atoi(argv[1]);

    if ((n % 256)!=0) {
        printf("n must be divisible by 256.\n");
        exit(-1);
    }
    
    double c = atof(argv[2]);
    int m = ceil(c*(double)n);
    int k = atoi(argv[3]);

    int toRemove  = atoi(argv[4]);

    unsigned long runs = 1UL << atoi(argv[5]);

    printf("n=%d, c=%f, m=%d, k=%d, runs=%lu, rows removed: %d\n",n,c,m,k,runs, toRemove);

    Matrix *MOnes = (Matrix*) malloc(runs*sizeof(Matrix));

    unsigned long i;
    clock_t start = clock();
    double mystart = omp_get_wtime();
    unsigned long count = 0UL;
    unsigned long percentage = 0UL;
    unsigned long failures = 0UL;

    //unsigned long successes = 0UL;
    //reduction(+:failures,successes)
#pragma omp parallel for if(runs>1)
    for (i = 0UL; i<runs; i++) {
        MOnes[i] = NewMatrix(n, m);
        MOnes[i] = InitMatrix( MOnes[i], k);
        RemoveRandomRows(MOnes[i], toRemove);
        MtxToReducedREForm(MOnes[i]);
        if (solutions(MOnes[i]) == -1) {
	  	#pragma omp atomic
            failures ++;
        } 

	#pragma omp atomic
        count ++;

	#pragma omp critical
        {
            if ( 100UL*count/runs > percentage ) {
                percentage = 100UL*count/runs;
                printf("\rFull rank: %lu (%lu%%), failures: %lu (%lu%%),", count-failures, 100*(count-failures)/count, failures, 100*failures/count);
                printf(" number of runs done: %ld%%             ", percentage);
                fflush(stdout);
            }
        }


        //Malloc is thread save starting from Glibc 2.2 ragma omp critical
        {
            free(MOnes[i]->m_stor);
            free(MOnes[i]->mtx);
            free(MOnes[i]);
        }
    }
    double myend = omp_get_wtime();
    clock_t end = clock();

    free(MOnes);

    double elapsed_time = (end - start)/(double)CLOCKS_PER_SEC;

    printf("\nFull rank: %lu (%lu%%), failures: %lu (%lu%%)\n", runs-failures, 100*(runs-failures)/runs, failures, 100*failures/runs);

    printf("Total CPU time  (sum of all threads): %.2fs, ", elapsed_time);
    printf("wall clock time: %.2fs (%.2fs per run)\n",myend-mystart,(myend-mystart)/(double)runs);


    return 0;
}
