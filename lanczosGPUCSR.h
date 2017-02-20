#include "cuda.h"
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
//INFINITE SHAME on me for using the line below...but I was getting undefined reference errors using "#include mmio.h"
//and trying to compile mmio separately and link to this. And I am working with a limited time frame.
#include "mmio.c"

using namespace std;
float dotProduct(float* vec1, float* vec2, long dim);
float vectorNorm(float* vec, long dim);

void matrixProductSmart(float *mat1,long xDim1,long yDim1,float *mat2,long xDim2,long yDim2,float *matResult,long xResult,long yResult);
void matrixVectorProduct(float *A,float* vec, float* result, long dim);
void CSRmatrixVectorProduct(long* rows, long* cols, float* vals,float* vec, float* result, long dim);


float dotProduct(float* vec1, float* vec2, long dim){
  //Calculate dot product of two vectors
  //int dim1 = sizeof(vec1)/sizeof(vec1[0]);
  //int dim2 = sizeof(vec2)/sizeof(vec2[0]);
  //if (dim1 != dim2) {
  //  cout << "DIMENSIONS DONT MATCH FOR DOT PRODUCT!!!" << endl;
  //  return -1;
  //}



  //cout << "dim in dotProduct is " << dim1 << endl;
  float product = 0;
  for (long i=0; i< dim; i++){

    product += vec1[i]*vec2[i];
    //cout << "Product is " << product << endl;

  }

  return product;

}


float vectorNorm(float* vec, long dim){
  //Calculate the norm of a vector
  //  int dim = sizeof(vec)/sizeof(vec[0]);
  //  cout << "dim is " << dim << endl;
  float normSquared = 0;
  for (long i=0; i<dim; i++){
    normSquared += vec[i]*vec[i];
  }

  return sqrt(normSquared);

}


void matrixProductSmart(float* mat1, long xDim1, long yDim1, float* mat2, long xDim2, long yDim2,
			float* matResult, long xResult, long yResult){

  //Check dimensions
  if (xResult != xDim1 || xDim2 != yDim1){
    cout << "ILLEGAL X DIMENSION!" << endl;
  }
  if (yResult != yDim2){
    cout << "ILLEGAL Y DIMENSION!" << endl;
  }

  for (long i=0;i<xResult;i++){
    for (long j=0; j<yResult; j++){
      matResult[i*yResult + j] = 0;
      //matResult[i*yResult + j] = i*yResult+j;
      for (long k=0;k<yDim1;k++){
	matResult[i*yResult + j] += mat1[i*yDim1+k]*mat2[k*yDim2+j];
        //              matResult[i*yResult + j] += 1;
      }
    }
  }

  return;

}


void printGridSmart(float *array, long xDim, long yDim){

  for (long i=0; i<xDim; i++){
    for (long j=0; j<yDim; j++){
      cout << array[i*yDim + j] << " ";
    }
    cout << endl;
  }

}

void printCOOMatrix(long* rows, long* cols, float* vals, long nnz, long dim){

  float marker[dim][dim];
  for (long i=0; i<dim; i++){
    for (long j=0; j<dim; j++){
      marker[i][j]=0;
    }
  }

  for (long z=0;z<nnz;z++){
    marker[rows[z]][cols[z]] = vals[z];
  }

  for (long i=0; i<dim; i++){
    for (long j=0; j<dim; j++){
      cout << marker[i][j] << " ";
    }
    cout << endl;
  }

}

void printVecSmart(float* vec, long dim){
  for (long i=0; i<dim; i++){
    cout << vec[i] << " ";
  }
  cout << endl;
}


void readMatrixFromFile(long* rows, long* cols, float* vals, char* filename){
  //Read matrix file in MatrixMarket format. Code comes mostly from example_read.c from
  //MM website.

  int ret_code;
  MM_typecode matcode;
  FILE *f;
  long M, N, nz;
  //  int M, N;
  //int i, *I, *J;
  //  double *val;


  //Open file
  if((f = fopen(filename, "r")) == NULL){
    cout << "CANNOT FIND FILE!" << endl;
    exit(1);
  }

  //Look at banner code
  if (mm_read_banner(f, &matcode) != 0)
    {
      printf("Could not process Matrix Market banner.\n");
      exit(1);
    }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode) )
    {
      printf("Sorry, this application does not support ");
      printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
      exit(1);
    }

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
    exit(1);

  //Check if matrix is square...it should be!
  if (M!=N){
    cout << "WARNING! Matrix is not square: " << M << " by " << N<< endl;
  }

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (long i=0; i<nz; i++)
    {
      //      fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
      fscanf(f, "%li %li %f\n", &rows[i], &cols[i], &vals[i]);
      /*
      if (i%1000 == 0){
        cout << "READING " << rows[i] << " " << cols[i] << " "<<vals[i] << endl;
      }
      */
      //Below not fully tested...just stick with simple case for now
      //If the file only has two values (the row and column) then assume
      //that the value is 1.
      /*
      if (fscanf(f, "%d %d %g\n", &rows[i], &cols[i], &vals[i]) != 3){
        vals[i] = 1;
      }
      */
      //      I[i]--;  /* adjust from 1-based to 0-based */
      rows[i]--;  /* adjust from 1-based to 0-based */
      //      J[i]--;
      cols[i]--;
    }

  fclose(f);

  return;

}


void readMatrixInfo(char* filename, long* dim, long* nnz){
  //Read dim and nnz from matrix file (nothing else)
  //This info is used to allocate space for the matrix before actually reading it in

  int ret_code;
  MM_typecode matcode;
  FILE *f;
  long M, N, nz;
  //  int M, N;
  //int i, *I, *J;
  //  double *val;

  if((f = fopen(filename, "r")) == NULL){
    cout << "CANNOT FIND FILE!" << endl;
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0)
    {
      printf("Could not process Matrix Market banner.\n");
      exit(1);
    }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode) )
    {
      printf("Sorry, this application does not support ");
      printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
      exit(1);
    }

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
    exit(1);

  if (M!=N){
    cout << "WARNING! Matrix is not square: " << M << " by " << N<< endl;
  }
  *dim = M;
  *nnz = nz;

}


void convertFromCOO_ToCSR(long* rowsCOO, long* colsCOO, float* valsCOO, long dim, long nnz,
			  long* rowsCSR, long* colsCSR, float* valsCSR){
  //Based on scipy sparsetools coo_tocsr function
  //rowsCSR is size dim+1
  //colsCSR is size nnz
  //valsCSR is size nnz

  std::fill(rowsCSR,rowsCSR + dim, 0);
  //Compute number of nonzero entries per row of matrix
  for (long n=0; n<nnz; n++){
    rowsCSR[rowsCOO[n]]++;
  }

  //Take cumulative sum to get rowsCSR
  long sum = 0;
  for (long i=0; i<dim; i++){
    long temp = rowsCSR[i];
    rowsCSR[i] = sum;
    sum += temp;
  }
  rowsCSR[dim] = nnz;

  //Write  colsCOO and valsCOO into colsCSR and valsCSR
  for (long n=0; n<nnz; n++){
    long row = rowsCOO[n];
    long dest = rowsCSR[row];

    colsCSR[dest] = colsCOO[n];
    valsCSR[dest] = valsCOO[n];

    rowsCSR[row]++;
  }

  long last = 0;
  for (long i=0; i<= dim; i++){
    long temp = rowsCSR[i];
    rowsCSR[i] = last;
    last = temp;
  }
  
  //Now I have the matrix in CSR format in rowsCSR, colsCSR, and valsCSR.
}


void matrixVectorProduct(float* A, float* vec, float* result, long dim){

  for (long i=0; i<dim; i++){
    result[i]=0;
    //cout << "FOR ELEMENT " << i << endl;
    for (long k=0; k<dim; k++){

      result[i]+=vec[k]*A[i*dim + k];
      //        cout << " A element is " << A[i*dim + k] << endl;
    }
    //    cout << "Result is " << result[i] << endl;
  }

}

void CSRmatrixVectorProduct(long* rows, long* cols, float* vals, float* vec, float* result, long dim){
  //Do a matrix vector product using CSR representation
  //rows is an array of row pointers (size dim+1)
  //cols is an array of column indices (size nnz)
  //vals is an array of values (size nnz)
  //vec is input vector, result is result vector (both size dim)

  //Loop over number of rows of matrix
  for (long i=0;i<dim;i++){
    result[i]=0;
    //Loop across row pointers
    for (long k=rows[i]; k<rows[i+1];k++){
      //cout << "Start at " << rows[i]<< " and end at " << rows[i+1] << endl;
      //Fill result vector
      result[i] = result[i] + vals[k]*vec[cols[k]];
      //result[i] = result[i] + 1;

    }
  }

}
