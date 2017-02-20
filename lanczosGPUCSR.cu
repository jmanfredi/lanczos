#include "lanczosGPUCSR.h"
/*
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
*/

__global__ void sparseMVKernel(long* rows, long* cols, float* vals,float* vj,float* result,long dim,long nnz){
  //Matrix-vector multiplication kernel for sparse matrices (CSR format)

  int tx = threadIdx.x;
  //int ty = threadIdx.y;
  int bx = blockIdx.x;
  //int by = blockIdx.y;
  int dimx = blockDim.x;
  //int dimy = blockDim.y;

  int i = (bx*dimx) + tx;
  //  int j = (by*dimy) + ty;

  float sum;
  long rowStart;
  long rowEnd;

  //Check to see if the row of interest exists
  if (i<dim){
    rowStart = rows[i];
    rowEnd = rows[i+1];
    result[i]=0;
    //Initialize inner product
    sum = 0;

    //Loop over entries in row i and compute inner product
    for (int index=rowStart; index<rowEnd; index++){
      sum = sum + vals[index]*vj[cols[index]];
    }
    //Set final sum to the appropriate entry in the resulting vector
    result[i] = sum;
  }
}

/*
void allocateAndTransferMemory(long* rows, long* cols, float* vals, float* v, float* result, int dim, int nnz){
  //Allocate sufficient memory on the GPU and transfer from host to device

  long * rows_device;
  long * cols_device;
  float * vals_device;
  int sizeMatrix = sizeof(float)*dim*dim;
  cout << "**Allocating matrix with " << sizeMatrix << " elements**" << endl;
  cudaMalloc((void **) &rows_device, sizeof(long)*(dim+1));
  cudaMalloc((void **) &cols_device, sizeof(long)*nnz);
  cudaMalloc((void **) &vals_device, sizeof(float)*nnz);
  //Allocate input vector
  float * v_device;
  int sizeVector = sizeof(float)*dim;
  cudaMalloc((void **) &v_device, sizeVector);
  //Allocate result vector
  float * result_device;
  cudaMalloc((void **) &result_device, sizeVector);

  //STEP 2: TRANSFER
  //Transfer matrix
  cudaMemcpy(rows_device, rows, sizeof(long)*(dim+1), cudaMemcpyHostToDevice);
  cudaMemcpy(cols_device, cols, sizeof(long)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(vals_device, vals, sizeof(float)*nnz, cudaMemcpyHostToDevice);
  //Transfer vector
  cudaMemcpy(v_device, v, sizeVector, cudaMemcpyHostToDevice);


}
*/
/*
long* allocateAndTransferMemory(long* dataHost, int size){
  //Function that takes as input two pointers (one to an array of data on the host CPU
  //and another to an array of data on the device) and one integer that should be the 
  //size of both arrays. The function allocates memory on the GPU and then transfers
  //dataHost to dataDevice. Ultimately, the function returns the pointer dataDevice.

  long* dataDevice;

  //Allocate on GPU
  cudaMalloc((void **) &dataDevice, sizeof(long)*size);
  
  //Transfer to from host to GPU/device
  cudaMemcpy(dataDevice, dataHost, sizeof(long)*size, cudaMemcpyHostToDevice);

  return dataDevice;

}

long* transferAndFreeMemory(long *dataDevice, int size){
  //Function that takes as input a pointer to an array on the device, and returns
  //a pointer to the transferred result on the host. The function also frees
  //the GPU memory.

  long dataHost[size];

  //Transfer from GPU/device to host
  cudaMemcpy(dataHost, dataDevice, size, cudaMemcpyDeviceToHost);

  //Free memory from device
  cudaFree(dataDevice);

  return dataHost;
}
*/
/*
void matrixMemSetup(long* rows, long* cols, float* vals, int dim, int nnz){
  //This function allocates and transfers the memory related to the sparse matrix. This
  //matrix is invariant throughout the entire calculation, so this function needs only
  //to be run once at the start of the calculation.
}
*/

void sparseMV(long* rows, long*cols, float* vals, float* v, float* result, int dim, int nnz){
  //Wrapper function for calling sparseMVKernal

  //STEP 1: ALLOCATE
  //Allocate rows,cols, and vals
  long * rows_device;
  long * cols_device;
  float * vals_device;
  int sizeMatrix = sizeof(float)*dim*dim;
  cout << "**Allocating matrix with " << sizeMatrix << " elements**" << endl;
  cudaMalloc((void **) &rows_device, sizeof(long)*(dim+1));
  cudaMalloc((void **) &cols_device, sizeof(long)*nnz);
  cudaMalloc((void **) &vals_device, sizeof(float)*nnz);
  //Allocate input vector
  float * v_device;
  int sizeVector = sizeof(float)*dim;
  cudaMalloc((void **) &v_device, sizeVector);
  //Allocate result vector
  float * result_device;
  cudaMalloc((void **) &result_device, sizeVector);

  //STEP 2: TRANSFER
  //Transfer matrix
  cudaMemcpy(rows_device, rows, sizeof(long)*(dim+1), cudaMemcpyHostToDevice);
  cudaMemcpy(cols_device, cols, sizeof(long)*nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(vals_device, vals, sizeof(float)*nnz, cudaMemcpyHostToDevice);
  //Transfer vector
  cudaMemcpy(v_device, v, sizeVector, cudaMemcpyHostToDevice);

  //STEP 3: SET UP
  //  dim3 blockSize(ceil(MATRIXDIM/TILEWIDTH)+1,1,1);
  dim3 blockSize(dim,1,1);
  dim3 gridSize(1,1,1);

  //STEP 4: RUN KERNEL
  sparseMVKernel<<<gridSize, blockSize>>>(rows_device,cols_device,vals_device,
					  v_device,result_device,dim,nnz);

  //STEP 5: TRANSFER
  //Transfer result vector
  cudaMemcpy(result, result_device, sizeVector, cudaMemcpyDeviceToHost);
  //  cout << "--------------------" << endl;

  //Step 6: Free memory on device
  cudaFree(rows_device);
  cudaFree(cols_device);
  cudaFree(vals_device);
  cudaFree(v_device);
  cudaFree(result_device);

  return;
}


/*

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

    //  This is how one can screen matrix types if their application 
    //only supports a subset of the Matrix Market data types.      

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode) )
    {
      printf("Sorry, this application does not support ");
      printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
      exit(1);
    }

  // find out size of sparse matrix .... 

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
    exit(1);

  //Check if matrix is square...it should be!
  if (M!=N){
    cout << "WARNING! Matrix is not square: " << M << " by " << N<< endl;
  }

  // NOTE: when reading in doubles, ANSI C requires the use of the "l"  
  //   specifier as in "%lg", "%lf", "%le", otherwise errors will occur 
  //  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            

  for (long i=0; i<nz; i++)
    {
      //      fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);      
      fscanf(f, "%li %li %f\n", &rows[i], &cols[i], &vals[i]);
      
      //if (i%1000 == 0){
	//cout << "READING " << rows[i] << " " << cols[i] << " "<<vals[i] << endl;
      //}
      
      //Below not fully tested...just stick with simple case for now
      //If the file only has two values (the row and column) then assume
      //that the value is 1.
      
      //if (fscanf(f, "%d %d %g\n", &rows[i], &cols[i], &vals[i]) != 3){
	//vals[i] = 1;
      //}
      
	//      I[i]--;  // adjust from 1-based to 0-based 
      rows[i]--;  // adjust from 1-based to 0-based 
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

  //  This is how one can screen matrix types if their application 
  //  only supports a subset of the Matrix Market data types.      

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode) )
    {
      printf("Sorry, this application does not support ");
      printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
      exit(1);
    }

  // find out size of sparse matrix .... 

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
*/

void lanczosCSR(long* rows,long* cols, float* vals,float* randomVec,long dim, long nnz,float precision,int eigenNum,int maxKrylov){
  //This is a simple implementation of tha LA using CSR sparse matrix vector multiplication. For details, see the Wikipedia
  //article about the Lanczos Algorithm


  /*
  for (int k=0;k<dim+1;k++){
    cout << "ROWSCSR is " << rows[k] << endl;
  }
  */

  //Allocate space
  float wjPrime[dim];
  float wj[dim];
  float alphaj;
  float betaj=0;
  float betajPlus1=0;
  float vjMinus1[dim];
  float vjPlus1[dim];
  
  float alphaArray[maxKrylov];
  float betaArray[maxKrylov];
  betaArray[0] = 0;

  cout << "**Initializing vectors**" << endl;
  float vj[dim];

  //Initialize vj to randomVec
  for (long i=0;i<dim;i++){
    vj[i] = randomVec[i];
  }

  //Initialize vectors
  for (long i=0; i<dim; i++){
    wjPrime[i]=0;
    wj[i]=0;
    vjMinus1[i]=0;
    vjPlus1[i]=0;
  }

  //test FOR ALLOCATING MEMORY OF MATRIX
  //  allocateAndTransferRows(rows,dim+1);//Rows
  //  allocateAndTransferCols(cols,nnz);//Cols
  //  allocateAndTransferVals(vals,nnz);//Vals

  long* rowsDevice;
  cudaMalloc((void **) &rowsDevice, sizeof(long)*(dim+1));
  cudaMemcpy(rowsDevice, rows, sizeof(long)*(dim+1), cudaMemcpyHostToDevice);

  long* colsDevice;
  cudaMalloc((void **) &colsDevice, sizeof(long)*nnz);
  cudaMemcpy(colsDevice, cols, sizeof(long)*nnz, cudaMemcpyHostToDevice);

  float* valsDevice;
  cudaMalloc((void **) &valsDevice, sizeof(float)*nnz);
  cudaMemcpy(valsDevice, vals, sizeof(float)*nnz, cudaMemcpyHostToDevice);

  cout << "**Initializing Q matrix of size " << dim<< " by " << maxKrylov<< endl;
  float *Q;
  Q = (float *)malloc(sizeof(float)*dim*maxKrylov);

  std::clock_t start;
  long double duration;
  start = std::clock();

  cout << "**Starting Krylov loop (and timer)**" << endl;
  //Loop over the maxKrylov
  for (int j=0; j<maxKrylov; j++){
    //Start Krylov loop

    //    cout << "Iteration number " << j << endl;

    //******************************************************
    //Do a matrix vector multiplication: A times vj equals wjPrime
    //CSRmatrixVectorProduct(rows,cols,vals,vj,wjPrime,dim);
    //sparseMV(rows,cols,vals,vj,wjPrime,dim,nnz);
    
    //Allocate memory for vector
    //    vjDevice = allocateAndTransferVector();//Vector
    //    wjPrimeDevice = allocateAndTransferResult();//Result
    float* vjDevice;
    cudaMalloc((void**) &vjDevice, sizeof(float)*dim);
    cudaMemcpy(vjDevice, vj, sizeof(float)*dim, cudaMemcpyHostToDevice);

    float* wjPrimeDevice;
    cudaMalloc((void **) &wjPrimeDevice, sizeof(float)*dim);

    //    sparseMVTest(rowsDevice,colsDevice,valsDevice,vjDevice,wjPrimeDevice,dim,nnz);

    dim3 blockSize(dim,1,1);
    dim3 gridSize(1,1,1);
    sparseMVKernel<<<gridSize,blockSize>>>(rowsDevice,colsDevice,valsDevice,
					   vjDevice,wjPrimeDevice,dim,nnz);

    //Transfer result
    cudaMemcpy(wjPrime,wjPrimeDevice, sizeof(float)*dim, cudaMemcpyDeviceToHost);

    //Free memory
    cudaFree(vjDevice);
    cudaFree(wjPrimeDevice);

    //According to T&B, the following implementation of Lanczos has better stability properties
    //than the one above. All I do is subtract betaj*vjMinus1 from the result of the matrix-vector
    //product above. NOT TESTED
    /*
    for (int i=0; i<8; i++){
      wjPrime[i] = wjPrime[i] - betaj*vjMinus1[i];
    }
    */

    
    //Calculate dot product alphaj of wjPrime and vj
    alphaj = dotProduct(wjPrime,vj,dim);
    alphaArray[j] = alphaj;

    //Calculate wj = wjPrime - alphaj*vj - betaj*vjMinus1 element by element
    for (long i=0; i<dim; i++){
      wj[i] = wjPrime[i] - alphaj*vj[i] - betaj*vjMinus1[i];
    }

    //Calculate betajPlus1 = norm(wj)
    float normWjSquared=0;
    for (long i=0; i<dim; i++){
      normWjSquared += wj[i]*wj[i];
    }
    betajPlus1 = sqrt(normWjSquared);
    if (j<maxKrylov-1){betaArray[j+1] = betajPlus1;}

    //Set the next vj
    for (long i=0; i<dim; i++){
      vjPlus1[i] = wj[i]/betajPlus1;
    }

    //Orthogonality check
    //    cout << "Check orthogonality between vj and vjPlus1: "<<dotProduct(vj,vjPlus1,dim) << endl;


    //Write the vj's to a matrix (use to test tridiagonality)
    for (long i=0; i<dim; i++){
      //Write elements of vj to a column of Q (fixed j, vary i)
      //Q[i][j] = vj[i];
      Q[i*maxKrylov + j] = vj[i];
    }
    

    //Assign vectors for the next iteration
    for (long i=0; i<dim; i++){
      vjMinus1[i] = vj[i];
      vj[i] = vjPlus1[i];
    }
    betaj = betajPlus1;
    
    
  }//end loop over Krylov
    
  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  
  //  float Qstar[maxKrylov][dim];
  float* Qstar;
  Qstar = (float *)malloc(sizeof(float)*maxKrylov*dim);

  for (int i=0;i<maxKrylov;i++){
    for (long j=0; j<dim;j++){
      //      Qstar[i][j] = Q[j][i];
      Qstar[i*dim + j] = Q[j*maxKrylov + i];
    }
  }
  
  
  //cout << "This is Q: " << endl;
  //printGridSmart(Q,dim,maxKrylov);
  //cout << " ***********************" << endl;
  //cout << "This is Q*" << endl;
  //printGridSmart(Qstar,maxKrylov,dim);

  //cout << " ***********************" << endl;
  //float* identityCheck;
  //identityCheck = (float *)malloc(sizeof(float)*ITERATION*ITERATION);

  //matrixProductSmart(Qstar,maxKrylov,dim,Q,dim,maxKrylov,
  //		     identityCheck,maxKrylov,maxKrylov);

  //cout << "This is identity check" << endl;
  //printGridSmart(identityCheck,ITERATION,ITERATION);
  //cout << " ***********************" << endl;
  //float* QstarA;
  //QstarA = (float *)malloc(sizeof(float)*maxKrylov*dim);

  //matrixProductSmart(Qstar,maxKrylov,dim,A,dim,dim,QstarA,maxKrylov,dim);
  
  //float* QstarAQ;
  //QstarAQ = (float *)malloc(sizeof(float)*maxKrylov*maxKrylov);

  //matrixProductSmart(QstarA,maxKrylov,dim,Q,dim,maxKrylov,QstarAQ,maxKrylov,maxKrylov);
  //cout << "This is QstarAQ" << endl;
  //printGridSmart(QstarAQ,maxKrylov,maxKrylov);
  
  
  cout << "Duration of Krylov loop: " << duration << endl;

  //  printVecSmart(alphaArray,maxKrylov);
  //  printVecSmart(betaArray,maxKrylov);

}


/*
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
*/
/*
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
	//      	matResult[i*yResult + j] += 1;
      }
    }
  }

  return;

}
*/
/*
void matrixVectorProduct(float* A, float* vec, float* result, long dim){

  for (long i=0; i<dim; i++){
    result[i]=0;
    //cout << "FOR ELEMENT " << i << endl;
    for (long k=0; k<dim; k++){

      result[i]+=vec[k]*A[i*dim + k];	
      //	cout << " A element is " << A[i*dim + k] << endl;
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
*/

int main()
{
  //Initialize filename
  char filename[100] = "../matrices/b1_ss/b1_ss.mtx";
  //char filename[100] = "../matrices/SmallW/SmallW.mtx";
  //  char filename[100] = "../matrices/M80PI_n1/M80PI_n1.mtx";
  //char filename[100] = "../matrices/bips07_2476/bips07_2476.mtx";
  //char filename[100] = "../matrices/copter2/copter2.mtx";
  //char filename[100] = "../matrices/atmosmodd/atmosmodd.mtx";
  //char filename[100] = "../matrices/circuit5M/circuit5M.mtx";
  //char filename[100] = "../matrices/nlpkkt240/nlpkkt240.mtx";  

  long nnz=0;
  long dim=0;

  //Read matrix dimension and number of nonzero values
  cout <<"**Reading matrix info**"<<endl;
  readMatrixInfo(filename,&dim,&nnz);
  cout << "nnz " << nnz << " and dim " << dim << endl;

  //Initialize sparse matrix vectors
  //COO
  cout <<"**Initializing COO vectors**"<<endl;
  long* rowsCOO;
  rowsCOO = (long *)malloc(nnz * sizeof(long));
  long* colsCOO;
  colsCOO = (long *)malloc(nnz * sizeof(long));
  float* valsCOO;
  valsCOO = (float *)malloc(nnz * sizeof(float));

  //CSR
  //rowsCSR is size dim+1
  //colsCSR is size nnz
  //valsCSR is size nnz
  cout <<"**Initializing CSR vectors**"<<endl;
  long* rowsCSR;
  rowsCSR = (long *)malloc((dim+1) * sizeof(long));
  long* colsCSR;
  colsCSR = (long *)malloc(nnz * sizeof(long));
  float* valsCSR;
  valsCSR = (float *)malloc(nnz * sizeof(float));

  //Read matrix from file
  cout <<"**Read Matrix from file**"<<endl;
  readMatrixFromFile(rowsCOO,colsCOO,valsCOO,filename);

  //Check number of nonzero values
  if (nnz==0 || dim==0){
    cout<< "Either matrix read wrong or some other problem. Returning" << endl;
    return 0;
  }

  
  //  cout << "Here is A" << endl;
  //printGrid(our_array);
  //  printGridSmart(A,dim,dim);

  //  int matrixRows = sizeof(our_array)/sizeof(our_array[0]);
  //  int matrixCols = sizeof(our_array[0])/sizeof(int);

  //  cout << "Matrix rows " << matrixRows << endl;
  //  cout << "Matrix cols " << matrixCols << endl;

  cout <<"**Initializing random vector**"<<endl;
  float randomVec[dim];
  float norm=0;
  for (long i=0;i<dim;i++)
    {
      randomVec[i] = (float)rand()/RAND_MAX;
      norm += randomVec[i]*randomVec[i];
    }

  float newNorm=0;
  for (long i=0;i<dim;i++)
    {
      randomVec[i] = randomVec[i]/sqrt(norm);
      newNorm += randomVec[i]*randomVec[i];
    }

  //Test that I am reading in matrix properly
  //printCOOMatrix(rowsCOO,colsCOO,valsCOO,nnz,dim);

  //Convert matrix to CSR form
  cout <<"**Convert matrix from COO to CSR**"<<endl;
  convertFromCOO_ToCSR(rowsCOO,colsCOO,valsCOO,dim,nnz,rowsCSR,colsCSR,valsCSR);

  int iterationNum = min(max(10,(int)ceil(dim*0.1)),100);
  //  int iterationNum = 5;
  cout << "**Iteration Number: " <<iterationNum<<"**"<<endl;
  //Lanczos CSR method for CPU
  lanczosCSR(rowsCSR,colsCSR,valsCSR,randomVec,dim,nnz,0.75,10,iterationNum);
  
  cout << "**Computation complete** " << endl;

  /*
  //TEST
  //Lanczos CSR method
  float testVec[dim];
  for (int i=0; i<dim; i++){
    testVec[i]=1;
  }

  float resultVec[dim];
  CSRmatrixVectorProduct(rowsCSR,colsCSR,valsCSR,testVec,resultVec,dim);
  printVecSmart(resultVec,dim);
  */
}


