#include "lanczosGPUCSR.h"

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

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
    //result[i]=0;
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

__global__ void csrVectorKernel(long* Ap, long* Aj, float* Ax, float* x, float* y,long num_rows, long nnz){
  //Matrix-vector multiplication kernel using CSR-vector
  
  const int BLOCK_SIZE = 256;
  const int WARP_SIZE = 32;
  const bool UseCache = false;//Texture memory does not make sense to use for Lanczos

  //Set up shared memory for sums
  __shared__ float sdata[BLOCK_SIZE];
  __shared__ long ptrs[BLOCK_SIZE/WARP_SIZE][2];

  long thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x; //global thread index
  long thread_lane = thread_id & (WARP_SIZE - 1);  //Lane corresponds to Warp local thread ID (that is, which coalesced memory bunch)
  long warp_id = thread_id / WARP_SIZE;  //Warp ID corresponds to one row
  long warp_lane = threadIdx.x / WARP_SIZE;
  long num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;

  for (long row = warp_id; row < num_rows; row+=num_warps){
    //use two threads to get Ap[row] and Ap[row+1] for memory considerations
    if (thread_lane<2)
      ptrs[warp_lane][thread_lane] = Ap[row+thread_lane];
    long row_start = ptrs[warp_lane][0];
    long row_end = ptrs[warp_lane][1];

    //compute local sum
    sdata[threadIdx.x] = 0;
    for (long jj = row_start + thread_lane; jj < row_end; jj+= WARP_SIZE)
      sdata[threadIdx.x] += Ax[jj] * fetch_x<UseCache>(Aj[jj],x);

    //Assume warp size 32, reduce local sums to row sums
    if (thread_lane < 16) { sdata[threadIdx.x] += sdata[threadIdx.x + 16]; __syncthreads(); }
    if (thread_lane < 8) { sdata[threadIdx.x] += sdata[threadIdx.x + 8]; __syncthreads(); }
    if (thread_lane < 4) { sdata[threadIdx.x] += sdata[threadIdx.x + 4]; __syncthreads(); }
    if (thread_lane < 2) { sdata[threadIdx.x] += sdata[threadIdx.x + 2]; __syncthreads(); }
    if (thread_lane < 1) { sdata[threadIdx.x] += sdata[threadIdx.x + 1]; __syncthreads(); }

    //first thread writes warp result
    if (thread_lane == 0)
      y[row] += sdata[threadIdx.x];
  }

}

  /*
  //Check if row number is good
  if (row < dim){
    int row_start = rows[row];
    int row_end = rows[row+1];

    //Compute running sum
    sums[threadIdx.x] = 0;
    __syncthreads();

    //Run the loop
    for (int jj = row_start + lane; jj < row_end; jj += WARP_SIZE){
      sums[threadIdx.x] += vals[jj] * vj[cols[jj]];
      __syncthreads();
    }

    //Parallel reduction in shared memory to combine into one sum
    for (int d = WARP_SIZE >> 1; d >= 1; d>>=1){
      if (lane < d) sums[threadIdx.x] += sums[threadIdx.x + d];
      __syncthreads();
    }

    //First thread in the warp writes the result
    if (lane = 0){
      result[row] += sums[threadIdx.x];
    }

  }

}
  */

void lanczosCSR(long* rows,long* cols, float* vals,float* randomVec,long dim, long nnz,float precision,int eigenNum,int maxKrylov){
  //This is a simple implementation of tha LA using CSR sparse matrix vector multiplication. For details
  //on the algorithm itself, see the Wikipedia article about the Lanczos Algorithm
  //Also, I implement a simple reorthogonalization using Gram Schmidt.

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

  cout << "Allocating row array" << endl;
  long* rowsDevice;
  gpuErrCheck( cudaMalloc((void **) &rowsDevice, sizeof(long)*(dim+1)) );
  gpuErrCheck( cudaMemcpy(rowsDevice, rows, sizeof(long)*(dim+1), cudaMemcpyHostToDevice) );

  cout << "Allocating column array" << endl;
  long* colsDevice;
  gpuErrCheck( cudaMalloc((void **) &colsDevice, sizeof(long)*nnz) );
  gpuErrCheck( cudaMemcpy(colsDevice, cols, sizeof(long)*nnz, cudaMemcpyHostToDevice) );

  cout << "Allocating value array (" << sizeof(float)*nnz << " bytes)" << endl;
  float* valsDevice;
  gpuErrCheck( cudaMalloc((void **) &valsDevice, sizeof(float)*nnz) );
  gpuErrCheck( cudaMemcpy(valsDevice, vals, sizeof(float)*nnz, cudaMemcpyHostToDevice) );

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

    if (j%10==0) cout << "Iteration number " << j << endl;

    //******************************************************
    //Do a matrix vector multiplication: A times vj equals wjPrime
    //CSRmatrixVectorProduct(rows,cols,vals,vj,wjPrime,dim);
    //sparseMV(rows,cols,vals,vj,wjPrime,dim,nnz);
    
    //Allocate memory for vector
    //    vjDevice = allocateAndTransferVector();//Vector
    //    wjPrimeDevice = allocateAndTransferResult();//Result
    float* vjDevice;
    gpuErrCheck( cudaMalloc((void**) &vjDevice, sizeof(float)*dim) );
    gpuErrCheck( cudaMemcpy(vjDevice, vj, sizeof(float)*dim, cudaMemcpyHostToDevice) );

    float* wjPrimeDevice;
    gpuErrCheck( cudaMalloc((void **) &wjPrimeDevice, sizeof(float)*dim) );

    //    sparseMVTest(rowsDevice,colsDevice,valsDevice,vjDevice,wjPrimeDevice,dim,nnz);

    dim3 blockSize(1024,1,1);
    dim3 gridSize(1,1,1);
    int BLOCK_SIZE = 256;
    int NUM_BLOCKS = 120;
    //    int NUM_BLOCKS = ceil(dim/256);
    
    //CSR SCALER KERNEL
    //    sparseMVKernel<<<NUM_BLOCKS,BLOCK_SIZE>>>(rowsDevice,colsDevice,valsDevice,
    //					      vjDevice,wjPrimeDevice,dim,nnz);

    //CSR VECTOR KERNEL
    csrVectorKernel<<<NUM_BLOCKS,BLOCK_SIZE>>>(rowsDevice,colsDevice,valsDevice,
        				       vjDevice,wjPrimeDevice,dim,nnz);

    gpuErrCheck( cudaPeekAtLastError() );
    gpuErrCheck( cudaDeviceSynchronize() );

    //Transfer result
    gpuErrCheck( cudaMemcpy(wjPrime,wjPrimeDevice, sizeof(float)*dim, cudaMemcpyDeviceToHost) );

    //Free memory
    gpuErrCheck( cudaFree(vjDevice) );
    gpuErrCheck( cudaFree(wjPrimeDevice) );

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

    //Reorthogonalization
    
    for (int i = 0; i < j; i++)
      {
	//Get Krylov vector
	//	viennacl::vector_base<NumericT> q_j(Q.handle(), Q.size1(), j * Q.internal_size1(), 1);
	float qi[dim];
	for (int k=0; k<dim ; k++){
	  //Get ith vector from Q
	  qi[k] = Q[k*maxKrylov + i];
	}

	//Inner product of wj and Krylov vector
	//	NumericT inner_rq = viennacl::linalg::inner_prod(Aq, q_j);
	float orthoCorr = dotProduct(wj,qi,dim);

	//Subtract from wj
	//Aq -= inner_rq * q_j;

	for (int z=0; z<dim; z++){
	  wj[z] = wj[z] - orthoCorr*qi[z];
	}

      }//end reortho
    

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

    /*
    //Write the vj's to a matrix (use to test tridiagonality)
    for (long i=0; i<dim; i++){
      //Write elements of vj to a column of Q (fixed j, vary i)
      //Q[i][j] = vj[i];
      Q[i*maxKrylov + j] = vj[i];
    }
    */

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
  
  /*  
  cout << "This is Q: " << endl;
  printGridSmart(Q,dim,maxKrylov);
  cout << " ***********************" << endl;
  cout << "This is Q*" << endl;
  printGridSmart(Qstar,maxKrylov,dim);

  cout << " ***********************" << endl;
  float* identityCheck;
  identityCheck = (float *)malloc(sizeof(float)*maxKrylov*maxKrylov);

  matrixProductSmart(Qstar,maxKrylov,dim,Q,dim,maxKrylov,
  		     identityCheck,maxKrylov,maxKrylov);

  cout << "This is identity check" << endl;
  printGridSmart(identityCheck,ITERATION,ITERATION);
  cout << " ***********************" << endl;
  float* QstarA;
  QstarA = (float *)malloc(sizeof(float)*maxKrylov*dim);

  matrixProductSmart(Qstar,maxKrylov,dim,A,dim,dim,QstarA,maxKrylov,dim);
  
  float* QstarAQ;
  QstarAQ = (float *)malloc(sizeof(float)*maxKrylov*maxKrylov);

  matrixProductSmart(QstarA,maxKrylov,dim,Q,dim,maxKrylov,QstarAQ,maxKrylov,maxKrylov);
  cout << "This is QstarAQ" << endl;
  printGridSmart(QstarAQ,maxKrylov,maxKrylov);
  */
  
  cout << "Duration of Krylov loop: " << duration << endl;

  //  printVecSmart(alphaArray,maxKrylov);
  //  printVecSmart(betaArray,maxKrylov);

}


int main()
{
  //Initialize filename
  //char filename[100] = "../matrices/b1_ss/b1_ss.mtx";
  //  char filename[100] = "../matrices/SmallW/SmallW.mtx";
  //char filename[100] = "../matrices/M80PI_n1/M80PI_n1.mtx";
  //  char filename[100] = "../matrices/bips07_2476/bips07_2476.mtx";
  //char filename[100] = "../matrices/copter2/copter2.mtx";
  //char filename[100] = "../matrices/atmosmodd/atmosmodd.mtx";
  //char filename[100] = "../matrices/circuit5M/circuit5M.mtx";
  //char filename[100] = "../matrices/nlpkkt240/nlpkkt240.mtx";  

  //BellGarland
  //  char filename[100] = "../matrices/bellGarland/dense2.mtx";
  char filename[100] = "../matrices/bellGarland/webbase-1M.mtx";
  //char filename[100] = "../matrices/bellGarland/cop20k_A.mtx";
  //char filename[100] = "../matrices/bellGarland/mac_econ_fwd500.mtx";
  //char filename[100] = "../matrices/bellGarland/mc2depi.mtx";
  //char filename[100] = "../matrices/bellGarland/pdb1HYS.mtx";
  //char filename[100] = "../matrices/bellGarland/pwtk.mtx";
  //char filename[100] = "../matrices/bellGarland/qcd5_4.mtx";
  //char filename[100] = "../matrices/bellGarland/rma10.mtx";
  //char filename[100] = "../matrices/bellGarland/scircuit.mtx";
  //char filename[100] = "../matrices/bellGarland/shipsec1.mtx";
  //char filename[100] = "../matrices/bellGarland/cant.mtx";
  //char filename[100] = "../matrices/bellGarland/consph.mtx";

  long nnz=0;
  long dim=0;

  //Read matrix dimension and number of nonzero values
  cout <<"**Reading matrix info**"<<endl;
  readMatrixInfo(filename,&dim,&nnz);
  cout << "nnz " << nnz << " and dim " << dim << endl;
  cout << "Average nnz/row: " << nnz/dim << endl;

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


