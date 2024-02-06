# LightGBM

## Code example of calling LGBM_BoosterPredictForCSRSingleRowFast
```c
#include <lightgbm/c_api.h>
#include <stdlib.h>

int main() {
    // Assume we have a trained model and a FastConfigHandle
    FastConfigHandle fastConfig_handle;

    // Define a single row of data in CSR format
    int indptr_type = C_API_DTYPE_INT32;
    int64_t nindptr = 2;
    int64_t nelem = 3;

    int* indptr = (int*)malloc(nindptr * sizeof(int));
    indptr[0] = 0;
    indptr[1] = nelem;

    int32_t* indices = (int32_t*)malloc(nelem * sizeof(int32_t));
    indices[0] = 0;
    indices[1] = 2;
    indices[2] = 4;

    double* data = (double*)malloc(nelem * sizeof(double));
    data[0] = 1.0;
    data[1] = 3.0;
    data[2] = 5.0;

    // Output variables
    int64_t out_len;
    double* out_result = (double*)malloc(sizeof(double));

    // Call the function
    int result = LGBM_BoosterPredictForCSRSingleRowFast(fastConfig_handle, indptr, indptr_type, indices, data, nindptr, nelem, &out_len, out_result);

    // Check the result
    if (result == 0) {
        printf("Prediction successful, output length: %lld\n", out_len);
    } else {
        printf("Prediction failed, error code: %d\n", result);
    }

    // Free allocated memory
    free(indptr);
    free(indices);
    free(data);
    free(out_result);

    return 0;
}
```

In this example, we're creating a single row of data with three non-zero elements. The `indptr` array defines the start and end of the row in the `indices` and `data` arrays. The `indices` array contains the column indices of the non-zero elements, and the `data` array contains the values of the non-zero elements.

Please replace the placeholders with your actual data and handle. Also, make sure to handle the memory correctly, especially for the `out_result` pointer. This is just a basic example and does not include error checking or memory management. You should add those in your actual code. 

Remember to link against the LightGBM library when compiling this code. If you're using gcc, you can do this with `-lLightGBM`. If you're using Visual Studio, you'll need to add the LightGBM library to your project settings. 



## Training code snipet
python 

c


## CSR(Compressed Sparsed Row) format
It uses three one dimensional arrays to store non-zero values.
This is efficient for sparse matrixes where most of the elements 
are zero.


In the context of a Compressed Sparse Row (CSR) matrix, `indptr` and `indices` are arrays used to represent sparse data. Here's what they mean:

- `indptr`: This is short for "index pointer". It is an integer array of length `n_rows + 1` where `n_rows` is the number of rows in your matrix. The `indptr` array defines the index ranges for rows of the matrix. If `indptr[i] = x` and `indptr[i+1] = y`, this means that the data for row `i` in the matrix is stored in `indices[x:y]` and `data[x:y]`.

- `indices`: This is an integer array that contains column indices of non-zero elements. The column indices for row `i` are stored in `indices[indptr[i]:indptr[i+1]]`.

Here's a simple example:

```python
import numpy as np
from scipy.sparse import csr_matrix

# Create a CSR matrix
data = np.array([1, 2, 3, 4, 5, 6])
indices = np.array([0, 2, 2, 0, 1, 2])
indptr = np.array([0, 2, 3, 6])
csr = csr_matrix((data, indices, indptr), shape=(3, 3))

print(csr.toarray())
```

This will output:

```python
[[1 0 2]
 [0 0 3]
 [4 5 6]]
```

In this matrix, the non-zero elements of the first row are `1` (at column `0`) and `2` (at column `2`), which corresponds to `indices[0:2]` and `data[0:2]`. The non-zero elements of the second row are `3` (at column `2`), which corresponds to `indices[2:3]` and `data[2:3]`. The non-zero elements of the third row are `4` (at column `0`), `5` (at column `1`), and `6` (at column `2`), which corresponds to `indices[3:6]` and `data[3:6]`.

I hope this helps! Let me know if you have any other questions. 😊


