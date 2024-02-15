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

## Source code of GBDT train()
```c
class ObjectiveFunction {
 public:
  /*! \brief virtual destructor */
  virtual ~ObjectiveFunction() {}

  /*!
  * \brief Initialize
  * \param metadata Label data
  * \param num_data Number of data
  */
  virtual void Init(const Metadata& metadata, data_size_t num_data) = 0;

  /*!
  * \brief calculating first order derivative of loss function
  * \param score prediction score in this round
  * \gradients Output gradients
  * \hessians Output hessians
  */
  virtual void GetGradients(const double* score,
    score_t* gradients, score_t* hessians) const = 0;


```

```c
void GBDT::Train(int snapshot_freq, const std::string& model_output_path) {
  Common::FunctionTimer fun_timer("GBDT::Train", global_timer);
  bool is_finished = false;
  auto start_time = std::chrono::steady_clock::now();
  for (int iter = 0; iter < config_->num_iterations && !is_finished; ++iter) {
    is_finished = TrainOneIter(nullptr, nullptr);
    if (!is_finished) {
      is_finished = EvalAndCheckEarlyStopping();
    }
    auto end_time = std::chrono::steady_clock::now();
    // output used time per iteration
    Log::Info("%f seconds elapsed, finished iteration %d", std::chrono::duration<double,
              std::milli>(end_time - start_time) * 1e-3, iter + 1);
    if (snapshot_freq > 0
        && (iter + 1) % snapshot_freq == 0) {
      std::string snapshot_out = model_output_path + ".snapshot_iter_" + std::to_string(iter + 1);
      SaveModelToFile(0, -1, config_->saved_feature_importance_type, snapshot_out.c_str());
    }
  }
}
    bool GBDT::TrainOneIter(const score_t* gradients, const score_t* hessians) {
      Common::FunctionTimer fun_timer("GBDT::TrainOneIter", global_timer);
      std::vector<double> init_scores(num_tree_per_iteration_, 0.0);
      // boosting first
      if (gradients == nullptr || hessians == nullptr) {
        for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
          init_scores[cur_tree_id] = BoostFromAverage(cur_tree_id, true);
        }
        Boosting();
        gradients = gradients_pointer_;
        hessians = hessians_pointer_;
      } else {
        // use customized objective function
        CHECK(objective_function_ == nullptr);
        if (data_sample_strategy_->IsHessianChange()) {
          // need to copy customized gradients when using GOSS
          int64_t total_size = static_cast<int64_t>(num_data_) * num_tree_per_iteration_;
          #pragma omp parallel for schedule(static)
          for (int64_t i = 0; i < total_size; ++i) {
            gradients_[i] = gradients[i];
            hessians_[i] = hessians[i];
          }
          CHECK_EQ(gradients_pointer_, gradients_.data());
          CHECK_EQ(hessians_pointer_, hessians_.data());
          gradients = gradients_pointer_;
          hessians = hessians_pointer_;
        }
      }

            void GBDT::Boosting() {
              Common::FunctionTimer fun_timer("GBDT::Boosting", global_timer);
              if (objective_function_ == nullptr) {
                Log::Fatal("No objective function provided");
              }
              // objective function will calculate gradients and hessians
              int64_t num_score = 0;
              objective_function_->
                GetGradients(GetTrainingScore(&num_score), gradients_pointer_, hessians_pointer_);
            }
              void GetGradients(const double* score, score_t* gradients, score_t* hessians) const override {
                if (!need_train_) {
                  return;
                }
                if (weights_ == nullptr) {
                  #pragma omp parallel for schedule(static)
                  for (data_size_t i = 0; i < num_data_; ++i) {
                    // get label and label weights
                    const int is_pos = is_pos_(label_[i]);
                    const int label = label_val_[is_pos];
                    const double label_weight = label_weights_[is_pos];
                    // calculate gradients and hessians
                    const double response = -label * sigmoid_ / (1.0f + std::exp(label * sigmoid_ * score[i]));
                    const double abs_response = fabs(response);
                    gradients[i] = static_cast<score_t>(response * label_weight);
                    hessians[i] = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight);
                  }
                } else {

```


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


