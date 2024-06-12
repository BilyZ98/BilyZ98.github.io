---
layout: post
title: LightGBM usage and implementation
date: 2024-06-08 07:59:00-0400
description: Artificial Intelligence
tags: ai ml 
categories: ai
featured: false
---


# LightGBM

## LightGBM pay more attention minority class 
In LightGBM, there are several ways to handle imbalanced data and pay more attention to the minority class:

is_unbalance parameter: You can set the is_unbalance parameter to true when training the LightGBM model. This will automatically adjust the weights of the classes to be inversely proportional to their frequencies1.
scale_pos_weight parameter: This parameter can provide better control of the weights of the minority and majority classes1. It allows you to set a configurable weight for the minority class1.
class_weight parameter: This parameter is available in the LGBMClassifier and can be used to directly handle imbalanced data2.
Customized metric or sample weights: You can also use a customized metric or apply weights to your samples3.
Remember, when dealing with imbalanced data, accuracy is not the best metric. Instead, consider using metrics such as recall, precision, F1 score, or AUC4. You can also use ‘weighted’ or ‘micro’ averaging methods4.

Here’s a simple example of how you might use the is_unbalance parameter in the C API:

```c
#include <LightGBM/c_api.h>

int main() {
    const char* param = "objective=binary is_unbalance=true";
    const char* train_data_filename = "train.txt";
    DatasetHandle train_data;
    BoosterHandle booster;

    // Load training data
    LGBM_DatasetCreateFromFile(train_data_filename, param, NULL, &train_data);

    // Create booster
    LGBM_BoosterCreate(train_data, param, &booster);

    // Train model
    for (int i = 0; i < 100; ++i) {
        LGBM_BoosterUpdateOneIter(booster);
    }

    // Save model
    LGBM_BoosterSaveModel(booster, 0, -1, "LightGBM_model.txt");

    // Free resources
    LGBM_BoosterFree(booster);
```

## ID3 algorithm that trains decision tree
The training process of a decision tree involves building the tree and determining the split points of the features that most effectively separate the data. Here's a simplified example of how a decision tree might be trained using the ID3 algorithm:

```python
def ID3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node
    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree!
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure
        tree = {best_feature: {}}

        # Remove the feature with the best info gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return tree
```

In this code, the ID3 function is a recursive function that builds a decision tree using the ID3 algorithm. The function takes as input a dataset, a list of features to consider for splitting, and the name of the target attribute. It also takes an optional argument parent_node_class that is used to determine the class label for a node when no further splitting can be done.

The function works by first checking for the base cases that would terminate the recursion: all instances having the same class, no more instances left, no more features left, or no more value left for the feature to split upon. If none of these cases are met, it calculates the information gain of the features and selects the feature with the highest information gain to split upon. It then creates a new tree node for this feature and calls the ID3 function recursively on the sub-datasets to build the subtrees.

Please note that this is a simplified example. Real-world decision tree learning algorithms like those used in scikit-learn or LightGBM are more complex and optimized. They also handle more cases, like missing values and different types of features (continuous, categorical, etc.). They also use different criteria for splitting, like Gini impurity or information gain ratio. The ID3 algorithm is one of the simplest decision tree learning algorithms and is a good starting point for understanding the basics.



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



