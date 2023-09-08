#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include "../include/taskflow/taskflow.hpp"
#include "../include/taskflow/algorithm/for_each.hpp"

const int num_record_to_sort = 20;

struct MergeSortParams {
    double** distances;
    int datasetSize;
    int low;
    int high;
};

class Knn {
private:
    int neighbours_number;

public:
    Knn(int k) : neighbours_number(k) {}

    int predict_class(double* dataset[], const double* target, int dataset_size, int feature_size) {
        double* distances[3];
        int zeros_count = 0;
        int ones_count = 0;

        // Allocate memory for distances and index order
        distances[0] = new double[dataset_size];
        distances[1] = new double[dataset_size];
        distances[2] = new double[dataset_size];

        get_knn(dataset, target, distances, dataset_size, feature_size);

        serial_merge_sort(distances, 0, dataset_size - 1);

        for (int i = 0; i < 10; i++) {
            std::cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << std::endl;
        }

        // Count label occurrences in the K nearest neighbors
        for (int i = 0; i < neighbours_number; i++) {
            if (distances[1][i] == 0) {
                zeros_count += 1;
                std::cout << "0: " << distances[0][i] << "," << distances[2][i] << std::endl;
            }
            else if (distances[1][i] == 1) {
                ones_count += 1;
                std::cout << "1: " << distances[0][i] << "," << distances[2][i] << std::endl;
            }
        }

        int prediction = (zeros_count > ones_count) ? 0 : 1;

        // Clean up memory
        //delete[] distances[0];
        //delete[] distances[1];
        //delete[] distances[2];

        return prediction;
    }

private:

    static void merge(double** distances, int low, int middle, int high) {
        int n1 = middle - low + 1;
        int n2 = high - middle;

        double* left[3];
        double* right[3];

        // Create temporary arrays
        for (int i = 0; i < 3; i++) {
            left[i] = new double[n1];
            right[i] = new double[n2];
        }

        // Copy data to temporary arrays left[] and right[]
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < 3; j++) {
                left[j][i] = distances[j][low + i];
            }
        }
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < 3; j++) {
                right[j][i] = distances[j][middle + 1 + i];
            }
        }

        // Merge the temporary arrays back into distances[3]
        int i = 0, j = 0, k = low;
        while (i < n1 && j < n2) {
            if (left[0][i] <= right[0][j]) {
                for (int x = 0; x < 3; x++) {
                    distances[x][k] = left[x][i];
                }
                i++;
            }
            else {
                for (int x = 0; x < 3; x++) {
                    distances[x][k] = right[x][j];
                }
                j++;
            }
            k++;
        }

        // Copy the remaining elements of left[], if any
        while (i < n1) {
            for (int x = 0; x < 3; x++) {
                distances[x][k] = left[x][i];
            }
            i++;
            k++;
        }

        // Copy the remaining elements of right[], if any
        while (j < n2) {
            for (int x = 0; x < 3; x++) {
                distances[x][k] = right[x][j];
            }
            j++;
            k++;
        }

        // Clean up temporary arrays
        for (int x = 0; x < 3; x++) {
            delete[] left[x];
            delete[] right[x];
        }
    }

    static void serial_merge_sort(double** distances, int low, int high) {
        if (low < high) {
            int middle = low + (high - low) / 2;
            serial_merge_sort(distances, low, middle);
            serial_merge_sort(distances, middle + 1, high);
            merge(distances, low, middle, high);
        }
    }

    double euclidean_distance(const double* x, const double* y, int feature_size) {
        double l2 = 0.0;
        for (int i = 1; i < feature_size; i++) {
            l2 += std::pow((x[i] - y[i]), 2);
        }
        return std::sqrt(l2);
    }

    void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {
        int count = 0;
        for (int i = 0; i < dataset_size; i++) {
            if (x[i] == y) continue; // do not use the same point
            distances[0][count] = this->euclidean_distance(y, x[i], feature_size);
            distances[1][count] = x[i][0]; // Store outcome label
            distances[2][count] = i; // Store index
            count++;
        }
        std::cout << "Number of euclidean run:" << count << std::endl;
    }
};

class TaskflowKnn {
private:
    int neighbours_number;

public:
    TaskflowKnn(int k) : neighbours_number(k) {}

    int predict_class(double* dataset[], const double* target, int dataset_size, int feature_size) {
        double* distances[3];
        int zeros_count = 0;
        int ones_count = 0;

        // Allocate memory for distances and index order
        distances[0] = new double[dataset_size];
        distances[1] = new double[dataset_size];
        distances[2] = new double[dataset_size];

        get_knn(dataset, target, distances, dataset_size, feature_size);

        tf::Taskflow taskflow;

        // Add sorting task for merge sort
        MergeSortParams params{ distances, dataset_size, 0, dataset_size - 1 };
        taskflow.emplace([params]() {
            parallel_merge_sort(params);
            });

        tf::Executor executor;
        executor.run(taskflow).wait();

        double* finalSortedDistances[3];
        finalSortedDistances[0] = new double[num_record_to_sort];
        finalSortedDistances[1] = new double[num_record_to_sort];
        finalSortedDistances[2] = new double[num_record_to_sort];

        // Extract first 3 from the sorted array (shortest distance)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < num_record_to_sort; j++) {
                finalSortedDistances[i][j] = distances[i][j];
            }
        }

        // Count label occurrences in the K nearest neighbors
        for (int i = 0; i < neighbours_number; i++) {
            if (finalSortedDistances[1][i] == 0) {
                zeros_count += 1;
                std::cout << "0: " << finalSortedDistances[0][i] << "," << finalSortedDistances[2][i] << std::endl;
            }
            else if (finalSortedDistances[1][i] == 1) {
                ones_count += 1;
                std::cout << "1: " << finalSortedDistances[0][i] << "," << finalSortedDistances[2][i] << std::endl;
            }
        }

        int prediction = (zeros_count > ones_count) ? 0 : 1;

        // Clean up memory
        delete[] distances[0];
        delete[] distances[1];
        delete[] distances[2];

        return prediction;

    }

private:
    static void merge(double** distances, int low, int middle, int high) {
        int n1 = middle - low + 1;
        int n2 = high - middle;

        double* left[3];
        double* right[3];

        // Create temporary arrays
        for (int i = 0; i < 3; i++) {
            left[i] = new double[n1];
            right[i] = new double[n2];
        }

        // Copy data to temporary arrays left[] and right[]
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < 3; j++) {
                left[j][i] = distances[j][low + i];
            }
        }
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < 3; j++) {
                right[j][i] = distances[j][middle + 1 + i];
            }
        }

        // Merge the temporary arrays back into distances[3]
        int i = 0, j = 0, k = low;
        while (i < n1 && j < n2) {
            if (left[0][i] <= right[0][j]) {
                for (int x = 0; x < 3; x++) {
                    distances[x][k] = left[x][i];
                }
                i++;
            }
            else {
                for (int x = 0; x < 3; x++) {
                    distances[x][k] = right[x][j];
                }
                j++;
            }
            k++;
        }

        // Copy the remaining elements of left[], if any
        while (i < n1) {
            for (int x = 0; x < 3; x++) {
                distances[x][k] = left[x][i];
            }
            i++;
            k++;
        }

        // Copy the remaining elements of right[], if any
        while (j < n2) {
            for (int x = 0; x < 3; x++) {
                distances[x][k] = right[x][j];
            }
            j++;
            k++;
        }

        // Clean up temporary arrays
        for (int x = 0; x < 3; x++) {
            delete[] left[x];
            delete[] right[x];
        }
    }

    static void parallel_merge_sort(MergeSortParams params) {
        int low = params.low;
        int high = params.high;

        if (low < high) {
            int middle = low + (high - low) / 2;

            parallel_merge_sort({ params.distances, params.datasetSize, low, middle });
            parallel_merge_sort({ params.distances, params.datasetSize, middle + 1, high });

            merge(params.distances, low, middle, high);
        }
    }

    double euclidean_distance(const double* x, const double* y, int feature_size) {
        double l2 = 0.0;
        for (int i = 1; i < feature_size; i++) {
            l2 += std::pow((x[i] - y[i]), 2);
        }
        return std::sqrt(l2);
    }

    void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {
        tf::Executor executor;
        tf::Taskflow taskflow;

        //int num_task = 4;
        //int step_size = dataset_size / 13420;

        // Parallelized loop using Taskflow's for_each_index algorithm
        taskflow.for_each_index(0, dataset_size, 1, [&, y, x, feature_size](int i) {
            if (x[i] == y) return;
            distances[0][i] = this->euclidean_distance(y, x[i], feature_size);
            distances[1][i] = x[i][0]; // Store outcome label
            distances[2][i] = i; // Store index
            });

        executor.run(taskflow).wait();

        std::cout << "Number of euclidean run: " << dataset_size << std::endl;
    }
};

std::vector<double> parseLine(const std::string& line) {
    std::vector<double> row;
    std::istringstream iss(line);
    std::string value;

    while (std::getline(iss, value, ',')) {
        try {
            double num = std::stod(value);
            row.push_back(num);
        }
        catch (const std::invalid_argument&) {
            std::cerr << "Invalid data in CSV: " << value << std::endl;
        }
    }

    return row;
}

int main() {
    std::string filename = "diabetes_binary.csv";

    //const int dataset_size = 253681; 
    const int dataset_size = 53681;
    const int feature_size = 22;

    double** dataset = new double* [dataset_size];
    //double target[feature_size] = { 0.0, 0.0, 0.0, 1.0, 24.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 5.0, 3.0 };
    double target[feature_size] = { 1.0, 1.0, 1.0, 1.0, 30.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 5.0, 30.0, 30.0, 1.0, 0.0, 9.0, 5.0, 1.0 };

    // Allocate memory for dataset and target
    for (int i = 0; i < dataset_size; i++) {
        dataset[i] = new double[feature_size];
    }

    // Read data from CSV and populate dataset and target
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }

    std::string header;
    std::getline(file, header);

    std::string line;
    int index = 0;
    while (std::getline(file, line) && index < dataset_size) {
        std::vector<double> row = parseLine(line);
        for (int j = 0; j < feature_size; j++) {
            dataset[index][j] = row[j];
        }
        index++;
    }

    std::cout << "Number of records: " << index << std::endl;

    //Taskflow Knn
#pragma region TaskflowKnn
    std::cout << "\n\Taskflow KNN: " << std::endl;
    std::chrono::steady_clock::time_point taskflowBegin = std::chrono::steady_clock::now();

    TaskflowKnn tfknn(3); // Use K=3
    int taskFlowPrediction = tfknn.predict_class(dataset, target, dataset_size, feature_size);

    std::chrono::steady_clock::time_point taskflowEnd = std::chrono::steady_clock::now();
    std::cout << "Taskflow Prediction: " << taskFlowPrediction << std::endl;
    if (taskFlowPrediction == 0) {
        std::cout << "Predicted class: Negative" << std::endl;
    }
    else if (taskFlowPrediction == 1) {
        std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
    }
    else {
        std::cout << "Prediction could not be made." << std::endl;
    }

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(taskflowEnd - taskflowBegin).count() << "[µs]" << std::endl;
#pragma endregion

    //Knn
#pragma region Knn
    std::cout << "\n\nKNN: " << std::endl;
    std::chrono::steady_clock::time_point knnBegin = std::chrono::steady_clock::now();
    Knn knn(3); // Use K=3

    int prediction = knn.predict_class(dataset, target, dataset_size, feature_size);
    std::cout << "Prediction: " << prediction << std::endl;

    if (prediction == 0) {
        std::cout << "Predicted class: Negative" << std::endl;
    }
    else if (prediction == 1) {
        std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
    }
    else {
        std::cout << "Prediction could not be made." << std::endl;
    }

    std::chrono::steady_clock::time_point knnEnd = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(knnEnd - knnBegin).count() << "[µs]" << std::endl;
#pragma endregion


    // Deallocate memory for dataset
    for (int i = 0; i < dataset_size; i++) {
        delete[] dataset[i];
    }

    return 0;
}
