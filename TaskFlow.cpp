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
#include "../include/taskflow/algorithm/sort.hpp"

bool sort_by_dist(const double* v1, const double* v2);

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

        quick_sort(distances, 0, dataset_size - 1);

        //display first 10 sorted record to check
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

    static int partition(double** distances, int low, int high) {
        double pivot = distances[0][high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (distances[0][j] <= pivot) {
                i++;
                swap(distances, i, j);
            }
        }
        swap(distances, i + 1, high);
        return i + 1;
    }

    static void swap(double** distances, int i, int j) {
        std::swap(distances[0][i], distances[0][j]);
        std::swap(distances[1][i], distances[1][j]);
        std::swap(distances[2][i], distances[2][j]);
    }

    static void quick_sort(double** distances, int low, int high) {
        if (low < high) {
            int pivotIndex = partition(distances, low, high);
            quick_sort(distances, low, pivotIndex - 1);
            quick_sort(distances, pivotIndex + 1, high);
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

        int* index_order = new int[dataset_size];
        for (int i = 0; i < dataset_size; ++i) {
            index_order[i] = i;
        }

        // Parallelized sorting using Taskflow
        tf::Executor executor;
        tf::Taskflow taskflow;

        taskflow.emplace([&]() {
            std::sort(index_order, index_order + dataset_size, [distances](int i, int j) {
                return distances[0][i] < distances[0][j];
                });
            });
        executor.run(taskflow).wait();

        /* taskflow.emplace([&]() {
             taskflow.sort(index_order, index_order + dataset_size, [distances](int i, int j) {
                 return static_cast<int>(distances[0][i] - distances[0][j]);
                 });
             });
         executor.run(taskflow).wait();*/


         // Count label occurrences in the K nearest neighbors
        for (int i = 0; i < neighbours_number; i++) {
            if (distances[1][index_order[i]] == 0) {
                zeros_count += 1;
                std::cout << "0: " << distances[0][index_order[i]] << "," << distances[2][index_order[i]] << std::endl;
            }
            else if (distances[1][index_order[i]] == 1) {
                ones_count += 1;
                std::cout << "1: " << distances[0][index_order[i]] << "," << distances[2][index_order[i]] << std::endl;
            }
        }

        int prediction = (zeros_count > ones_count) ? 0 : 1;

        // Clean up memory
        delete[] distances[0];
        delete[] distances[1];
        delete[] distances[2];
        delete[] index_order;

        return prediction;
    }

private:
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

//// Function to read a chunk of lines from the CSV file
//void readCsvChunk(std::ifstream& file, int startLine, int endLine, std::vector<std::vector<double>>& dataset) {
//    std::string line;
//    for (int i = startLine; i < endLine; ++i) {
//        if (std::getline(file, line)) {
//            dataset[i] = parseLine(line);
//        } else {
//            break; // Break if end of file reached
//        }
//    }
//}

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
