#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include "taskflow/taskflow/taskflow.hpp"

//bool sort_by_dist(const double* v1, const double* v2);

class Knn {
private:
    int neighbours_number;

public:
    Knn(int k) : neighbours_number(k) {}

    int predict_class(double* dataset[], const double* target, int dataset_size, int feature_size) {
        double* distances[3];
        int zeros_count = 0;
        int ones_count = 0;
        int prediction = -1;

        distances[0] = new double[dataset_size];
        distances[1] = new double[dataset_size];
        distances[2] = new double[dataset_size];

        get_knn(dataset, target, distances, dataset_size, feature_size);

        for (int i = 0; i < dataset_size; i++) {
            if (distances[1][i] == 0) {
                zeros_count += 1;
            }
            if (distances[1][i] == 1) {
                ones_count += 1;
            }
        }

        if (zeros_count > ones_count) {
            prediction = 0;
        }
        else {
            prediction = 1;
        }

        delete[] distances[0];
        delete[] distances[1];
        delete[] distances[2];

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
        int count = 0;
        for (int i = 0; i < dataset_size; i++) {
            if (x[i] == y) continue; // do not use the same point
            distances[0][count] = this->euclidean_distance(y, x[i], feature_size);
            distances[1][count] = x[i][0]; // Store outcome label
            distances[2][count] = i; // Store index
            count++;
        }
        std::cout << "Number of euclidean run:" << count << std::endl;
        std::sort(distances[2], distances[2] + count, [distances](int i, int j) {
            return distances[0][i] < distances[0][j];
            });
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
        int prediction = -1;

        distances[0] = new double[dataset_size];
        distances[1] = new double[dataset_size];
        distances[2] = new double[dataset_size];

        get_knn(dataset, target, distances, dataset_size, feature_size);

        for (int i = 0; i < dataset_size; i++) {
            if (distances[1][i] == 0) {
                zeros_count += 1;
            }
            if (distances[1][i] == 1) {
                ones_count += 1;
            }
        }

        if (zeros_count > ones_count) {
            prediction = 0;
        }
        else {
            prediction = 1;
        }

        delete[] distances[0];
        delete[] distances[1];
        delete[] distances[2];

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
        int count = 0;

        tf::Executor executor;
        tf::Taskflow taskflow;

        for (int i = 0; i < dataset_size; i++) {
            if (x[i] == y) continue; // do not use the same point
            auto task = taskflow.emplace([this, i, x, y, distances, feature_size, &count]() {
                distances[0][i] = this->euclidean_distance(y, x[i], feature_size);
                distances[1][i] = x[i][0]; // Store outcome label
                distances[2][i] = i; // Store index
                count++;
                });
        }

        executor.run(taskflow).wait();

        std::sort(distances[2], distances[2] + dataset_size, [distances](int i, int j) {
            return distances[0][i] < distances[0][j];
            });

        std::cout << "Number of euclidean run: " << count << std::endl;
    }

};

bool sort_by_dist(const double* v1, const double* v2) {
    return v1[0] < v2[0];
}

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
    double target[feature_size] = { 0.0, 0.0, 0.0, 1.0, 24.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 5.0, 3.0 };

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

    TaskflowKnn taskflowknn(3); // Use K=3
    int taskflowPrediction = taskflowknn.predict_class(dataset, target, dataset_size, feature_size);
    std::cout << "Taskflow Prediction: " << taskflowPrediction << std::endl;

    if (taskflowPrediction == 0) {
        std::cout << "Predicted class: Negative" << std::endl;
    }
    else if (taskflowPrediction == 1) {
        std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
    }
    else {
        std::cout << "Prediction could not be made." << std::endl;
    }

    std::chrono::steady_clock::time_point taskflowEnd = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(taskflowEnd - taskflowBegin).count() << "[�s]" << std::endl;
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
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(knnEnd - knnBegin).count() << "[�s]" << std::endl;
#pragma endregion


    // Deallocate memory for dataset
    for (int i = 0; i < dataset_size; i++) {
        delete[] dataset[i];
    }

    return 0;
}
