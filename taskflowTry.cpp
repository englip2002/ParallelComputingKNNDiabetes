#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>
#include "taskflow/taskflow/taskflow.hpp"

bool sort_by_dist(const double* v1, const double* v2);

struct ThreadParams {
    double** dataset;
    const double* target;
    double** distances;
    int dataset_size;
    int feature_size;
    int start;
    int end;
    int thread_id;
};

class TaskflowKnn {
private:
    int neighbours_number;
    int num_threads;

public:
    TaskflowKnn(int k, int threads) : neighbours_number(k), num_threads(threads) {}

    int predict_class(double* dataset[], const double* target, int dataset_size, int feature_size) {
        double* distances[3];
        int zeros_count = 0;
        int ones_count = 0;

        distances[0] = new double[dataset_size];
        distances[1] = new double[dataset_size];
        distances[2] = new double[dataset_size];

        tf::Executor executor(num_threads);
        tf::Taskflow taskflow;

        for (int t = 0; t < num_threads; ++t) {
            int chunk_size = dataset_size / num_threads;
            int start_idx = t * chunk_size;
            int end_idx = (t == num_threads - 1) ? dataset_size : (start_idx + chunk_size);

            taskflow.emplace([&, start_idx, end_idx, t] {
                ThreadParams params = { dataset, target, distances, dataset_size, feature_size, start_idx, end_idx, t };
                compute_distances(&params);
                });
        }

        executor.run(taskflow).wait();

        int* index_order = new int[dataset_size];
        for (int i = 0; i < dataset_size; ++i) {
            index_order[i] = i;
        }

        // Partial sort using nth_element instead of full sort
        std::nth_element(index_order, index_order + neighbours_number, index_order + dataset_size,
            [&](int i, int j) {
                return distances[0][i] < distances[0][j];
            });



        // Count label occurrences in the K nearest neighbors
        for (int i = 0; i < neighbours_number; i++) {
            if (distances[1][index_order[i]] == 0) {
                zeros_count += 1;
            }
            else if (distances[1][index_order[i]] == 1) {
                ones_count += 1;
            }
        }

        int prediction = (zeros_count > ones_count) ? 0 : 1;

        delete[] distances[0];
        delete[] distances[1];
        delete[] distances[2];

        return prediction;
    }

private:
    static double euclidean_distance(const double* x, const double* y, int feature_size) {
        double l2 = 0.0;
        for (int i = 1; i < feature_size; i++) {
            l2 += std::pow((x[i] - y[i]), 2);
        }
        return std::sqrt(l2);
    }

    static std::mutex outputMutex;

    static void compute_distances(ThreadParams* params) {
        int count = 0;

        for (int i = params->start; i < params->end; i++) {
            if (params->dataset[i] == params->target) continue; // do not use the same point
            params->distances[0][i] = euclidean_distance(params->target, params->dataset[i], params->feature_size);
            params->distances[1][i] = params->dataset[i][0]; // Store outcome label
            params->distances[2][i] = i; // Store index
            count++;
        }

        // Lock the mutex before printing
        std::lock_guard<std::mutex> lock(outputMutex);
        std::cout << "Thread " << params->thread_id << " - Number of euclidean run: " << count << std::endl;
    }
};

std::mutex TaskflowKnn::outputMutex; // Definition of static mutex

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

    const int dataset_size = 53681;
    const int feature_size = 22;
    const int num_threads = 4;
    // Get the number of available CPU cores
    //const int num_threads = std::thread::hardware_concurrency();

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

    std::cout << "Number of records: " << dataset_size << std::endl;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    TaskflowKnn tfknn(3, num_threads); // Use K=3, with 4 threads
    int taskFlowPrediction = tfknn.predict_class(dataset, target, dataset_size, feature_size);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Prediction: " << taskFlowPrediction << std::endl;

    if (taskFlowPrediction == 0) {
        std::cout << "Predicted class: Negative" << std::endl;
    }
    else if (taskFlowPrediction == 1) {
        std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
    }
    else {
        std::cout << "Prediction could not be made." << std::endl;
    }

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

	// Deallocate memory for dataset
	for (int i = 0; i < dataset_size; i++) {
		delete[] dataset[i];
	}
	delete[] dataset;

	return 0;
}