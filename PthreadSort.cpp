#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#define HAVE_STRUCT_TIMESPEC
#include <pthread.h>

const int num_threads = 4;

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


struct SortParams {
	double* distances;
	int* index_order;
	int start;
	int end;
};

class PthreadKnn {
private:
	int neighbours_number;

public:
	PthreadKnn(int k) : neighbours_number(k) {}

	int predict_class(double* dataset[], const double* target, int dataset_size, int feature_size) {
		double* distances[3];

		// Allocate memory for distances and index order
		distances[0] = new double[dataset_size];
		distances[1] = new double[dataset_size];

		get_knn(dataset, target, distances, dataset_size, feature_size);

		//assign index to rearrange while sorting and use it to count label occurance
		int* index_order = new int[dataset_size];
		for (int i = 0; i < dataset_size; ++i) {
			index_order[i] = i;
		}

		// Parallelize the sorting process using pthreads
		pthread_t sort_thread_ids[num_threads];
		SortParams sort_params[num_threads];
		int rows_per_thread = dataset_size / num_threads;

		for (int i = 0; i < num_threads; i++) {
			int start = i * rows_per_thread;
			int end = (i == num_threads - 1) ? dataset_size : (i + 1) * rows_per_thread;
			sort_params[i] = { distances[0], index_order, start, end };
			pthread_create(&sort_thread_ids[i], nullptr, sort_distances, &sort_params[i]);
		}

		for (int i = 0; i < num_threads; i++) {
			pthread_join(sort_thread_ids[i], nullptr);
		}
		int zeros_count = 0;
		int ones_count = 0;

		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (distances[1][index_order[i]] == 0) {
				zeros_count += 1;
				std::cout << "0: " << index_order[i] << "," << distances[0][index_order[i]] << std::endl;
			}
			else if (distances[1][index_order[i]] == 1) {
				ones_count += 1;
				std::cout << "1: " << index_order[i] << "," << distances[0][index_order[i]] << std::endl;
			}
		}

		int prediction = (zeros_count > ones_count) ? 0 : 1;

		// Clean up memory
		delete[] distances[0];
		delete[] distances[1];
		delete[] index_order;

		return prediction;
	}


private:
	static void* sort_distances(void* arg) {
		SortParams* params = static_cast<SortParams*>(arg);

		for (int i = params->start; i < params->end - 1; ++i) {
			for (int j = i + 1; j < params->end; ++j) {
				if (params->distances[i] > params->distances[j]) {
					std::swap(params->distances[i], params->distances[j]);
					std::swap(params->index_order[i], params->index_order[j]);
				}
			}
		}

		return nullptr;
	}


	static double euclidean_distance(const double* x, const double* y, int feature_size) {
		double l2 = 0.0;
		for (int i = 1; i < feature_size; i++) {
			l2 += std::pow((x[i] - y[i]), 2);
		}
		return std::sqrt(l2);
	}

	static void* compute_distances(void* arg) {
		ThreadParams* params = static_cast<ThreadParams*>(arg);
		int count = 0;

		//different thread is accessing different index range
		for (int i = params->start; i < params->end; i++) {
			if (params->dataset[i] == params->target) continue; // do not use the same point
			params->distances[0][i] = euclidean_distance(params->target, params->dataset[i], params->feature_size);
			params->distances[1][i] = params->dataset[i][0]; // Store outcome label
			count++;
		}
		std::cout << "Thread " << params->thread_id << " - Number of euclidean run: " << count << std::endl;

		return nullptr;
	}

	void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {
		ThreadParams params[num_threads];
		pthread_t thread_ids[num_threads];

		int rows_per_thread = dataset_size / num_threads;

		for (int i = 0; i < num_threads; i++) {
			//assign start and end point for each thread
			int start = i * rows_per_thread;
			int end = (i == num_threads - 1) ? dataset_size : (i + 1) * rows_per_thread;

			//store parameter
			params[i] = { x, y, distances, dataset_size, feature_size, start, end ,i };
			pthread_create(&thread_ids[i], nullptr, compute_distances, &params[i]);
		}

		for (int i = 0; i < num_threads; i++) {
			pthread_join(thread_ids[i], nullptr);
		}
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

	//Pthread Knn
#pragma region PthreadKnn
	std::cout << "\n\nPthread KNN: " << std::endl;
	std::chrono::steady_clock::time_point pthreadBegin = std::chrono::steady_clock::now();

	PthreadKnn pthreadknn(3); // Use K=3
	int pthreadPrediction = pthreadknn.predict_class(dataset, target, dataset_size, feature_size);
	std::cout << "Pthread Prediction: " << pthreadPrediction << std::endl;

	if (pthreadPrediction == 0) {
		std::cout << "Predicted class: Negative" << std::endl;
	}
	else if (pthreadPrediction == 1) {
		std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
	}
	else {
		std::cout << "Prediction could not be made." << std::endl;
	}

	std::chrono::steady_clock::time_point pthreadEnd = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(pthreadEnd - pthreadBegin).count() << "[µs]" << std::endl;
#pragma endregion


	// Deallocate memory for dataset
	for (int i = 0; i < dataset_size; i++) {
		delete[] dataset[i];
	}

	return 0;
}
