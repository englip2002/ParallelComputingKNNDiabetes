#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include <mutex>
#include "../include/taskflow/taskflow.hpp"
#include "../include/taskflow/algorithm/for_each.hpp"

const int num_threads = 4;

struct MergeSortParams {
	double** distances;
	int datasetSize;
	int low;
	int high;
	int start;
	int end;
};

class SerialMergeSortKnn {
private:
	int neighbours_number;

public:
	SerialMergeSortKnn(int k) : neighbours_number(k) {}

	int predict_class(double* dataset[], const double* target, int dataset_size, int feature_size) {
		double* distances[3];
		int zeros_count = 0;
		int ones_count = 0;

		// Allocate memory for distances and index order
		distances[0] = new double[dataset_size];
		distances[1] = new double[dataset_size];
		distances[2] = new double[dataset_size];

		get_knn(dataset, target, distances, dataset_size, feature_size);

		merge_sort(distances, 0, dataset_size - 1);

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

	static void merge_sort(double** distances, int low, int high) {
		if (low < high) {
			int middle = low + (high - low) / 2;
			merge_sort(distances, low, middle);
			merge_sort(distances, middle + 1, high);
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

class ParallelMergeSortKnn {
private:
	int neighbours_number;

public:
	ParallelMergeSortKnn(int k) : neighbours_number(k) {}

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
		tf::Executor executor;

		// Adjust the granularity as needed
		int task_size = dataset_size / num_threads;

		//Using numthread to loop will be more slower 
		/*for (int i = 0; i < num_threads; i++) {
			int start = i * task_size;
			int end = (i == num_threads - 1) ? (dataset_size - 1) : ((i + 1) * task_size);*/

		auto merge_sort_task = [=, &distances]() {
			MergeSortParams params{ distances, dataset_size, 0, dataset_size - 1 };
			parallel_merge_sort(params);
		};

		taskflow.emplace(merge_sort_task);
		//}

		executor.run(taskflow).wait();

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

	static void parallel_merge_sort(MergeSortParams params) {
		//This cause the wrong sorting
		/*int low = params.start;
		int high = params.end;*/

		int low = params.low;
		int high = params.high;

		if (params.low < high) {
			int middle = low + (high - low) / 2;

			parallel_merge_sort({ params.distances, params.datasetSize, low, middle });
			parallel_merge_sort({ params.distances, params.datasetSize, middle + 1, high });

			merge(params.distances, low, middle, high);
		}
	}

	/*static void parallel_merge_sort(MergeSortParams params) {
		int low = params.low;
		int high = params.high;

		if (low < high) {
			int middle = low + (high - low) / 2;

			parallel_merge_sort({ params.distances, params.datasetSize, low, middle });
			parallel_merge_sort({ params.distances, params.datasetSize, middle + 1, high });

			merge(params.distances, low, middle, high);
		}
	}*/

	double euclidean_distance(const double* x, const double* y, int feature_size) {
		double l2 = 0.0;
		for (int i = 1; i < feature_size; i++) {
			l2 += std::pow((x[i] - y[i]), 2);
		}
		return std::sqrt(l2);
	}

	//The most fastest with using the atomic counter 
	//Passing a lambda function as a parameter to run 
	void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {
		tf::Executor executor;
		tf::Taskflow taskflow;
		std::atomic<int> count(0);  // Use an atomic counter for thread safety

		// Create a lambda function to be used for the for_each_index task
		auto task_lambda = [&, y, x, feature_size, distances](int i) {
			if (x[i] == y) return;  // do not use the same point
			int local_count = count.fetch_add(1, std::memory_order_relaxed);
			distances[0][local_count] = this->euclidean_distance(y, x[i], feature_size);
			distances[1][local_count] = x[i][0]; // Store outcome label
			distances[2][local_count] = i; // Store index
		};

		// Create the for_each_index tasks and add them to the taskflow
		taskflow.for_each_index(0, dataset_size, 1, task_lambda);

		// Run the taskflow
		executor.run(taskflow).wait();

		// Print the count (total number of euclidean runs)
		std::cout << "Number of euclidean run: " << count.load(std::memory_order_relaxed) << std::endl;
	}

	//More faster
	//void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {
	//	tf::Executor executor;
	//	tf::Taskflow taskflow;
	//	int count = 0;
	//	std::mutex distancesMutex;

	//	// Create a lambda function to be used for the for_each_index task
	//	auto task_lambda = [&, y, x, feature_size](int i) {
	//		std::lock_guard<std::mutex> lock(distancesMutex);
	//		if (x[i] == y) return; // do not use the same point
	//		distances[0][count] = this->euclidean_distance(y, x[i], feature_size);
	//		distances[1][count] = x[i][0]; // Store outcome label
	//		distances[2][count] = i; // Store index
	//		count++;
	//	};

	//	// Create the for_each_index tasks and add them to the taskflow
	//	taskflow.for_each_index(0, dataset_size, 1, task_lambda);

	//	// Run the taskflow
	//	executor.run(taskflow).wait();

	//	// Print the count
	//	std::cout << "Number of euclidean run:" << count << std::endl;
	//}

	//void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {
	//	tf::Executor executor;
	//	tf::Taskflow taskflow;
	//	int count = 0;
	//	std::mutex distancesMutex;

	//	// Define the number of tasks
	//	int num_tasks = 4;
	//	int step_size = dataset_size / num_tasks;
	//	int remaining = dataset_size % num_tasks;

	//	for (int task_id = 0; task_id < num_tasks; task_id++) {
	//		int start_index = task_id * step_size;
	//		int end_index = start_index + step_size + (task_id < remaining ? 1 : 0);

	//		taskflow.for_each_index(start_index, end_index, 1, [&, y, x, feature_size, task_id](int i) {
	//			if (x[i] == y) return;
	//			std::lock_guard<std::mutex> lock(distancesMutex);
	//			distances[0][i] = this->euclidean_distance(y, x[i], feature_size);
	//			distances[1][i] = x[i][0]; // Store outcome label
	//			distances[2][i] = i; // Store index
	//			count++;
	//			});
	//		/*int task_count = step_size + (task_id < remaining ? 1 : 0);
	//		std::cout << "Task " << task_id << " - Number of euclidean run: " << task_count << std::endl;*/
	//	}

	//	executor.run(taskflow).wait();

	//	// Print the count for each task
	//	for (int task_id = 0; task_id < num_tasks; ++task_id) {
	//		int task_count = step_size + (task_id < remaining ? 1 : 0);
	//		std::cout << "Task " << task_id << " - Number of euclidean run: " << task_count << std::endl;
	//	}

	//}

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

		/*for (int i = 0; i < num_record_to_sort; i++) {
			std::cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << std::endl;
		}*/

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
		delete[] distances[0];
		delete[] distances[1];
		delete[] distances[2];

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

	//Knn
#pragma region SerialMergeSortKnn
	std::cout << "\n\nSerial Merge Sort KNN: " << std::endl;
	std::chrono::steady_clock::time_point knnBegin = std::chrono::steady_clock::now();
	SerialMergeSortKnn knn(3); // Use K=3

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

	//KNN
#pragma region ParallelMergeSortKnn
	std::cout << "\n\nParallel Merge Sort KNN: " << std::endl;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	ParallelMergeSortKnn parallelKnn(3); // Use K=3

	int parallelPrediction = parallelKnn.predict_class(dataset, target, dataset_size, feature_size);
	std::cout << "Prediction: " << parallelPrediction << std::endl;

	if (parallelPrediction == 0) {
		std::cout << "Predicted class: Negative" << std::endl;
	}
	else if (parallelPrediction == 1) {
		std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
	}
	else {
		std::cout << "Prediction could not be made." << std::endl;
	}

	std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(e - start).count() << "[µs]" << std::endl;
#pragma endregion

	//Knn
#pragma region Knn
	std::cout << "\n\nQuick Sort KNN: " << std::endl;
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	Knn Quickknn(3); // Use K=3

	int Quickprediction = Quickknn.predict_class(dataset, target, dataset_size, feature_size);
	std::cout << "Prediction: " << Quickprediction << std::endl;

	if (Quickprediction == 0) {
		std::cout << "Predicted class: Negative" << std::endl;
	}
	else if (Quickprediction == 1) {
		std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
	}
	else {
		std::cout << "Prediction could not be made." << std::endl;
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#pragma endregion

	// Deallocate memory for dataset
	for (int i = 0; i < dataset_size; i++) {
		delete[] dataset[i];
	}

	return 0;
}
