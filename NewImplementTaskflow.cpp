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
#include "../include/taskflow/taskflow.hpp"
#include "../include/taskflow/algorithm/for_each.hpp"

using namespace std;
using namespace chrono;
using namespace tf;

const int num_tasks = 4;
const int sort_record_each_thread = 5;
const int num_record_to_sort = num_tasks * sort_record_each_thread;

class TaskflowKnn {
private:
	int neighbours_number;

public:
	TaskflowKnn(int k) : neighbours_number(k) {}

	int predict_class(double* dataset[], double* target, int dataset_size, int feature_size) {
		double* distances[3];
		int zeros_count = 0;
		int ones_count = 0;
		int prediction = -1;
		int chunk_size = dataset_size / num_tasks;

		distances[0] = new double[dataset_size];
		distances[1] = new double[dataset_size];
		distances[2] = new double[dataset_size];

		Taskflow taskflow;
		Executor executor;

		auto [get_knn_task, merge_sort_task, subsequence_task] = taskflow.emplace(
			[&](Subflow subflow)
			{

				 
				subflow.for_each_index(
					0, num_tasks, 1,
					[&](int t)
					{
						int count = 0;
						int start = t * chunk_size;
						int end = (t == num_tasks - 1) ? dataset_size : (t + 1) * chunk_size;

						for (int i = start; i < end; i++) {
							if (dataset[i] == target) continue; // do not use the same point
							double l2 = 0.0;
							for (int j = 1; j < feature_size; j++) {
								l2 += pow((target[j] - dataset[i][j]), 2);
							}
							distances[0][i] = sqrt(l2);
							//distances[0][i] = euclidean_distance(target, dataset[i], feature_size);
							distances[1][i] = dataset[i][0]; // Store outcome label
							distances[2][i] = i; // Store index
							count++;
						}
						cout << "Task " << t << " - Number of euclidean run: " << count << endl;
					}
				);
			},
			[&](Subflow subflow)
			{
				subflow.for_each_index(
					0, num_tasks, 1,
					[&](int t)
					{
						int start = t * chunk_size;
						int end = (t == num_tasks - 1) ? dataset_size : (t + 1) * chunk_size;

						//Sorting
						merge_sort(distances, start, end - 1);

						/*for (int i = start; i < start + 5; i++) {
							cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << endl;
						}*/
					}
				);


			},
			[&]()
			{
				double* sortedDistances[3];
				sortedDistances[0] = new double[num_record_to_sort];
				sortedDistances[1] = new double[num_record_to_sort];
				sortedDistances[2] = new double[num_record_to_sort];

				//extract first 5 from each thread (shortest distance)
				for (int i = 0; i < 3; i++) {
					//cout << "A" << endl;
					for (int k = 0; k < num_tasks; k++) {
						//cout << "B" << endl;
						for (int j = 0; j < sort_record_each_thread; j++) {
							// cout << "C" << endl;
							sortedDistances[i][(k * sort_record_each_thread) + j] = distances[i][k * chunk_size + j];
						}
					}
				}

				//sort again
				merge_sort(sortedDistances, 0, num_record_to_sort - 1);

				//for (int i = 0; i < num_record_to_sort; i++) {
				//	cout << sortedDistances[0][i] << "," << sortedDistances[1][i] << "," << sortedDistances[2][i] << endl;
				//}

				// Count label occurrences in the K nearest neighbors
				for (int i = 0; i < neighbours_number; i++) {
					//cout << neighbours_number << " ";
					if (sortedDistances[1][i] == 0) {
						zeros_count += 1;
						cout << "0: " << sortedDistances[0][i] << "," << sortedDistances[2][i] << endl;
					}
					else if (sortedDistances[1][i] == 1) {
						ones_count += 1;
						cout << "1: " << sortedDistances[0][i] << "," << sortedDistances[2][i] << endl;
					}
				}

				prediction = (zeros_count > ones_count) ? 0 : 1;

				//delete[] distances[0];
				//delete[] distances[1];
				//delete[] distances[2];
				//delete[] sortedDistances[0];
				//delete[] sortedDistances[1];
				//delete[] sortedDistances[2];
			}
		);
		
		get_knn_task.precede(merge_sort_task.precede(subsequence_task));

		executor.run(taskflow).get();

		delete[] distances[0];
		delete[] distances[1];
		delete[] distances[2];

		return prediction;
	}
private:
	static void merge(double** distances, int low, int middle, int high) {
		int n1 = middle - low + 1;
		int n2 = high - middle;  //high = middle + 1 

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

	/*double euclidean_distance(const double* x, const double* y, int feature_size) {
		double l2 = 0.0;
		for (int i = 1; i < feature_size; i++) {
			l2 += pow((x[i] - y[i]), 2);
		}
		return sqrt(l2);
	}*/
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

		/*for (int i = 0; i < 10; i++) {
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

		//Clean up memory
		delete[] distances[0];
		delete[] distances[1];
		delete[] distances[2];

		return prediction;
	}

private:

	static void merge(double** distances, int low, int middle, int high) {
		int n1 = middle - low + 1;
		int n2 = high - middle;  //high = middle + 1 

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

vector<double> parseLine(const string& line) {
	vector<double> row;
	istringstream iss(line);
	string value;

	while (getline(iss, value, ',')) {
		try {
			double num = stod(value);
			row.push_back(num);
		}
		catch (const invalid_argument&) {
			cerr << "Invalid data in CSV: " << value << endl;
		}
	}

	return row;
}

int main() {
	string filename = "diabetes_binary.csv";

	const int dataset_size = 53681;
	const int feature_size = 22;
	// Get the number of available CPU cores
	//const int num_threads = thread::hardware_concurrency();

	double** dataset = new double* [dataset_size];
	//double target[feature_size] = { 0.0, 0.0, 0.0, 1.0, 24.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 5.0, 3.0 };
	double target[feature_size] = { 1.0, 1.0, 1.0, 1.0, 30.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 5.0, 30.0, 30.0, 1.0, 0.0, 9.0, 5.0, 1.0 };
	// Allocate memory for dataset and target
	for (int i = 0; i < dataset_size; i++) {
		dataset[i] = new double[feature_size];
	}

	// Read data from CSV and populate dataset and target
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error opening file: " << filename << endl;
		return 1;
	}

	string header;
	getline(file, header);

	string line;
	int index = 0;
	while (getline(file, line) && index < dataset_size) {
		vector<double> row = parseLine(line);
		for (int j = 0; j < feature_size; j++) {
			dataset[index][j] = row[j];
		}
		index++;
	}

	cout << "Number of records: " << dataset_size << endl;

	steady_clock::time_point begin = steady_clock::now();

#pragma region TasflowKnn
	TaskflowKnn tfknn(3); // Use K=3, with 4 threads
	int taskFlowPrediction = tfknn.predict_class(dataset, target, dataset_size, feature_size);

	steady_clock::time_point end = steady_clock::now();
	cout << "Prediction: " << taskFlowPrediction << endl;

	if (taskFlowPrediction == 0) {
		cout << "Predicted class: Negative" << endl;
	}
	else if (taskFlowPrediction == 1) {
		cout << "Predicted class: Prediabetes or Diabetes" << endl;
	}
	else {
		cout << "Prediction could not be made." << endl;
	}

	cout << "Time difference = " << duration_cast<microseconds>(end - begin).count() << "[ s]" << endl;

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

	return 0;
}