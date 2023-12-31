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
#include "../include/taskflow/algorithm/sort.hpp"

using namespace std;
using namespace chrono;
using namespace tf;

//the number that serial sort again after the parallel sort to resolve the race condition
const int num_record_to_sort = 20;

class TaskflowParallelKnn {
private:
	int neighbours_number;

public:
	TaskflowParallelKnn(int k) : neighbours_number(k) {}

	int predict_class(double* dataset[], const double* target, int dataset_size, int feature_size) {
		double* distances[3];
		int zeros_count = 0;
		int ones_count = 0;

		// Allocate memory for distances and index order
		distances[0] = new double[dataset_size]; //euclidean distance 
		distances[1] = new double[dataset_size]; //label : prediction 0 or 1
		distances[2] = new double[dataset_size]; //index number 

		double* index_order = new double[dataset_size];

		Taskflow taskflow;
		Executor executor;

		//Create a task into taskflow not execute immediately
		//Taskflow parallel iteration --> 4 parameter
		//(first_index, last_index, step_size, lambda function)
		taskflow.for_each_index(0, dataset_size, 1, [=, &distances](int i) {
			distances[0][i] = euclidean_distance(target, dataset[i], feature_size);
			distances[1][i] = dataset[i][0]; // Store outcome label
			distances[2][i] = i; // Store index
			});

		//Execute the task within the taskflow and wiat all task is complete 
		executor.run(taskflow).wait();

		//Call for sorting --> Parallel sorting implement 
		selection_sort(distances, dataset_size);

		//for (int i = 0; i < 10; i++) {
		//	cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << endl;
		//}

		cout << "Top 3 Nearest K value: " << endl;

		int count = 0;
		// Count label occurrences in the K nearest neighbors
		for (int i = 0; count < neighbours_number; i++) {
			if (distances[1][i] == 0 && distances[0][i] > 0) {
				zeros_count += 1;
				cout << "0: " << distances[0][i] << endl;
				count++;
			}
			else if (distances[1][i] == 1 && distances[0][i] > 0) {
				ones_count += 1;
				cout << "1: " << distances[0][i] << endl;
				count++;
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
	static void selection_sort(double** distances, int dataset_size) {
		Taskflow taskflow;
		Executor executor;

		//Taskflow parallel iteration --> 4 parameter
		//(first_index, last_index, step_size, lambda function)
		taskflow.for_each_index(0, dataset_size, 1, [=, &distances](int i) {
			int min_index = i;
			for (int j = i + 1; j < dataset_size; j++) {
				if (distances[0][j] < distances[0][min_index]) {
					min_index = j;
				}
			}

			if (min_index != i) {
				// Swap distances for all dimensions 
				for (int x = 0; x < 3; x++) {
					double temp = distances[x][i];
					distances[x][i] = distances[x][min_index];
					distances[x][min_index] = temp;
				}
			}
			});

		executor.run(taskflow).wait();

		//Serial Sort again to resolve the race condition but the number sort is the first 20 records
		for (int i = 0; i < num_record_to_sort; i++) {
			int min_index = i;
			for (int j = i + 1; j < dataset_size; j++) {
				if (distances[0][j] < distances[0][min_index]) {
					min_index = j;
				}
			}

			if (min_index != i) {
				// Swap distances for all dimensions 
				for (int x = 0; x < 3; x++) {
					double temp = distances[x][i];
					distances[x][i] = distances[x][min_index];
					distances[x][min_index] = temp;
				}
			}
		}
	}

	double euclidean_distance(const double* x, const double* y, int feature_size) {
		double l2 = 0.0;
		for (int i = 1; i < feature_size; i++) {
			l2 += pow((x[i] - y[i]), 2);
		}
		return sqrt(l2);
	}

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

		selectionSort(distances, dataset_size);

		/*for (int i = 0; i < 10; i++) {
			cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << std::endl;
		}*/

		cout << "Top 3 Nearest K value: " << endl;

		int count = 0;
		// Count label occurrences in the K nearest neighbors
		for (int i = 0; count < neighbours_number; i++) {
			if (distances[1][i] == 0 && distances[0][i] > 0) {
				zeros_count += 1;
				cout << "0: " << distances[0][i] << endl;
				count++;
			}
			else if (distances[1][i] == 1 && distances[0][i] > 0) {
				ones_count += 1;
				cout << "1: " << distances[0][i] << endl;
				count++;
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

	static void selectionSort(double** distances, int dataset_size) {
		for (int i = 0; i < dataset_size - 1; i++) {
			int min_index = i;
			for (int j = i + 1; j < dataset_size; j++) {
				if (distances[0][j] < distances[0][min_index]) {
					min_index = j;
				}
			}

			if (min_index != i) {
				// Swap distances for all dimensions 
				for (int x = 0; x < 3; x++) {
					double temp = distances[x][i];
					distances[x][i] = distances[x][min_index];
					distances[x][min_index] = temp;
				}
			}
		}
	}

	double euclidean_distance(const double* x, const double* y, int feature_size) {
		double l2 = 0.0;
		for (int i = 1; i < feature_size; i++) {
			l2 += pow((x[i] - y[i]), 2);
		}
		return sqrt(l2);
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
		cout << "Number of euclidean run:" << count << endl;
	}

};

std::vector<double> parseLine(const string& line) {
	std::vector<double> row;
	std::istringstream iss(line);
	std::string value;

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

	//const int dataset_size = 30000; 
	//const int dataset_size = 100000;
	const int dataset_size = 250000;
	const int feature_size = 22;
	int time_parallel_knn = 0;
	int time_serial_knn = 0;
	int time_reduce = 0;

	double** dataset = new double* [dataset_size];
	double target[feature_size] = { 0.0, 0.0, 0.0, 1.0, 24.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 5.0, 3.0 };
	//double target[feature_size] = { 1.0, 1.0, 1.0, 1.0, 30.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 5.0, 30.0, 30.0, 1.0, 0.0, 9.0, 5.0, 1.0 };
	//double target[feature_size] = { 0.0, 1.0, 1.0, 1.0, 28.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 4.0, 0.0, 10.0, 1.0, 0.0, 12.0, 6.0, 2.0 };

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

	string header;
	getline(file, header);

	string line;
	int index = 0;
	while (getline(file, line) && index < dataset_size) {
		std::vector<double> row = parseLine(line);
		for (int j = 0; j < feature_size; j++) {
			dataset[index][j] = row[j];
		}
		index++;
	}

	cout << "Number of records: " << index << endl;

#pragma region ParallelMergeSortKnn
	cout << "\n\Taskflow KNN: " << endl;
	steady_clock::time_point start = steady_clock::now();
	TaskflowParallelKnn parallelKnn(3); // Use K=3

	int parallelPrediction = parallelKnn.predict_class(dataset, target, dataset_size, feature_size);
	cout << "Taskflow Prediction: " << parallelPrediction << endl;

	if (parallelPrediction == 0) {
		cout << "Predicted class: Negative" << endl;
	}
	else if (parallelPrediction == 1) {
		cout << "Predicted class: Prediabetes or Diabetes" << endl;
	}
	else {
		cout << "Prediction could not be made." << endl;
	}

	steady_clock::time_point e = steady_clock::now();
	time_parallel_knn = duration_cast<std::chrono::microseconds>(e - start).count();
	cout << "Classification Time = " << time_parallel_knn << "[�s]" << endl;
#pragma endregion


	//Knn
#pragma region SerialMergeSortKnn
	cout << "\n\nSerial KNN: " << endl;
	steady_clock::time_point knnBegin = steady_clock::now();
	SerialMergeSortKnn knn(3); // Use K=3

	int prediction = knn.predict_class(dataset, target, dataset_size, feature_size);
	cout << "Prediction: " << prediction << endl;

	if (prediction == 0) {
		cout << "Predicted class: Negative" << endl;
	}
	else if (prediction == 1) {
		cout << "Predicted class: Prediabetes or Diabetes" << endl;
	}
	else {
		cout << "Prediction could not be made." << endl;
	}

	steady_clock::time_point knnEnd = steady_clock::now();
	time_serial_knn = duration_cast<microseconds>(knnEnd - knnBegin).count();
	cout << "Classification Time = " << time_serial_knn << "[�s]" << endl;
#pragma endregion

	time_reduce = time_serial_knn - time_parallel_knn;

	cout << "\n\nThe speed of classification is " << time_reduce << " faster" << endl;

	return 0;
}