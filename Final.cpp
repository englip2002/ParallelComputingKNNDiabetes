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

#pragma warning(disable:4146)

using namespace std;
using namespace chrono;
using namespace tf;

const int num_tasks = 4;

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
		distances[0] = new double[dataset_size];
		distances[1] = new double[dataset_size];
		distances[2] = new double[dataset_size];

		//get_knn(dataset, target, distances, dataset_size, feature_size);

		Taskflow taskflow;
		Executor executor;


		taskflow.for_each_index(0, dataset_size, 1, [&](int i) {

			int count = 0;
			/*double l2 = 0.0;
			for (int j = 1; j < feature_size; j++) {
				l2 += pow((target[j] - dataset[i][j]), 2);
			}
			distances[0][i] = sqrt(l2);*/
			distances[0][i] = euclidean_distance(target, dataset[i], feature_size);
			distances[1][i] = dataset[i][0]; // Store outcome label
			distances[2][i] = i; // Store index
			count++;
			});

		executor.run(taskflow).wait();

		taskflow.clear();

		//tf.for_each_index(0, num_tasks, 1, [&](int t) {
		//	int count = 0;

		//	int chunk_size = dataset_size / num_tasks;   
		//	int start_idx = t * chunk_size;  
		//	int end_idx = (t == num_tasks - 1) ? dataset_size : (start_idx + chunk_size);

		//	for (int i = start_idx; i < end_idx; i++) {
		//		if (dataset[i] == target) continue; // do not use the same point
		//		double l2 = 0.0;
		//		for (int j = 1; j < feature_size; j++) {
		//			l2 += pow((target[j] - dataset[i][j]), 2);
		//		}
		//		distances[0][i] = sqrt(l2);
		//		//distances[0][i] = euclidean_distance(target, dataset[i], feature_size);
		//		distances[1][i] = dataset[i][0]; // Store outcome label
		//		distances[2][i] = i; // Store index
		//		count++;
		//	}

		//	cout << "Task " << t << " - Number of euclidean run: " << count << endl;

		//	});
	
		

		/*auto merge_sort_task = [=, &distances]() {
			merge_sort(distances, 0, dataset_size - 1);
		};*/
		//tf.emplace(merge_sort_task);



		/*int chunk_size = dataset_size / num_tasks;

		for (int i = 0; i < num_tasks; i++) {
			int start = i * (chunk_size);
			int end = (i == num_tasks - 1) ? (dataset_size) : ((i + 1) * chunk_size);

			tf.emplace([&] {
				merge_sort(distances, start, end - 1);
				});

		}*/

//#pragma region Sorting
//		int* index_order = new int[dataset_size];
//		for (int i = 0; i < dataset_size; ++i) {
//			index_order[i] = i;
//		}
//
//		auto compare_function = [&distances](int i, int j) {
//			double diff = distances[0][i] - distances[0][j];
//
//			return distances[0][j] > distances[0][i];
//
//			//return (diff < 0) ? -1 : (diff > 0) ? 1 : 0;
//		};
//
//		taskflow.emplace([&, index_order]() {
//			taskflow.sort(index_order, index_order + dataset_size, compare_function);
//			});
//		
//
//		executor.run(taskflow).wait();
//	
//
//		for (int i = 0; i < 10; i++) {
//			cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << std::endl;
//		}
//
//#pragma endregion

		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (distances[1][i] == 0) {
				zeros_count += 1;
				cout << "0: " << distances[0][i] << "," << distances[2][i] << endl;
			}
			else if (distances[1][i] == 1) {
				ones_count += 1;
				cout << "1: " << distances[0][i] << "," << distances[2][i] << endl;
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
			l2 += pow((x[i] - y[i]), 2);
		}
		return sqrt(l2);
	}



	//void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {

	//	Taskflow tf;
	//	Executor executor;

	//	int count = 0;
	//	/*int start = 0;
	//	int end = dataset_size - 1;*/

	//	/*auto init = tf.emplace([&]() {
	//		int count = 0;
	//		int start = 0;
	//		int end = dataset_size;
	//		});*/



	//	tf.for_each_index(0, dataset_size, 1, [&](int i) {
	//		for (int i = 0; i < dataset_size - 1; i++) {
	//			if (x[i] == y) continue; // do not use the same point
	//			double l2 = 0.0;
	//			for (int j = 1; j < feature_size; j++) {
	//				l2 += pow((y[j] - x[i][j]), 2);
	//			}
	//			distances[0][i] = sqrt(l2);
	//			//distances[0][i] = euclidean_distance(target, dataset[i], feature_size);
	//			distances[1][i] = x[i][0]; // Store outcome label
	//			distances[2][i] = i; // Store index
	//			//cout << "A" << endl;
	//			count++;
	//		}
	//		});
	//	//init.precede(tf);
	//	executor.run(tf);
	//	cout << "Number of euclidean run:" << count << endl;
	//	}
	

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

		/*merge_sort(distances, 0, dataset_size - 1);*/

		/*for (int i = 0; i < 10; i++) {
			cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << std::endl;
		}*/

		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (distances[1][i] == 0) {
				zeros_count += 1;
				cout << "0: " << distances[0][i] << "," << distances[2][i] << endl;
			}
			else if (distances[1][i] == 1) {
				ones_count += 1;
				cout << "1: " << distances[0][i] << "," << distances[2][i] << endl;
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
	cout << "\n\nParallel KNN: " << endl;
	steady_clock::time_point start = steady_clock::now();
	TaskflowParallelKnn parallelKnn(3); // Use K=3

	int parallelPrediction = parallelKnn.predict_class(dataset, target, dataset_size, feature_size);
	cout << "Prediction: " << parallelPrediction << endl;

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
	cout << "Time difference = " << duration_cast<std::chrono::microseconds>(e - start).count() << "[µs]" << endl;
#pragma endregion


	//Knn
#pragma region SerialMergeSortKnn
	cout << "\n\nSerial Merge Sort KNN: " << endl;
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
	cout << "Time difference = " << duration_cast<microseconds>(knnEnd - knnBegin).count() << "[µs]" << endl;
#pragma endregion

	return 0;
}
