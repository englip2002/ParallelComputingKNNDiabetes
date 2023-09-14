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
using namespace std;

const int num_threads = 8;
const int best_record_each_thread = 5;
const int num_record_to_sort = num_threads * best_record_each_thread;
const int k_value = 3;

struct PthreadParams {
	double** dataset;
	const double* target;
	double** distances;
	int dataset_size;
	int feature_size;
	int start;
	int end;
	int thread_id;
};

struct quickSortParams {
	double** distances;
	int low;
	int high;
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
		distances[2] = new double[dataset_size];

		//chrono::steady_clock::time_point knnBegin = chrono::steady_clock::now();
		get_knn(dataset, target, distances, dataset_size, feature_size);
		//chrono::steady_clock::time_point knnEnd = chrono::steady_clock::now();
		//cout << "KNN = " << chrono::duration_cast<chrono::microseconds>(knnEnd - knnBegin).count() << "[µs]" << endl;

		//for (int i = 0; i < 100; i++) {
		//	cout << i + 1 <<") " << distances[0][i] << ", " << distances[1][i] << ", " << distances[2][i] << endl;
		//}

#pragma region quickSorting
		// creating 4 threads
		pthread_t sortThreads[num_threads];
		quickSortParams sortParams[num_threads];

		int rows_per_thread = dataset_size / num_threads;
		for (int i = 0; i < num_threads; i++) {
			//calculate the start and end for each thread
			int low = i * rows_per_thread;
			int high = (i == num_threads - 1) ? dataset_size : (i + 1) * rows_per_thread;
			sortParams[i] = { distances, low, high };

			pthread_create(&sortThreads[i], NULL, quick_sort_parallel, &sortParams[i]);
		}

		for (int i = 0; i < num_threads; i++) {
			pthread_join(sortThreads[i], nullptr);
		}

		double* finalSortedDistances[3];
		finalSortedDistances[0] = new double[num_record_to_sort];
		finalSortedDistances[1] = new double[num_record_to_sort];
		finalSortedDistances[2] = new double[num_record_to_sort];

		//extract first 5 from each thread (shortest distance for each thread)
		//i < 3 because distance is 2d array and distance[3][i] is largest
		for (int i = 0; i < 3; i++) {
			// k < num_thread is to extract value from each thread
			for (int k = 0; k < num_threads; k++) {
				// to extract first 5 record from each thread
				for (int j = 0; j < best_record_each_thread; j++) {
					finalSortedDistances[i][(k * best_record_each_thread) + j] = distances[i][((dataset_size / num_threads) * k) + j];
				}
			}
		}

		//for (int i = 0; i < num_record_to_sort; i++) {
		//	cout << finalSortedDistances[0][i] << "," << finalSortedDistances[1][i] << "," << finalSortedDistances[2][i] << endl;
		//}
		//cout << "\n";

		//sort again
		//the number of record need to serial sort is rapidly decreased
		quick_sort(finalSortedDistances, 0, num_record_to_sort - 1);

		//for (int i = 0; i < num_record_to_sort; i++) {
		//	cout << finalSortedDistances[0][i] << "," << finalSortedDistances[1][i] << "," << finalSortedDistances[2][i] << endl;
		//}

#pragma endregion

		int zeros_count = 0;
		int ones_count = 0;

		cout << "First K value: " << endl;
		//Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (finalSortedDistances[1][i] == 0) {
				zeros_count += 1;
				cout << "0: " << finalSortedDistances[0][i] << endl;
			}
			else if (finalSortedDistances[1][i] == 1) {
				ones_count += 1;
				cout << "1: " << finalSortedDistances[0][i] << endl;
			}
		}

		//return prediction
		int prediction = (zeros_count > ones_count) ? 0 : 1;

		// Clean up memory
		delete[] distances[0];
		delete[] distances[1];
		delete[] distances[2];
		delete[] finalSortedDistances[0];
		delete[] finalSortedDistances[1];
		delete[] finalSortedDistances[2];
		return prediction;
	}


private:
	static void* quick_sort_parallel(void* arg) {
		//recieve parameter
		quickSortParams* params = static_cast<quickSortParams*>(arg);

		//call sort
		quick_sort(params->distances, params->low, params->high - 1);

		return nullptr;
	}

	// Function to partition the array into two sub-arrays and return the index of the pivot element
	static int partition(double** distances, int low, int high) {
		// Choose the last element as the pivot
		double pivot = distances[0][high];
		// Index of the smaller element
		int i = low - 1;
		//check if value of distance[0] is lesser than pivot value, if yes swap
		//to move shorter distance to left
		for (int j = low; j < high; j++) {
			if (distances[0][j] <= pivot) {
				i++;
				swap(distances, i, j);
			}
		}
		//placing the pivot in the correct position
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
			// Recursively sort the sub-arrays
			quick_sort(distances, low, pivotIndex - 1);
			quick_sort(distances, pivotIndex + 1, high);
		}
	}

	//to calculate euclidean distance
	static double euclidean_distance(const double* x, const double* y, int feature_size) {
		double l2 = 0.0;
		//loop through each label in a row to calculate distance
		for (int i = 1; i < feature_size; i++) {
			l2 += pow((x[i] - y[i]), 2);
		}
		return sqrt(l2);
	}

	//function to be parse to pthread for multi-threading
	static void* compute_distances(void* arg) {
		//recieve parameters
		PthreadParams* params = static_cast<PthreadParams*>(arg);
		int count = 0;

		//different thread is accessing different index range, so no race condition
		for (int i = params->start; i < params->end; i++) {
			if (params->dataset[i] == params->target) continue; // do not use the same point
			params->distances[0][i] = euclidean_distance(params->target, params->dataset[i], params->feature_size);
			params->distances[1][i] = params->dataset[i][0]; // Store outcome label
			params->distances[2][i] = i; // Store index
			count++;
		}
		//cout << "Thread " << params->thread_id << " - Number of euclidean run: " << count << endl;

		return nullptr;
	}

	//the function to be call to get KNN 
	void get_knn(double* x[], const double* y, double* distances[3], int dataset_size, int feature_size) {
		//create parameters to be parse to compute_distance function
		PthreadParams knnParams[num_threads];
		pthread_t knnThreads[num_threads];

		//to calculate the number of dataset need to handled by each thread
		int rows_per_thread = dataset_size / num_threads;

		for (int i = 0; i < num_threads; i++) {
			//assign start and end point for each thread
			int start = i * rows_per_thread;
			int end = (i == num_threads - 1) ? dataset_size : (i + 1) * rows_per_thread;

			//store parameter
			knnParams[i] = { x, y, distances, dataset_size, feature_size, start, end ,i };
			//create and assign task to thread
			pthread_create(&knnThreads[i], nullptr, compute_distances, &knnParams[i]);
		}

		//wait all thread to complete
		for (int i = 0; i < num_threads; i++) {
			pthread_join(knnThreads[i], nullptr);
		}
	}
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

		//for (int i = 0; i < 100; i++) {
		//	cout << i + 1 << ") " << distances[0][i] << ", " << distances[1][i] << ", " << distances[2][i] << endl;
		//}

		selectionSort(distances, dataset_size);

		/*for (int i = 0; i < num_record_to_sort; i++) {
			cout << distances[0][i] << "," << distances[1][i] << "," << distances[2][i] << endl;
		}*/

		cout << "First K value: " << endl;
		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (distances[1][i] == 0) {
				zeros_count += 1;
				cout << "0: " << distances[0][i] << endl;
			}
			else if (distances[1][i] == 1) {
				ones_count += 1;
				cout << "1: " << distances[0][i] << endl;
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
		//cout << "Number of euclidean run:" << count << endl;
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

	//const int dataset_size = 30000;
	//const int dataset_size = 100000;
	const int dataset_size = 250000;
	const int feature_size = 22;

	double** dataset = new double* [dataset_size];
	double target[feature_size] = { 0.0, 0.0, 0.0, 1.0, 24.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 5.0, 3.0 };
	//double target[feature_size] = { 1.0, 1.0, 1.0, 1.0, 30.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 5.0, 30.0, 30.0, 1.0, 0.0, 9.0, 5.0, 1.0 };
	//double target[feature_size] = { 1.0, 0.0, 0.0, 1.0, 25.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 13.0, 6.0, 8.0 };
	//double target[feature_size] = { 0.0, 1.0, 1.0, 1.0, 28.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 4.0, 0.0, 10.0, 1.0, 0.0, 12.0, 6.0, 2.0 };

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
	//to eliminate first line which is the header
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

	cout << "Number of records: " << index << endl;

		//Pthread Knn
	#pragma region PthreadKnn
		cout << "\nPthread KNN: " << endl;
		chrono::steady_clock::time_point pthreadBegin = chrono::steady_clock::now();
	
		PthreadKnn pthreadknn(k_value); // Use K=3
		int pthreadPrediction = pthreadknn.predict_class(dataset, target, dataset_size, feature_size);
		cout << "Pthread Prediction: " << pthreadPrediction << endl;
	
		if (pthreadPrediction == 0) {
			cout << "Predicted class: Negative" << endl;
		}
		else if (pthreadPrediction == 1) {
			cout << "Predicted class: Prediabetes or Diabetes" << endl;
		}
		else {
			cout << "Prediction could not be made." << endl;
		}
	
		chrono::steady_clock::time_point pthreadEnd = chrono::steady_clock::now();
		cout << "Classification Time = " << chrono::duration_cast<chrono::microseconds>(pthreadEnd - pthreadBegin).count() << "[µs]" << endl;
	#pragma endregion

		//Knn
//#pragma region Knn
//	cout << "\nKNN: " << endl;
//	chrono::steady_clock::time_point knnBegin = chrono::steady_clock::now();
//	Knn knn(k_value); // Use K=3
//
//	int prediction = knn.predict_class(dataset, target, dataset_size, feature_size);
//	cout << "KNN Prediction: " << prediction << endl;
//
//	if (prediction == 0) {
//		cout << "Predicted class: Negative" << endl;
//	}
//	else if (prediction == 1) {
//		cout << "Predicted class: Prediabetes or Diabetes" << endl;
//	}
//	else {
//		cout << "Prediction could not be made." << endl;
//	}
//
//	chrono::steady_clock::time_point knnEnd = chrono::steady_clock::now();
//	cout << "Classification Time = " << chrono::duration_cast<chrono::microseconds>(knnEnd - knnBegin).count() << "[µs]" << endl;
//
//#pragma endregion

	//cout << "The speed of classification is " << (double)((knnEnd - knnBegin) / (pthreadEnd - pthreadBegin)) << " Times fasters" << endl;

	// Deallocate memory for dataset
	for (int i = 0; i < dataset_size; i++) {
		delete[] dataset[i];
	}

	return 0;
}
