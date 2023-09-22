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
		//Count label occurrences in the K nearest neighbors
		int count = 0;
		for (int i = 0; count < neighbours_number; i++) {
			if (distances[1][i] == 0 && distances[0][i] > 0) {
				zeros_count += 1;
				cout << "0: " << distances[0][i] << endl;
				//cout << "0: " << distances[0][i] << "," << distances[2][i] << endl;
				count++;
			}
			else if (distances[1][i] == 1 && distances[0][i] > 0) {
				ones_count += 1;
				cout << "1: " << distances[0][i] << endl;
				//cout << "1: " << distances[0][i] << "," << distances[2][i] << endl;
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
#pragma region Knn
	cout << "\nKNN: " << endl;
	chrono::steady_clock::time_point knnBegin = chrono::steady_clock::now();
	Knn knn(k_value); // Use K=3

	int prediction = knn.predict_class(dataset, target, dataset_size, feature_size);
	cout << "KNN Prediction: " << prediction << endl;

	if (prediction == 0) {
		cout << "Predicted class: Negative" << endl;
	}
	else if (prediction == 1) {
		cout << "Predicted class: Prediabetes or Diabetes" << endl;
	}
	else {
		cout << "Prediction could not be made." << endl;
	}

	chrono::steady_clock::time_point knnEnd = chrono::steady_clock::now();
	cout << "Classification Time = " << chrono::duration_cast<chrono::microseconds>(knnEnd - knnBegin).count() << "[µs]" << endl;

#pragma endregion

	// Deallocate memory for dataset
	for (int i = 0; i < dataset_size; i++) {
		delete[] dataset[i];
	}

	return 0;
}
