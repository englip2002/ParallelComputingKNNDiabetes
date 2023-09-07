#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#include <ppl.h>
#include <concurrent_vector.h>
#include <concurrent_unordered_set.h>
#include <ppltasks.h>

using namespace std;
using namespace concurrency;
class Knn {
private:
	int neighbours_number;

public:
	Knn(int k) : neighbours_number(k) {}

	int predict_class_parallel_algorithm(concurrent_vector<vector<double>> dataset, const vector<double> target, int dataset_size, int feature_size) {
		concurrent_vector<double> euclideanDistance;
		int zeros_count = 0;
		int ones_count = 0;
		

		/*for (int i = 0; i < dataset_size; i++)
		{
			if (dataset.at(i) != target) {
				euclideanList.push_back(euclidean_distance(dataset.at(i), target, feature_size));
			}
		}*/
		parallel_for(0, dataset_size, [&](int value) {
			if (dataset.at(value) != target) {
				double l2 = 0.0;
				for (int i = 1; i < feature_size; i++)
				{
					l2 += pow((dataset.at(value).at(i) - target[i]), 2);
				}
				euclideanDistance.push_back(round(sqrt(l2) * 10000) + (dataset.at(value).at(0) + 1));
			}
			});
		cout << "Number of euclidean run:" << euclideanDistance.size() << endl;
		


		parallel_sort(begin(euclideanDistance),end(euclideanDistance));
		//nth_element(begin(euclideanDistance), begin(euclideanDistance) + neighbours_number, end(euclideanDistance));

		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (fmod(euclideanDistance[i], 2) == 1) {
				zeros_count += 1;
			}
			else if (fmod(euclideanDistance[i], 2) == 0) {
				ones_count += 1;
			}
		}

		int prediction = (zeros_count > ones_count) ? 0 : 1;

		return prediction;
	}

private:
	

};
class
	vector<double> parseLine(const  string& line) {
	vector<double> row;
	istringstream iss(line);
	string value;

	while (getline(iss, value, ',')) {
		try {
			double num = stod(value);
			row.push_back(num);
		}
		catch (const  invalid_argument&) {
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

	concurrent_vector<vector<double>> dataset;
	vector<double> target = { 0.0, 0.0, 0.0, 1.0, 24.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 5.0, 3.0 };

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
		dataset.push_back(row);
		index++;
	}
	cout << "Number of records: " << index << endl;

	Knn knn(5); // Use K=3
	chrono::steady_clock::time_point begin = chrono::steady_clock::now();
	int prediction = knn.predict_class_parallel_algorithm(dataset, target, dataset_size, feature_size);
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	cout << "Time difference of KNN using Parallel Algorithm = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;
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


	return 0;
}
