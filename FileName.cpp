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
#include <agents.h>
#include <ppltasks.h>

using namespace std;
using namespace concurrency;
class Knn {
private:
	int neighbours_number;

public:
	Knn(int k) : neighbours_number(k) {}

	int predict_class(concurrent_vector<vector<double>> dataset, const vector<double> target, int dataset_size, int feature_size) {

		int zeros_count = 0;
		int ones_count = 0;

		get_knn(dataset, target, dataset_size, feature_size);


		


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

		return prediction;
	}

private:
	double euclidean_distance(const vector<double> x, const vector<double> y, int feature_size) {
		double l2 = 0.0;
		for (int i = 1; i < feature_size; i++) {
			l2 += pow((x[i] - y[i]), 2);
		}
		return  sqrt(l2);
	}



	concurrent_vector<vector<double>> get_knn(concurrent_vector<vector<double>> dataset, const vector<double> target, int dataset_size, int feature_size) {
		concurrent_vector<vector<double>> euclideanList;
		parallel_for(0, dataset_size, [&](int i) {
			if (dataset.at(i) != target) {
				euclideanList.push_back({this->euclidean_distance(dataset.at(i), target, feature_size), dataset.at(i).at(0) });
			}
			});
	cout << "Number of euclidean run:" << euclideanList.size() << endl;
	return euclideanList;
}

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

	chrono::steady_clock::time_point begin = chrono::steady_clock::now();
	//Knn knn(3); // Use K=3

	//int prediction = knn.predict_class(dataset, target, dataset_size, feature_size);
	//cout << "Prediction: " << prediction << endl;

	//if (prediction == 0) {
	//	cout << "Predicted class: Negative" << endl;
	//}
	//else if (prediction == 1) {
	//	cout << "Predicted class: Prediabetes or Diabetes" << endl;
	//}
	//else {
	//	cout << "Prediction could not be made." << endl;
	//}

	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	cout << "Time difference = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;



	return 0;
}
