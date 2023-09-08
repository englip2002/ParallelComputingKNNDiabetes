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

struct Output
{
	concurrent_vector<double> nearestDistanceFound;
	int prediction;
};

class Knn {
private:
	int neighbours_number;
	
public:
	Knn(int k) : neighbours_number(k) {}
	concurrent_vector<double> compare(concurrent_vector<double> sortingTarget, int startingNum) {
		concurrent_vector<double> smallerGroup;
		int maxIterations = sortingTarget.size();
		parallel_for(1, maxIterations, [&, sortingTarget, startingNum](int i) {
			if (sortingTarget.at(i) < sortingTarget.at(startingNum)) {
				smallerGroup.push_back(sortingTarget.at(i));
			}
			});
		if (smallerGroup.size() == 0)
		{
			smallerGroup.push_back(sortingTarget.at(0));
		}
		return smallerGroup;
	}
	concurrent_vector<double> parallelNthElement(concurrent_vector<double> sortingTarget) {
		concurrent_vector<double> sorted;
		concurrent_vector<double> placeholder = compare(sortingTarget, sorted.size());
		while (sorted.size() != neighbours_number)
		{
			if (placeholder.size() == neighbours_number)
			{
				sorted = placeholder;
			}
			else if (placeholder.size() < neighbours_number)
			{
				for (int i = 0; i < placeholder.size(); i++)
				{
					sorted.push_back(placeholder[i]);
				}
				placeholder = compare(placeholder, sorted.size());
			}
			else
			{
				placeholder = compare(placeholder, sorted.size());
			}
		}
		return sorted;
	}

	int predict_class_serial(concurrent_vector<vector<double>> dataset, vector<double> target, int dataset_size, int feature_size) {
		concurrent_vector<double> euclideanDistance;
		int zeros_count = 0;
		int ones_count = 0;
		chrono::steady_clock::time_point beginTime = chrono::steady_clock::now();
		for (int rowNum = 0; rowNum < dataset_size; rowNum++)
		{
			if (dataset.at(rowNum) != target) {
				double l2 = 0.0;
				for (int i = 1; i < feature_size; i++)
				{
					l2 += pow((dataset.at(rowNum).at(i) - target[i]), 2);
				}
				euclideanDistance.push_back(round(sqrt(l2) * 10000) + (dataset.at(rowNum).at(0) + 1));
			}
		}

		cout << "Number of euclidean run:" << euclideanDistance.size() << endl;

		sort(begin(euclideanDistance), end(euclideanDistance));

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
		chrono::steady_clock::time_point endTime = chrono::steady_clock::now();
		cout << "Time difference of serial KNN= " << chrono::duration_cast<chrono::microseconds>(endTime - beginTime).count() << "[µs]" << endl;
		return prediction;
	}
	Output predict_class_parallel_for(vector<vector<double>> dataset, vector<double> target, int dataset_size, int feature_size) {
		concurrent_vector<double> euclideanDistance;
		concurrent_vector<double> kNearestNeighbour;
		int zeros_count = 0;
		int ones_count = 0;


		chrono::steady_clock::time_point beginTime = chrono::steady_clock::now();

		parallel_for(0, dataset_size, [dataset, target, &euclideanDistance, feature_size](int value) {
			if (dataset.at(value) != target) {
				double l2 = 0.0;
				for (int i = 1; i < feature_size; i++)
				{
					l2 += pow((dataset.at(value).at(i) - target[i]), 2);
				}
				euclideanDistance.push_back(round(sqrt(l2) * 10000) + (dataset.at(value).at(0)/10 + 0.1) );
			}
			}, static_partitioner());

		kNearestNeighbour = parallelNthElement(euclideanDistance);
		

		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (fmod(kNearestNeighbour[i], 2) == 1) {
				zeros_count += 1;
			}
			else if (fmod(kNearestNeighbour[i], 2) == 0) {
				ones_count += 1;
			}
		}
		int prediction = (zeros_count > ones_count) ? 0 : 1;
		chrono::steady_clock::time_point endTime = chrono::steady_clock::now();
		cout << "Time difference of parallerized KNN prediction = " << chrono::duration_cast<chrono::microseconds>(endTime - beginTime).count() << "[µs]" << endl;

		return { euclideanDistance, prediction };
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
	const int dataset_size = 100000;
	const int feature_size = 22;

	concurrent_vector<vector<double>> dataset;
	vector<vector<double>> vecdataset;
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
		vecdataset.push_back(row);
		index++;
	}
	cout << "Number of records: " << index << endl;

	Knn KNN(3); // Use K=3
	Output prediction = KNN.predict_class_parallel_for(vecdataset, target, dataset_size, feature_size);
	cout << "Total Euclidean runs : " << prediction.nearestDistanceFound.size() << endl;
	if (prediction.prediction == 0) {
		cout << "Predicted class: Negative" << endl;
	}
	else if (prediction.prediction == 1) {
		cout << "Predicted class: Prediabetes or Diabetes" << endl;
	}
	else {
		cout << "Prediction could not be made." << endl;
	}
	

	int serialPrediction = KNN.predict_class_serial(dataset, target, dataset_size, feature_size);
	cout << "Prediction: " << serialPrediction << endl;

	if (serialPrediction == 0) {
		cout << "Predicted class: Negative" << endl;
	}
	else if (serialPrediction == 1) {
		cout << "Predicted class: Prediabetes or Diabetes" << endl;
	}
	else {
		cout << "Prediction could not be made." << endl;
	}
	return 0;
}
