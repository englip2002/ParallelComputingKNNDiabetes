#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
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
//output format structure
struct Output
{
	concurrent_vector<double> nearestDistanceFound;
	int prediction;
};

struct euclideanDistance
{
	double distance;
	double label;
};

class Knn {
private:
	int neighbours_number;

public:
	Knn(int k) : neighbours_number(k) {}
	//parallel nth element
	void compare(concurrent_vector<double> sortingTarget, int startingNum, concurrent_vector<double>* smallerGroup, concurrent_vector<double>* largerGroup) {

		int maxIterations = sortingTarget.size();
		parallel_for(startingNum + 1, maxIterations, [&](int i) {
			if (sortingTarget.at(i) < sortingTarget.at(startingNum)) {
				(*smallerGroup).push_back(sortingTarget.at(i));
			}
			else {
				(*largerGroup).push_back(sortingTarget.at(i));
			}
			});
		if ((*smallerGroup).size() == 0)
		{
			(*smallerGroup).push_back(sortingTarget.at(startingNum));
		}
		else {
			(*largerGroup).push_back(sortingTarget.at(startingNum));
		}
		return;
	}
	concurrent_vector<double> parallelNthElement(concurrent_vector<double> sortingTarget) {
		concurrent_vector<double> reduced = sortingTarget;
		concurrent_vector<double> smallerGroup, largerGroup;
		concurrent_vector<double> temp;

		do {
			// Partition into smaller and larger
			smallerGroup.clear();
			largerGroup.clear();
			compare(reduced, 0, &smallerGroup, &largerGroup);

			// Append previous smaller grp into current smaller grp
			for (int i = 0; i < temp.size(); i++) {
				smallerGroup.push_back(temp.at(i));
			}
			temp.clear();
			// If smaller.size == N, return
			if (smallerGroup.size() == neighbours_number) {
				return smallerGroup;
			}
			// If smaller.size > N, continue partition smaller grp
			else if (smallerGroup.size() > neighbours_number) {
				reduced = smallerGroup;
			}
			// If smaller.size < N, save smaller grp and continue partition larger grp
			else {
				reduced = largerGroup;
				temp = smallerGroup;
			}

		} while (true);
	}

	//KNN source code
	int predict_class_serial(vector<vector<double>> dataset, vector<double> target, int dataset_size, int feature_size) {
		vector<double> euclideanDistance;
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
				euclideanDistance.push_back(sqrt(l2));
			}
		}

		cout << "Number of euclidean run:" << euclideanDistance.size() << endl;

		sort(begin(euclideanDistance), end(euclideanDistance));

		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			cout << euclideanDistance[i] << endl;
			if (euclideanDistance[i] == 0.0) {
				zeros_count += 1;
			}
			else if (euclideanDistance[i] == 0.1) {
				ones_count += 1;
			}
		}
		int prediction = (zeros_count > ones_count) ? 0 : 1;
		chrono::steady_clock::time_point endTime = chrono::steady_clock::now();
		cout << "Time difference of serial KNN= " << chrono::duration_cast<chrono::microseconds>(endTime - beginTime).count() << "[µs]" << endl;
		return prediction;
	}
	Output predict_class_parallel_for(concurrent_vector<vector<double>> dataset, concurrent_vector<double> target, int dataset_size, int feature_size) {
		concurrent_vector<double> euclideanDistance;
		concurrent_vector<double> kNearestNeighbour;
		int zeros_count = 0;
		int ones_count = 0;
		chrono::steady_clock::time_point beginTime = chrono::steady_clock::now();

		//calculate all euclidean distance
		parallel_for(0, dataset_size, [&dataset, &target, &euclideanDistance, feature_size](int value) {
			double l2 = 0.0;
			double distance;
			char str[20];
			for (int i = 1; i < feature_size; i++)
			{
				l2 += pow((dataset.at(value).at(i) - target[i]), 2);

			}
			//round of result to 4 decimal places
			distance = round(sqrt(l2) * 10000) / 10000;
			if (distance > 0)
			{
				// compress euclidean distance and label of point
				euclideanDistance.push_back(distance + (dataset.at(value).at(0) / 1000000 + 0.000001));
			}
			});

		//sort euclidean distance
		kNearestNeighbour = parallelNthElement(euclideanDistance);

		
		// Count label occurrences in the K nearest neighbors
		for (int i = 0; i < neighbours_number; i++) {
			if (fmod(kNearestNeighbour[i] * 1000000, 2) == 1) {
				zeros_count += 1;
			}
			else if (fmod(kNearestNeighbour[i] * 1000000, 2) == 0) {
				ones_count += 1;
			}
		}
		int prediction = (zeros_count > ones_count) ? 0 : 1;
		chrono::steady_clock::time_point endTime = chrono::steady_clock::now();
		cout << "Time used by program = " << chrono::duration_cast<chrono::microseconds>(endTime - beginTime).count() << "[µs]" << endl;
		return { kNearestNeighbour, prediction };
	}
};

//self defined methods
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

#pragma region InitVariable
	string filename = "diabetes_binary.csv";

	//const int dataset_size = 253681; 
	const int dataset_size = 250000;
	const int feature_size = 22;
	vector<double> target = { 0.0, 0.0, 0.0, 1.0, 24.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0, 5.0, 3.0 };
	concurrent_vector<double> conTarget;
	for (size_t i = 0; i < target.size(); i++)
	{
		conTarget.push_back(target[i]);
	}
	concurrent_vector<vector<double>> conDataset;
	vector<vector<double>> vecdataset;
#pragma endregion
#pragma region LoadDataset
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
		conDataset.push_back(row);
		vecdataset.push_back(row);
		index++;
	}
	cout << "Number of records: " << index << endl;
#pragma endregion
	Knn KNN(3); // Use K=3
	Output prediction = KNN.predict_class_parallel_for(conDataset, conTarget, dataset_size, feature_size);
	
	//output formating
	for (int i = 0; i < prediction.nearestDistanceFound.size(); i++)
	{
		// decompressed output to find label and euclidean distance
		cout << "Closest distance calculated (unordered) : " << setprecision(6) << prediction.nearestDistanceFound[i] << endl;
		if (fmod(prediction.nearestDistanceFound[i] * 10000000, 2) == 1) {
			cout << "Corresponding label : false" << endl;
		}
		else
		{
			cout << "Corresponding label : true" << endl;
		}
	}
	if (prediction.prediction == 0) {
		cout << "Predicted class: Negative" << endl;
	}
	else if (prediction.prediction == 1) {
		cout << "Predicted class: Prediabetes or Diabetes" << endl;
	}
	else {
		cout << "Prediction could not be made." << endl;
	}



	return 0;
}
