#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>


bool sort_by_dist(const std::vector<double>& v1, const std::vector<double>& v2);

class Knn {
private:
    int neighbours_number;

public:
    Knn(int k) : neighbours_number(k) {}

    int predict_class(const std::vector<std::vector<double>>& dataset, const std::vector<double>& target) {
        std::vector<std::vector<double>> distances;
        int zeros_count = 0;
        int ones_count = 0;
        int prediction = -1;

        distances = this->get_knn(dataset, target);
        for (const auto& pair : distances) {
            if (pair[1] == 0) {
                zeros_count += 1;
            }
            if (pair[1] == 1) {
                ones_count += 1;
            }
        }

        if (zeros_count > ones_count) {
            prediction = 0;
        }
        else{
            prediction = 1;
        }

        return prediction;
    }

private:
    double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y) {
        double l2 = 0.0;
        for (size_t i = 1; i < x.size(); i++) { // Start from index 1 to exclude outcome
            l2 += std::pow((x[i] - y[i]), 2);
        }
        return std::sqrt(l2);
    }

    std::vector<std::vector<double>> get_knn(const std::vector<std::vector<double>>& x, const std::vector<double>& y) {
        std::vector<std::vector<double>> l2_distances;
        double l2 = 0.0;
        int count = 0;
        for (const auto& sample : x) {
            if (sample == y) continue; // do not use the same point 
            std::vector<double> neighbour; // structure: distance, label
            l2 = this->euclidean_distance(y, sample);
            neighbour.push_back(l2);
            neighbour.push_back(sample[0]); // Store outcome label
            l2_distances.push_back(neighbour);
            count++;
        }
        std::cout <<"Number of euclidean run:" << count << std::endl;

        std::sort(l2_distances.begin(), l2_distances.end(), sort_by_dist);
        std::vector<std::vector<double>> d(l2_distances.begin(), l2_distances.begin() + this->neighbours_number);
        return d;
    }
};

bool sort_by_dist(const std::vector<double>& v1, const std::vector<double>& v2) {
    return v1[0] < v2[0];
}

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

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::string filename = "diabetes_binary_5050.csv"; // Replace with your CSV file name

    std::vector<std::vector<double>> dataset;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 1;
    }

    // Read and skip the header line
    std::string header;
    std::getline(file, header);

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row = parseLine(line);
        dataset.push_back(row);
    }

    std::cout << "Number of records: " << dataset.size() << std::endl;

    //0
    //std::vector<double> target = {0.0,1.0,0.0,1.0,26.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,3.0,5.0,30.0,0.0,1.0,4.0,6.0,8.0};
    std::vector<double> target = {0.0,0.0,0.0,1.0,24.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0,0.0,1.0,3.0,0.0,0.0,0.0,2.0,5.0,3.0};
    //1
    //std::vector<double> target = { 1.0,1.0,0.0,1.0,30.0,1.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,2.0,0.0,0.0,0.0,1.0,11.0,5.0,7.0 };
    //std::vector<double> target = { 1.0,1.0,0.0,1.0,19.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,4.0,0.0,10.0,1.0,0.0,9.0,4.0,2.0 };
    //std::vector<double> target = { 1.0,1.0,1.0,1.0,52.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,5.0,0.0,30.0,1.0,1.0,8.0,3.0,2.0 };
    
    Knn knn(3); // Use K=3

    int prediction = knn.predict_class(dataset, target);
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


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    return 0;
}
