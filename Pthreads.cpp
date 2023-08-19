#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <pthread.h>

bool sort_by_dist(const std::vector<double>& v1, const std::vector<double>& v2);
struct ThreadParams {
    const std::vector<std::vector<double>>* x;
    const std::pair<int, int>* range;
    const std::vector<double>* y;
};

class Knn {
private:
    int neighbours_number;
    int num_threads;

public:
    Knn(int k, int threads) : neighbours_number(k), num_threads(threads) {}

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
        else {
            prediction = 1;
        }

        return prediction;
    }

private:
    static double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y) {
        double l2 = 0.0;
        for (size_t i = 1; i < x.size(); i++) { // Start from index 1 to exclude outcome
            l2 += std::pow((x[i] - y[i]), 2);
        }
        return std::sqrt(l2);
    }

    static void* compute_distance(void* arg) {
        auto params = static_cast<ThreadParams*>(arg);
        const auto& x = *(params->x);
        const auto& range = *(params->range);
        const auto& y = *(params->y);

        std::vector<std::vector<double>>* l2_distances = new std::vector<std::vector<double>>;

        std::cout << "Thread computing distances for range: " << range.first << " to " << range.second << std::endl;

        for (int i = range.first; i < range.second; ++i) {
            const auto& sample = x[i];
            //std::cout << "Sample: ";
            //for (double value : sample) {
            //    std::cout << value << " ";
            //}
            //std::cout << std::endl;

            if (sample == y) {
                std::cout << "Skipping same point" << std::endl;
                continue;
            }

            std::vector<double> neighbour; // structure: distance, label
            double l2 = euclidean_distance(y, sample); // Call the static helper function
            neighbour.push_back(l2);
            neighbour.push_back(sample[0]); // Store outcome label
            l2_distances->push_back(neighbour);
        }

        return l2_distances;
    }



    std::vector<std::vector<double>> get_knn(const std::vector<std::vector<double>>& x, const std::vector<double>& y) {
        std::vector<std::vector<double>> l2_distances;

        std::vector<pthread_t> thread_ids;
        std::vector<ThreadParams> params;

        int rows_per_thread = x.size() / num_threads;

        for (int i = 0; i < num_threads; ++i) {
            int start = i * rows_per_thread;
            int end = start + rows_per_thread;
            std::cout << "Thread" << i << ": " << start << " " << end << std::endl;
            params.push_back({ &x, new std::pair<int, int>(start, end), &y });
            pthread_t thread_id;
            pthread_create(&thread_id, nullptr, compute_distance, &params[i]);
            thread_ids.push_back(thread_id);
        }

        for (auto& thread_id : thread_ids) {
            void* result;
            pthread_join(thread_id, &result);
            auto distances_ptr = static_cast<std::vector<std::vector<double>>*>(result);
            l2_distances.insert(l2_distances.end(), distances_ptr->begin(), distances_ptr->end());
            delete distances_ptr;
        }

        std::cout << "Number of euclidean run: " << l2_distances.size() << std::endl;

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

    std::vector<double> target = { 0.0,1.0,0.0,1.0,26.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,1.0,0.0,3.0,5.0,30.0,0.0,1.0,4.0,6.0,8.0 };

    int num_threads = 3; // Set the number of threads
    Knn knn(3, num_threads); // Use K=3

    int prediction = knn.predict_class(dataset, target);
    std::cout << "Prediction: " << prediction << std::endl;

    if (prediction == 0) {
        std::cout << "Predicted class: Negative" << std::endl;
    }
    else {
        std::cout << "Predicted class: Prediabetes or Diabetes" << std::endl;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    return 0;
}
