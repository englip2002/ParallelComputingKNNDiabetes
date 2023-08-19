#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include "taskflow/taskflow/taskflow.hpp"

// ... (include headers and class definitions here)

int main() {
    // ... (read dataset and prepare target vectors here)
    std::string filename = "diabetes_binary_5050.csv"; // Replace with your CSV file name

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    tf::Taskflow taskflow;
    tf::Executor executor;

    Knn knn(3); // Use K=3

    // Function to execute KNN prediction for a single target vector
    auto knn_predict_task = [&](const std::vector<double>& target) {
        return [&, target]() {
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
        };
    };

    for (const auto& target : targetVectors) {
        taskflow.emplace(knn_predict_task(target));
    }

    executor.run(taskflow).wait();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    return 0;
}
