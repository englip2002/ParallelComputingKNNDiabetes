#include <iostream>
#include <pthread.h>

// Function to be executed by each thread
void* threadFunction(void* arg) {
    int threadID = *(int*)arg;
    std::cout << "Thread " << threadID << " is running.\n" << std::endl;
    return nullptr; // Return a value (nullptr) to satisfy the function signature
}

int main() {
    const int numThreads = 4;
    pthread_t threads[numThreads];
    int threadIDs[numThreads];

    // Create threads
    for (int i = 0; i < numThreads; ++i) {
        threadIDs[i] = i;
        int result = pthread_create(&threads[i], NULL, threadFunction, &threadIDs[i]);
        if (result) {
            std::cerr << "Error creating thread " << i << std::endl;
            return -1;
        }
    }

    // Wait for threads to finish
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], NULL);
    }

    std::cout << "All threads have finished." << std::endl;

    return 0;
}
