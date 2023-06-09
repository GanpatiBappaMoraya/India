#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Function to find the minimum value
int findMin(const int* data, int size)
{
    int minValue = data[0];

    #pragma omp parallel for reduction(min: minValue)
    for (int i = 1; i < size; ++i)
    {
        if (data[i] < minValue)
            minValue = data[i];
    }

    return minValue;
}

// Function to find the maximum value
int findMax(const int* data, int size)
{
    int maxValue = data[0];

    #pragma omp parallel for reduction(max: maxValue)
    for (int i = 1; i < size; ++i)
    {
        if (data[i] > maxValue)
            maxValue = data[i];
    }

    return maxValue;
}

// Function to calculate the sum of values
int calculateSum(const int* data, int size)
{
    int sum = 0;

    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < size; ++i)
    {
        sum += data[i];
    }

    return sum;
}

// Function to calculate the average of values
double calculateAverage(const int* data, int size)
{
    int sum = calculateSum(data, size);
    return static_cast<double>(sum) / size;
}

int main()
{
    int data[] = {5, 2, 8, 1, 6, 9, 3, 7, 4};
    int size = sizeof(data) / sizeof(data[0]);

    int minValue = findMin(data, size);
    int maxValue = findMax(data, size);
    int sum = calculateSum(data, size);
    double average = calculateAverage(data, size);

    cout << "Min Value: " << minValue << endl;
    cout << "Max Value: " << maxValue << endl;
    cout << "Sum: " << sum << endl;
    cout << "Average: " << average << endl;

    return 0;
}
