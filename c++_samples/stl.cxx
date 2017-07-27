#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>

// return a std::vector that lives on the heap
std::unique_ptr<std::vector<double>> make_array(int n)
{
    return std::unique_ptr<std::vector<double>>(new std::vector<double>(n));
}

// sort an array
void sort_array(std::vector<double>& array)
{
    std::sort(array.begin(), array.end());
}

// print an array when streamed (<<'ed) to std::cout
std::ostream& operator<<(std::ostream& os, const std::vector<double>& array)
{
    os << '[';
    for (double element : array) os << element << ", ";
    os << ']';
    return os;
}

// This is the main program
int main()
{
    // Whatever "auto" is acts like std::vector<double>*
    auto array_ptr = make_array(10);

    *array_ptr = {9, 5, 2, 4, 0, 13, 5, 3, 7, 3};

    sort_array(*array_ptr);

    std::cout << *array_ptr << '\n';

    return 0;
}
