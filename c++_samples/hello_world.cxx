#include <iostream>
#include <string>
#include <vector>

// This is the main program
int main()
{
    std::vector<std::string> continents =
        {"N. America", "S. America",
         "Europe", "Asia", "Africa",
         "Australia", "Antarctica"};

    for (int i = 0;i < continents.size();i++)
    {
        std::string& continent = continents[i];
        std::cout << "Hello " << continent << "!\n";
    }

    return 0;
}
