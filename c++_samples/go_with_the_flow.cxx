#include <iostream>
#include <vector>

// This is the main program
int main()
{
    std::vector<int> squares = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};

    // print all of the squares up to 100
    std::cout << "\nRange-for loop\n";
    for (int square : squares)
    {
        std::cout << square << '\n';
    }

    // but I really don't like 36...
    std::cout << "\nRange-for loop\n";
    for (int& square : squares)
    {
        if (square == 36)
        {
            square = -1;
        }
        else
        {
            std::cout << square << '\n';
        }
    }

    // now print them with an old-style for loop
    std::cout << "\nFor loop\n";
    for (int i = 0;i < squares.size();i++)
    {
        std::cout << squares[i] << '\n';
    }

    // maybe we can print them backwards with a while loop?
    std::cout << "\nWhile loop\n";
    int i = squares.size()-1;
    while (i >= 0)
    {
        std::cout << squares[i] << '\n';
        i--;
    }

    return 0;
}
