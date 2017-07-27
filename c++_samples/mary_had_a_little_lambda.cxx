#include <iostream>

// This is the main program
int main()
{
    int start = 0;
    int stop = 5;
    auto do_loop = [&,stop]
    {
        for (int i = start;i < stop;i++)
        {
            std::cout << i << '\n';
        }
        std::cout << '\n';
    };

    // use the initial settings
    do_loop();

    // change the start
    start = 2;
    do_loop();

    // change the stop (no effect, do_loop has a copy)
    stop = 100;
    do_loop();

    auto add = [](int x, int y) { return x+y; };
    std::cout << "Should be 7: " << add(3, 4) << '\n';

    return 0;
}
