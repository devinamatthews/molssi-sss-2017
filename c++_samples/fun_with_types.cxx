#include <iostream>
#include <type_traits>

// This is the main program
int main()
{
    int x = 2;
    std::cout << "x = " << x << '\n';

    // basic numerical types can be converted
    double y = x;
    std::cout << "y = " << y << '\n';

    // a change to a reference will change this original too
    int& xref = x;
    xref = 4;
    std::cout << "x = " << x << '\n';

    // a const reference can't be written to
    const int& xconstref = x;
    // this will cause a compiler error: try it!
    // xconstref = 7;

    // z will be an int because '3' is
    auto z = 3;
    if (std::is_same<decltype(z),int>::value)
        std::cout << "z is an int\n";
    else
        std::cout << "z is not an int\n";

    // w will be a double because '3.7' is
    auto w = 3.7;
    if (std::is_same<decltype(w),double>::value)
        std::cout << "w is a double\n";
    else
        std::cout << "w is not a double\n";

    // zref isn't an int but a reference to one
    auto& zref = z;
    if (std::is_same<decltype(zref),int>::value)
        std::cout << "zref is an int\n";
    else
        std::cout << "zref is not an int\n";

    return 0;
}
