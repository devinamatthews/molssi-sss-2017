#include <iostream>

namespace secrets
{

int meaning_of_life()
{
    return 42;
}

}

void print_variable(const std::string& name, int value)
{
    std::cout << name << "(int) = " << value << '\n';
}

void print_variable(const std::string& name, double value)
{
    std::cout << name << "(double) = " << value << '\n';
}

// This is the main program
int main()
{
    print_variable("the meaning", secrets::meaning_of_life());

    print_variable("pi", 3.141592654);

    // this won't compile (try it!) because it can't choose
    // which of int or double is a closer match for float
    //print_variable("less accurate pi", 3l); // l for long

    // but this works because float can always be converted
    // exactly to a double
    print_variable("mid-accuracy pi", 3.1416); // f for float

    return 0;
}
