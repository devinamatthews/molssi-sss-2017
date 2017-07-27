struct member
{
    int x = 0;
    member() {}
    member(int x) : x(x) {}
};

struct foo
{
    member m(4);
    foo() {}
    foo(const member& m) : m(m) {}
};
