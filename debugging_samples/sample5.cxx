template <typename T> struct base
{
    template <typename U>
    U convert() const;
};

template <typename T>
struct foo : base<T>
{
    foo()
    {
        this->convert<double>();
    }
};
