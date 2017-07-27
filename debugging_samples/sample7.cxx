void fail()
{
    *(int*)0 = 0;
}

int main()
{
    fail();
    return 0;
}
