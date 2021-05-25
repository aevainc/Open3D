def codegen_dispatch_elems(n):
    print('#define DISPATCH_ELEMS_TO_N(ELEMS, ...) \\')
    print('    switch (ELEMS) { \\')
    for i in range(1, n):
        print('    case {}: {{\\'.format(i))
        print('        const int N = {}; \\'.format(i))
        print('        return __VA_ARGS__(); \\')
        print('    }\\')
    print('        default: {\\')
    print('            std::cerr << \"Unsupported elements:\" << ELEMS << \"\\n\"; \\')
    print('        }\\')
    print('    }')


codegen_dispatch_elems(4096)
