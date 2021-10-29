```bash
build && cmake -DGLIBCXX_USE_CXX11_ABI=ON .. && make -j $(nproc) main && ./main && build && cmake -DGLIBCXX_USE_CXX11_ABI=OFF .. && make -j $(nproc) main && ./main
```
