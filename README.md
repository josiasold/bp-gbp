# BP-GBP: An open source library for (generalized) belief propagation decoding of quantum codes

A C++ library implementing Belief Propagation in binary (`Bp2Decoder`), quaternary vectorial (`BpDecoder`) and quaternary scalar (`BpDecoderKL`) framework. The latter is inspired by Kuo-Lai [arXiv:2104.13659](https://arxiv.org/abs/2104.13659). An additional decoder uses Generalized Belief Propagation [YFM-GBP](https://ieeexplore.ieee.org/abstract/document/14590449) either as postprocessor (e.g. for sparse random graph codes) or as standalone decoder (e.g. for topological codes).

## Build Library

### Prerequsites
Make sure to have a C++17 compatible compiler and CMAKE at least Version 3.16.
Libraries used are
- [JSON for Modern C++](https://github.com/nlohmann/json) for I/O handling
- [NTL](https://github.com/libntl/ntl) for computation of GF(2) ranks
- [gmp](https://gmplib.org) in conjunction with NTL
- [xtensor](https://github.com/xtensor-stack/xtensor), [xframe](https://github.com/xtensor-stack/xframe) and their dependencies as data structure
- [lemon graph library](https://lemon.cs.elte.hu/trac/lemon) as graph structure
- [CPPItertools](https://github.com/ryanhaining/cppitertools )

Header only libraries included in the `include` directory should work right away, NTL with gmp, xtensor and lemon might have to be installed separately.


### Build
Then simply run from the repository root

```
mkdir build
cd build
cmake ..
make
```

## Install and Run Example Simulations

We provide examples illustrating the usage of the library. They can be found within the `sim` directory and are compiled with the library. In order to get the binaries to the simulation folder just run

```
make install
```

The examples are
- `sim.cpp` for quaternary vectorial BP, GBP and BP+GBP
- `sim_KL.cpp` for quaternary scalar BP, GBP and BP+GBP

### Run examples
The example scripts use input files to set the properties of decoding. Exemplary input files are given in `sim/input/input_files`.
The binaries use two command line parameters to specify the input file and the output directory, an exemplary run command is

```
./sim_KL input/input_files/input_test.json output
```

## Quantum Codes

Some codes are provided in `sim/input/codes`, in the [alist](http://www.inference.org.uk/mackay/codes/alist.html) format and numpy `.npy`.
- `random_irregular`,`random_regular`: codes constructed from random classical base codes with the hypergraph product construction ([arXiv:0903.0566](https://arxiv.org/abs/0903.0566))
- `surface`: topological surface codes (e.g. [arXiv:quant-ph/0110143](https://arxiv.org/abs/quant-ph/0110143))
- `xzzx`: twisted surface codes for biased channels ([arXiv:2009.07851](https://arxiv.org/abs/2009.07851))

