#ifndef IO_TOOLS_HPP_
#define IO_TOOLS_HPP_

#include <valarray>
#include <stdlib.h>
#include <iostream>

#include <string>
#include <sstream>

#include <bp_gbp/la_tools.hpp>

template <typename T>
void print_container(T &inp)
{
    for (size_t i = 0; i < inp.size(); i++)
    {
        std::cout << inp[i] << " ";
    }
    std::cout << std::endl;
}

template <typename T>
void print_container(T &inp, const char * description)
{
    std::cout << description << " =\n";
    for (size_t i = 0; i < inp.size(); i++)
    {
        std::cout << inp[i] << " ";
    }
    std::cout << std::endl << std::endl;;
}

template <typename T>
void print_container(T &inp, const char * description, bool weight)
{
    std::cout << "|" << description << "|" << " = " << hamming_weight(inp) << "\n";
    int c = 0;
    for (size_t i = 0; i < inp.size(); i++)
    {
        if (inp(i) != 0)
        {
            std::cout << i << ":" << inp(i);
            if (c <  hamming_weight(inp) - 1)
            {
                std::cout << " , ";
            }
            c++;
        }
    }
    std::cout << std::endl;
}


template <typename T>
std::string container_to_string(T &inp, const char * description, bool weight)
{
    std::stringstream ss;
    int w = hamming_weight(inp);
    ss << description << " |"<<  hamming_weight(inp) <<"|";
    if (w > 0)
    {
        ss << "\t = ";
        int c = 0;
        for (size_t i = 0; i < inp.size(); i++)
        {
            if (inp(i) != 0)
            {
                ss << i << ":" << inp(i);
                if (c <  hamming_weight(inp) - 1)
                {
                    ss << " , ";
                }
                c++;
            }
        }
    } 
    
    return ss.str();
}

// https://stackoverflow.com/questions/15006269/c-get-substring-before-a-certain-char
std::string strip_(std::string const& s);

#endif