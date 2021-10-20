#include <bp_gbp/io_tools.hpp>

std::string strip_(std::string const& s)
{
    std::string::size_type pos = s.find('_');
    if (pos != std::string::npos)
    {
        return s.substr(0, pos);
    }
    else
    {
        return s;
    }
}
